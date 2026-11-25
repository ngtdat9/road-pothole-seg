"""Pseudo-label road on pothole-only images with the road teacher (pipeline stage 3).

Three steps, run in order:
  predict (Colab) teacher best.pt -> road polygons for the train and val images of
                  sources without real road labels (split.csv real_road = 0). Writes
                  <pseudo>/labels/<name>.txt (road lines only) and <pseudo>/pseudo_road.csv.
                  Test images are never read.
  review  (Mac)   contact sheets of `review_n` images (every flagged one plus a seeded
                  random sample) in outputs/pseudo_review/, and docs/pseudo_review.csv
                  with a `keep` column (1 = accept). Set keep to 0 for bad pseudo road.
  build   (Mac)   copies merged-v1-base to merged-v1, appends the pseudo road lines to
                  the label files and removes images with keep = 0 (a visible road must
                  never stay unlabelled). `--zip` writes merged-v1.zip for Drive.

An image is flagged when it gets no road, or when the road covers less than
`min_pothole_cover` of its pothole area (potholes lie on the road).
Settings: `pseudo_road` in configs/merge.yaml.

Usage:
    python -m training.scripts.pseudo_label predict --weights <best.pt> \
        --data-root /content/merged-v1-base --out <drive>/pseudo/teacher     # Colab
    python -m training.scripts.pseudo_label review                           # Mac
    python -m training.scripts.pseudo_label build --zip                      # Mac
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from training.scripts.inspect_dataset import parse_label_file, polygon_area
from training.scripts.merge_datasets import REPO_ROOT, _resolve, format_label

SPLITS = ("train", "val", "test")
PSEUDO_SPLITS = SPLITS[:2]  # never test
PSEUDO_CSV = "pseudo_road.csv"
PSEUDO_FIELDS = ["name", "file", "split", "n_road", "road_frac", "mean_conf",
                 "pothole_cover", "flagged"]  # fmt: skip
REVIEW_FIELDS = ["idx", "name", "split", "flagged", "pothole_cover", "mean_conf", "keep"]
COVER_GRID = 256  # raster size for the pothole-on-road check
TILE = 320  # px, each review image is fitted into a TILE x TILE square
CAPTION_H = 24
SHEET_COLS, SHEET_ROWS = 5, 2
ROAD_COLOUR = (80, 200, 80)
POTHOLE_COLOUR = (40, 40, 230)
FLAG_COLOUR = (0, 165, 255)


@dataclass
class PseudoSettings:
    base_dir: Path
    out_dir: Path
    review_csv: Path
    imgsz: int
    conf: float
    min_area_frac: float
    min_pothole_cover: float
    review_n: int
    seed: int


@dataclass
class Target:
    name: str
    file: str
    split: str


def load_settings(config_path: Path) -> PseudoSettings:
    with config_path.open() as f:
        raw = yaml.safe_load(f)["pseudo_road"]
    return PseudoSettings(
        base_dir=_resolve(raw["base_dir"]),
        out_dir=_resolve(raw["out_dir"]),
        review_csv=_resolve(raw["review_csv"]),
        imgsz=int(raw["imgsz"]),
        conf=float(raw["conf"]),
        min_area_frac=float(raw["min_area_frac"]),
        min_pothole_cover=float(raw["min_pothole_cover"]),
        review_n=int(raw["review_n"]),
        seed=int(raw["seed"]),
    )


def class_ids(data_root: Path) -> tuple[int, int]:
    """(road id, pothole id) from the dataset's data.yaml."""
    names = yaml.safe_load((data_root / "data.yaml").read_text())["names"]
    ids = {v: int(k) for k, v in names.items()}
    return ids["road"], ids["pothole"]


def _list_stems(path: Path) -> set[str]:
    return {Path(line).stem for line in path.read_text().split()}


def load_targets(data_root: Path) -> list[Target]:
    """Train and val images of sources without real road labels, in list order."""
    with (data_root / "split.csv").open() as f:
        wanted = {r["name"]: r["split"] for r in csv.DictReader(f) if r["real_road"] == "0"}
    targets = []
    for split in PSEUDO_SPLITS:
        for line in (data_root / f"{split}.txt").read_text().split():
            file = Path(line).name
            name = Path(file).stem
            if wanted.get(name) == split:
                targets.append(Target(name, file, split))
    leaked = {t.name for t in targets} & _list_stems(data_root / "test.txt")
    if leaked:
        raise AssertionError(f"test images among pseudo-label targets: {sorted(leaked)[:5]}")
    return targets


# ---------------------------------------------------------------- predict


def road_polygons(
    masks: np.ndarray,
    classes: np.ndarray,
    confs: np.ndarray,
    road_id: int,
    s: PseudoSettings,
) -> list[tuple[np.ndarray, float]]:
    """Normalised (polygon, confidence) for each confident road piece of enough area.

    `masks` is (N, H, W) at the original image size. Each outer contour of a road
    mask becomes one polygon (YOLO polygons have no holes).
    """
    out = []
    for mask, cls, conf in zip(masks, classes, confs, strict=True):
        if int(cls) != road_id or float(conf) < s.conf:
            continue
        binary = (np.asarray(mask) > 0.5).astype(np.uint8)
        h, w = binary.shape
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2).astype(float)
            if len(pts) < 3:
                continue
            poly = pts / [w, h]
            if polygon_area(poly) >= s.min_area_frac:
                out.append((poly, float(conf)))
    return out


def _raster(polygons: list[np.ndarray], size: int = COVER_GRID) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [np.round(poly * size).astype(np.int32)], 1)
    return mask.astype(bool)


def pothole_cover(road: list[np.ndarray], potholes: list[np.ndarray]) -> float | None:
    """Share of pothole area inside the road; None when the image has no pothole."""
    if not potholes:
        return None
    p = _raster(potholes)
    n = int(p.sum())
    return float((p & _raster(road)).sum() / n) if n else None


def _numpy(x: Any) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _potholes(label_file: Path, pothole_id: int) -> list[np.ndarray]:
    instances = parse_label_file(label_file)
    return [i.polygon for i in instances if i.class_id == pothole_id and not i.box_only]


def predict(model: Any, data_root: Path, out_dir: Path, s: PseudoSettings) -> list[dict]:
    """Run the teacher on every target and write road-only label files plus the CSV.

    `model` is an Ultralytics YOLO (or a test fake with the same `predict`).
    """
    road_id, pothole_id = class_ids(data_root)
    labels_dir = out_dir / "labels"
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    labels_dir.mkdir(parents=True)
    rows = []
    for t in load_targets(data_root):
        result = model.predict(
            str(data_root / "images" / t.file),
            imgsz=s.imgsz,
            conf=s.conf,
            retina_masks=True,  # masks at the original image size
            verbose=False,
        )[0]
        if result.masks is None:
            road = []
        else:
            road = road_polygons(
                _numpy(result.masks.data),
                _numpy(result.boxes.cls),
                _numpy(result.boxes.conf),
                road_id,
                s,
            )
        polys = [p for p, _ in road]
        cover = pothole_cover(polys, _potholes(data_root / "labels" / f"{t.name}.txt", pothole_id))
        flagged = not polys or (cover is not None and cover < s.min_pothole_cover)
        (labels_dir / f"{t.name}.txt").write_text(format_label([(road_id, p) for p in polys]))
        rows.append(
            {
                "name": t.name,
                "file": t.file,
                "split": t.split,
                "n_road": len(polys),
                "road_frac": round(min(1.0, sum(polygon_area(p) for p in polys)), 4),
                "mean_conf": round(float(np.mean([c for _, c in road])), 3) if road else "",
                "pothole_cover": "" if cover is None else round(cover, 3),
                "flagged": int(flagged),
            }
        )
    with (out_dir / PSEUDO_CSV).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PSEUDO_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ---------------------------------------------------------------- review


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def choose_review(rows: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    """Every flagged image, topped up to `n` with a seeded random sample of the rest."""
    flagged = sorted((r for r in rows if r["flagged"] == "1"), key=lambda r: r["name"])
    rest = sorted((r for r in rows if r["flagged"] != "1"), key=lambda r: r["name"])
    k = min(max(0, n - len(flagged)), len(rest))
    return flagged + sorted(random.Random(seed).sample(rest, k), key=lambda r: r["name"])


def draw_review(
    image: np.ndarray,
    road: list[np.ndarray],
    potholes: list[np.ndarray],
    caption: str,
    flagged: bool,
) -> np.ndarray:
    """Road tinted green, potholes outlined red, fitted into a captioned tile."""
    h, w = image.shape[:2]
    overlay = image.copy()
    road_px = [np.round(p * [w, h]).astype(np.int32) for p in road]
    cv2.fillPoly(overlay, road_px, ROAD_COLOUR)
    out = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    cv2.polylines(out, road_px, True, ROAD_COLOUR, max(1, w // 320))
    hole_px = [np.round(p * [w, h]).astype(np.int32) for p in potholes]
    cv2.polylines(out, hole_px, True, POTHOLE_COLOUR, max(2, w // 200))
    r = TILE / max(h, w)
    small = cv2.resize(out, (max(1, round(w * r)), max(1, round(h * r))))
    tile = np.full((TILE + CAPTION_H, TILE, 3), 255, np.uint8)
    tile[CAPTION_H : CAPTION_H + small.shape[0], : small.shape[1]] = small
    if flagged:
        tile[:CAPTION_H] = FLAG_COLOUR
    cv2.putText(tile, caption, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def write_sheets(tiles: list[np.ndarray], out_dir: Path) -> list[Path]:
    per_sheet = SHEET_COLS * SHEET_ROWS
    blank = np.full_like(tiles[0], 255) if tiles else None
    paths = []
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start : start + per_sheet]
        chunk += [blank] * (per_sheet - len(chunk))
        rows = [np.hstack(chunk[r * SHEET_COLS : (r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
        path = out_dir / f"sheet_{start // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows))
        paths.append(path)
    return paths


def review(
    pseudo_dir: Path, s: PseudoSettings, out_dir: Path, force: bool = False
) -> list[dict[str, str]]:
    """Draw the review sheets and write the review CSV (keep = 1 for every image)."""
    if s.review_csv.exists() and not force:
        raise SystemExit(f"{s.review_csv} exists (it may hold your decisions); use --force")
    road_id, pothole_id = class_ids(s.base_dir)
    chosen = choose_review(read_csv(pseudo_dir / PSEUDO_CSV), s.review_n, s.seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    tiles, records = [], []
    for idx, row in enumerate(chosen, start=1):
        image = cv2.imread(str(s.base_dir / "images" / row["file"]))
        road = [i.polygon for i in parse_label_file(pseudo_dir / "labels" / f"{row['name']}.txt")]
        holes = _potholes(s.base_dir / "labels" / f"{row['name']}.txt", pothole_id)
        cover = row["pothole_cover"] or "-"
        caption = f"#{idx} cover {cover} conf {row['mean_conf'] or '-'}"
        tiles.append(draw_review(image, road, holes, caption, row["flagged"] == "1"))
        records.append({**{k: row[k] for k in REVIEW_FIELDS[1:-1]}, "idx": idx, "keep": 1})
    write_sheets(tiles, out_dir)
    s.review_csv.parent.mkdir(parents=True, exist_ok=True)
    with s.review_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return records


# ---------------------------------------------------------------- build


def build(pseudo_dir: Path, s: PseudoSettings, make_zip: bool = False) -> dict:
    """merged-v1 = merged-v1-base + pseudo road lines, minus rejected images."""
    if not s.review_csv.exists():
        raise SystemExit(f"{s.review_csv} not found: run `review` and check the sheets first")
    road_id, _ = class_ids(s.base_dir)
    rows = read_csv(pseudo_dir / PSEUDO_CSV)
    rejected = {r["name"] for r in read_csv(s.review_csv) if r["keep"].strip() == "0"}
    test = _list_stems(s.base_dir / "test.txt")
    if {r["name"] for r in rows} & test:
        raise AssertionError("pseudo labels found for test images")

    out = s.out_dir
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(s.base_dir, out)
    for row in rows:
        name = row["name"]
        label = out / "labels" / f"{name}.txt"
        if name in rejected:
            label.unlink(missing_ok=True)
            (out / "images" / row["file"]).unlink(missing_ok=True)
            continue
        existing = label.read_text() if label.exists() else ""
        if any(line.split()[0] == str(road_id) for line in existing.splitlines() if line.strip()):
            raise ValueError(f"{name} already has road labels; refusing to add pseudo road")
        road_lines = (pseudo_dir / "labels" / f"{name}.txt").read_text()
        if existing and not existing.endswith("\n"):
            existing += "\n"
        label.write_text(existing + road_lines)

    for list_file in out.glob("*.txt"):
        lines = list_file.read_text().split()
        kept = [line for line in lines if Path(line).stem not in rejected]
        list_file.write_text("\n".join(kept) + "\n")
    split_rows = [r for r in read_csv(out / "split.csv") if r["name"] not in rejected]
    with (out / "split.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    data = yaml.safe_load((out / "data.yaml").read_text())
    (out / "data.yaml").write_text(
        "# Paths are relative to this file. Road on pothole-only (gn_) train/val images\n"
        "# is pseudo-labelled by the road teacher; test has none. Road metrics come from\n"
        "# val_real_road.txt / test_real_road.txt only. See docs/datasets.md.\n"
        + yaml.safe_dump(data, sort_keys=False)
    )
    shutil.copy2(pseudo_dir / PSEUDO_CSV, out / PSEUDO_CSV)

    labelled = [r for r in rows if r["name"] not in rejected]
    summary = {
        "pseudo_labelled": len(labelled),
        "pseudo_labelled_per_split": {
            sp: sum(r["split"] == sp for r in labelled) for sp in PSEUDO_SPLITS
        },
        "without_road": sum(r["n_road"] == "0" for r in labelled),
        "flagged": sum(r["flagged"] == "1" for r in rows),
        "rejected": sorted(rejected),
        "images": {sp: len((out / f"{sp}.txt").read_text().split()) for sp in SPLITS},
    }
    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["pseudo_road"] = summary
    report_path.write_text(json.dumps(report, indent=2))
    if make_zip:
        shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    return summary


# ---------------------------------------------------------------- CLI


def _default_pseudo_dir() -> Path:
    from training.scripts.collect_runs import find_drive_root

    return find_drive_root() / "pseudo" / "teacher"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "merge.yaml")
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("predict", help="Colab: run the teacher")
    p.add_argument("--weights", type=Path, required=True, help="teacher best.pt")
    p.add_argument("--data-root", type=Path, required=True, help="unzipped merged-v1-base")
    p.add_argument("--out", type=Path, required=True, help="pseudo folder, e.g. on Drive")
    r = sub.add_parser("review", help="Mac: draw review sheets")
    r.add_argument("--pseudo-dir", type=Path, help="default: <synced Drive>/pseudo/teacher")
    r.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "pseudo_review")
    r.add_argument("--force", action="store_true", help="overwrite the review CSV")
    b = sub.add_parser("build", help="Mac: write merged-v1")
    b.add_argument("--pseudo-dir", type=Path, help="default: <synced Drive>/pseudo/teacher")
    b.add_argument("--zip", action="store_true", help="also write <out_dir>.zip")
    args = parser.parse_args()
    s = load_settings(args.config)

    if args.step == "predict":
        from ultralytics import YOLO  # only on the training machine

        args.out.mkdir(parents=True, exist_ok=True)
        rows = predict(YOLO(str(args.weights)), args.data_root, args.out, s)
        flagged = sum(r["flagged"] for r in rows)
        no_road = sum(r["n_road"] == 0 for r in rows)
        print(f"{len(rows)} images labelled, {no_road} without road, {flagged} flagged")
    elif args.step == "review":
        records = review(args.pseudo_dir or _default_pseudo_dir(), s, args.out, args.force)
        flagged = sum(r["flagged"] == "1" for r in records)
        print(f"{len(records)} images ({flagged} flagged) in {args.out}; decisions: {s.review_csv}")
    else:
        print(json.dumps(build(args.pseudo_dir or _default_pseudo_dir(), s, args.zip), indent=2))


if __name__ == "__main__":
    main()
