"""Pseudo-label road on pothole-only images with the road teacher (pipeline stage 3).

Steps, run in order:
  predict (Colab) teacher best.pt -> road polygons for the train and val images of
                  sources without real road labels (split.csv real_road = 0). Labels are
                  written on Colab's local disk, then ONE labels.zip plus pseudo_road.csv
                  go to --out on Drive (Drive syncs thousands of small files badly).
                  Test images are never read.
  review  (Mac)   fetches the zip from the synced Drive into `local_dir`, checks every
                  pothole label against the pseudo road, and writes:
                  - review sheets (every flagged image + a seeded random sample) and
                    docs/pseudo_review.csv with keep = 1; set keep = 0 for bad road,
                  - off-road sheets and docs/pseudo_offroad.csv with drop = 1 for pothole
                    labels that lie off the road (GNSoft2 labels some vehicles as
                    potholes); set drop = 0 to keep one.
  build   (Mac)   merged-v1 = merged-v1-base + pseudo road lines, minus rejected images
                  and dropped pothole labels. `--zip` writes merged-v1.zip for Drive.

Per pothole label, `cover` = share of its area inside the pseudo road.
- Off-road (dropped): cover < offroad_max_cover and area >= offroad_min_area.
- Flagged image: no road, or a pothole of area >= flag_min_area with
  cover < min_pothole_cover that is not off-road (the road may have missed it).
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
import tempfile
import zipfile
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
LABELS_ZIP = "labels.zip"
PSEUDO_FIELDS = ["name", "file", "split", "n_road", "road_frac", "mean_conf"]
REVIEW_FIELDS = ["idx", "name", "split", "flagged", "min_cover", "mean_conf", "keep"]
OFFROAD_FIELDS = ["idx", "name", "line", "area", "cover", "drop"]
COVER_GRID = 512  # raster size for the pothole-on-road check
TILE = 320  # px, each review image is fitted into a TILE x TILE square
CAPTION_H = 24
SHEET_COLS, SHEET_ROWS = 5, 2
ROAD_COLOUR = (80, 200, 80)
POTHOLE_COLOUR = (40, 40, 230)
OFFROAD_COLOUR = (230, 40, 230)
FLAG_COLOUR = (0, 165, 255)


@dataclass
class PseudoSettings:
    base_dir: Path
    out_dir: Path
    local_dir: Path
    review_csv: Path
    offroad_csv: Path
    imgsz: int
    conf: float
    min_area_frac: float
    min_pothole_cover: float
    flag_min_area: float
    offroad_max_cover: float
    offroad_min_area: float
    review_n: int
    seed: int


@dataclass
class Target:
    name: str
    file: str
    split: str


@dataclass
class HoleCheck:
    """One pothole label of an image, checked against that image's pseudo road."""

    name: str
    line: int  # index of the line in the label file
    area: float
    cover: float
    polygon: np.ndarray


def load_settings(config_path: Path) -> PseudoSettings:
    with config_path.open() as f:
        raw = yaml.safe_load(f)["pseudo_road"]
    paths = ("base_dir", "out_dir", "local_dir", "review_csv", "offroad_csv")
    return PseudoSettings(
        **{k: _resolve(raw[k]) for k in paths},
        imgsz=int(raw["imgsz"]),
        conf=float(raw["conf"]),
        min_area_frac=float(raw["min_area_frac"]),
        min_pothole_cover=float(raw["min_pothole_cover"]),
        flag_min_area=float(raw["flag_min_area"]),
        offroad_max_cover=float(raw["offroad_max_cover"]),
        offroad_min_area=float(raw["offroad_min_area"]),
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- predict (Colab)


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


def _numpy(x: Any) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def predict(model: Any, data_root: Path, out_dir: Path, s: PseudoSettings) -> list[dict]:
    """Run the teacher on every target; write road-only label files and the CSV.

    `model` is an Ultralytics YOLO (or a test fake with the same `predict`).
    """
    road_id, _ = class_ids(data_root)
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
        road = []
        if result.masks is not None:
            road = road_polygons(
                _numpy(result.masks.data),
                _numpy(result.boxes.cls),
                _numpy(result.boxes.conf),
                road_id,
                s,
            )
        polys = [p for p, _ in road]
        (labels_dir / f"{t.name}.txt").write_text(format_label([(road_id, p) for p in polys]))
        rows.append(
            {
                "name": t.name,
                "file": t.file,
                "split": t.split,
                "n_road": len(polys),
                "road_frac": round(min(1.0, sum(polygon_area(p) for p in polys)), 4),
                "mean_conf": round(float(np.mean([c for _, c in road])), 3) if road else "",
            }
        )
    write_csv(out_dir / PSEUDO_CSV, PSEUDO_FIELDS, rows)
    return rows


def pack(work_dir: Path, out_dir: Path) -> None:
    """Copy the CSV and ONE labels.zip from the local work dir to `out_dir` (Drive)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(out_dir / Path(LABELS_ZIP).stem), "zip", root_dir=work_dir / "labels")
    shutil.copy2(work_dir / PSEUDO_CSV, out_dir / PSEUDO_CSV)


def fetch(drive_dir: Path, local_dir: Path) -> int:
    """Unpack <drive_dir>/labels.zip and the CSV into `local_dir`; returns the label count."""
    if local_dir.exists():
        shutil.rmtree(local_dir)
    (local_dir / "labels").mkdir(parents=True)
    shutil.copy2(drive_dir / PSEUDO_CSV, local_dir / PSEUDO_CSV)
    with zipfile.ZipFile(drive_dir / LABELS_ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".txt") and "/" not in n.strip("/")]
        for n in names:  # flat .txt files only: never extract paths from the archive
            (local_dir / "labels" / n).write_bytes(z.read(n))
    expected = len(read_csv(local_dir / PSEUDO_CSV))
    if len(names) != expected:
        raise SystemExit(f"{LABELS_ZIP} has {len(names)} labels, the CSV lists {expected}")
    return len(names)


# ---------------------------------------------------------------- checks (Mac)


def _raster(polygons: list[np.ndarray], size: int = COVER_GRID) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [np.round(poly * size).astype(np.int32)], 1)
    return mask.astype(bool)


def pothole_cover(road: list[np.ndarray], potholes: list[np.ndarray]) -> float | None:
    """Share of pothole area inside the road; None when there is no pothole."""
    if not potholes:
        return None
    p = _raster(potholes)
    n = int(p.sum())
    return float((p & _raster(road)).sum() / n) if n else None


def check_holes(
    name: str, road: list[np.ndarray], label_file: Path, pothole_id: int
) -> list[HoleCheck]:
    """Cover of every pothole polygon in one label file."""
    road_mask = _raster(road)
    checks = []
    for line, inst in enumerate(parse_label_file(label_file)):
        if inst.class_id != pothole_id or inst.box_only:
            continue
        hole = _raster([inst.polygon])
        n = int(hole.sum())
        cover = float((hole & road_mask).sum() / n) if n else 1.0
        checks.append(HoleCheck(name, line, polygon_area(inst.polygon), cover, inst.polygon))
    return checks


def is_offroad(h: HoleCheck, s: PseudoSettings) -> bool:
    return h.cover < s.offroad_max_cover and h.area >= s.offroad_min_area


def analyse(local_dir: Path, s: PseudoSettings) -> tuple[list[dict], dict[str, list[HoleCheck]]]:
    """Pseudo CSV rows with `flagged` and `min_cover` added, and each image's hole checks."""
    _, pothole_id = class_ids(s.base_dir)
    rows = read_csv(local_dir / PSEUDO_CSV)
    holes: dict[str, list[HoleCheck]] = {}
    for row in rows:
        name = row["name"]
        road = [i.polygon for i in parse_label_file(local_dir / "labels" / f"{name}.txt")]
        checks = check_holes(name, road, s.base_dir / "labels" / f"{name}.txt", pothole_id)
        holes[name] = checks
        missed = [
            h
            for h in checks
            if h.area >= s.flag_min_area and h.cover < s.min_pothole_cover and not is_offroad(h, s)
        ]
        row["flagged"] = int(not road or bool(missed))
        row["min_cover"] = round(min((h.cover for h in checks), default=1.0), 3)
    return rows, holes


# ---------------------------------------------------------------- review (Mac)


def choose_review(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Every flagged image, topped up to `n` with a seeded random sample of the rest."""
    flagged = sorted((r for r in rows if int(r["flagged"])), key=lambda r: r["name"])
    rest = sorted((r for r in rows if not int(r["flagged"])), key=lambda r: r["name"])
    k = min(max(0, n - len(flagged)), len(rest))
    return flagged + sorted(random.Random(seed).sample(rest, k), key=lambda r: r["name"])


def draw_tile(
    image: np.ndarray,
    road: list[np.ndarray],
    holes: list[tuple[np.ndarray, tuple[int, int, int]]],
    caption: str,
    flagged: bool,
) -> np.ndarray:
    """Road tinted green, each hole outlined in its colour, fitted into a captioned tile."""
    h, w = image.shape[:2]
    overlay = image.copy()
    road_px = [np.round(p * [w, h]).astype(np.int32) for p in road]
    cv2.fillPoly(overlay, road_px, ROAD_COLOUR)
    out = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    cv2.polylines(out, road_px, True, ROAD_COLOUR, max(1, w // 320))
    for poly, colour in holes:
        px = np.round(poly * [w, h]).astype(np.int32)
        cv2.polylines(out, [px], True, colour, max(2, w // 200))
    r = TILE / max(h, w)
    small = cv2.resize(out, (max(1, round(w * r)), max(1, round(h * r))))
    tile = np.full((TILE + CAPTION_H, TILE, 3), 255, np.uint8)
    tile[CAPTION_H : CAPTION_H + small.shape[0], : small.shape[1]] = small
    if flagged:
        tile[:CAPTION_H] = FLAG_COLOUR
    cv2.putText(tile, caption, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def write_sheets(tiles: list[np.ndarray], out_dir: Path, prefix: str) -> list[Path]:
    per_sheet = SHEET_COLS * SHEET_ROWS
    paths = []
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start : start + per_sheet]
        chunk += [np.full_like(tiles[0], 255)] * (per_sheet - len(chunk))
        rows = [np.hstack(chunk[r * SHEET_COLS : (r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
        path = out_dir / f"{prefix}_{start // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows))
        paths.append(path)
    return paths


def _road(local_dir: Path, name: str) -> list[np.ndarray]:
    return [i.polygon for i in parse_label_file(local_dir / "labels" / f"{name}.txt")]


def review(local_dir: Path, s: PseudoSettings, out_dir: Path, force: bool = False) -> dict:
    """Draw review and off-road sheets; write both decision CSVs (all accepted)."""
    for path in (s.review_csv, s.offroad_csv):
        if path.exists() and not force:
            raise SystemExit(f"{path} exists (it may hold your decisions); use --force")
    rows, holes = analyse(local_dir, s)
    by_name = {r["name"]: r for r in rows}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    tiles, records = [], []
    for idx, row in enumerate(choose_review(rows, s.review_n, s.seed), start=1):
        image = cv2.imread(str(s.base_dir / "images" / row["file"]))
        outlines = [
            (h.polygon, OFFROAD_COLOUR if is_offroad(h, s) else POTHOLE_COLOUR)
            for h in holes[row["name"]]
        ]
        caption = f"#{idx} cover {row['min_cover']} conf {row['mean_conf'] or '-'}"
        road = _road(local_dir, row["name"])
        tiles.append(draw_tile(image, road, outlines, caption, bool(row["flagged"])))
        records.append({**row, "idx": idx, "keep": 1})
    write_sheets(tiles, out_dir, "review")
    write_csv(s.review_csv, REVIEW_FIELDS, records)

    # Images without road say nothing about where potholes are; they are reviewed instead.
    with_road = sorted(n for n in holes if by_name[n]["n_road"] != "0")
    offroad = [h for name in with_road for h in holes[name] if is_offroad(h, s)]
    tiles, off_records = [], []
    for idx, h in enumerate(offroad, start=1):
        image = cv2.imread(str(s.base_dir / "images" / by_name[h.name]["file"]))
        caption = f"#{idx} cover {h.cover:.2f} area {h.area:.3f}"
        road = _road(local_dir, h.name)
        tiles.append(draw_tile(image, road, [(h.polygon, OFFROAD_COLOUR)], caption, False))
        off_records.append(
            {
                "idx": idx,
                "name": h.name,
                "line": h.line,
                "area": round(h.area, 4),
                "cover": round(h.cover, 3),
                "drop": 1,
            }  # fmt: skip
        )
    if tiles:
        write_sheets(tiles, out_dir, "offroad")
    write_csv(s.offroad_csv, OFFROAD_FIELDS, off_records)
    return {
        "images": len(rows),
        "without_road": sum(r["n_road"] == "0" for r in rows),
        "flagged": sum(r["flagged"] for r in rows),
        "reviewed": len(records),
        "offroad_labels": len(off_records),
        "offroad_images": len({r["name"] for r in off_records}),
        "pothole_labels": sum(len(v) for v in holes.values()),
    }


# ---------------------------------------------------------------- build (Mac)


def build(local_dir: Path, s: PseudoSettings, make_zip: bool = False) -> dict:
    """merged-v1 = base + pseudo road lines, minus rejected images and off-road labels."""
    for path in (s.review_csv, s.offroad_csv):
        if not path.exists():
            raise SystemExit(f"{path} not found: run `review` and check the sheets first")
    road_id, _ = class_ids(s.base_dir)
    rows = read_csv(local_dir / PSEUDO_CSV)
    rejected = {r["name"] for r in read_csv(s.review_csv) if r["keep"].strip() == "0"}
    drop_lines: dict[str, set[int]] = {}
    for r in read_csv(s.offroad_csv):
        if r["drop"].strip() == "1":
            drop_lines.setdefault(r["name"], set()).add(int(r["line"]))
    if {r["name"] for r in rows} & _list_stems(s.base_dir / "test.txt"):
        raise AssertionError("pseudo labels found for test images")

    out = s.out_dir
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(s.base_dir, out)
    dropped = 0
    for row in rows:
        name = row["name"]
        label = out / "labels" / f"{name}.txt"
        if name in rejected:
            label.unlink(missing_ok=True)
            (out / "images" / row["file"]).unlink(missing_ok=True)
            continue
        lines = [ln for ln in label.read_text().splitlines() if ln.strip()]
        if any(ln.split()[0] == str(road_id) for ln in lines):
            raise ValueError(f"{name} already has road labels; refusing to add pseudo road")
        skip = drop_lines.get(name, set())
        dropped += len(skip)
        kept = [ln for i, ln in enumerate(lines) if i not in skip]
        road_lines = (local_dir / "labels" / f"{name}.txt").read_text().splitlines()
        label.write_text("".join(f"{ln}\n" for ln in kept + road_lines if ln.strip()))

    for list_file in out.glob("*.txt"):
        kept = [ln for ln in list_file.read_text().split() if Path(ln).stem not in rejected]
        list_file.write_text("\n".join(kept) + "\n")
    split_rows = [r for r in read_csv(out / "split.csv") if r["name"] not in rejected]
    write_csv(out / "split.csv", list(split_rows[0]), split_rows)
    data = yaml.safe_load((out / "data.yaml").read_text())
    (out / "data.yaml").write_text(
        "# Paths are relative to this file. Road on pothole-only (gn_) train/val images\n"
        "# is pseudo-labelled by the road teacher; test has none. Road metrics come from\n"
        "# val_real_road.txt / test_real_road.txt only. See docs/datasets.md.\n"
        + yaml.safe_dump(data, sort_keys=False)
    )
    shutil.copy2(local_dir / PSEUDO_CSV, out / PSEUDO_CSV)

    labelled = [r for r in rows if r["name"] not in rejected]
    summary = {
        "pseudo_labelled": len(labelled),
        "pseudo_labelled_per_split": {
            sp: sum(r["split"] == sp for r in labelled) for sp in PSEUDO_SPLITS
        },
        "without_road": sum(r["n_road"] == "0" for r in labelled),
        "rejected": sorted(rejected),
        "offroad_pothole_labels_dropped": dropped,
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


def _drive_pseudo_dir() -> Path:
    from training.scripts.collect_runs import find_drive_root

    return find_drive_root() / "pseudo" / "teacher"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "merge.yaml")
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("predict", help="Colab: run the teacher")
    p.add_argument("--weights", type=Path, required=True, help="teacher best.pt")
    p.add_argument("--data-root", type=Path, required=True, help="unzipped merged-v1-base")
    p.add_argument("--out", type=Path, required=True, help="pseudo folder on Drive")
    r = sub.add_parser("review", help="Mac: fetch from Drive, check, draw sheets")
    r.add_argument("--drive-dir", type=Path, help="default: <synced Drive>/pseudo/teacher")
    r.add_argument("--no-fetch", action="store_true", help="reuse the local copy")
    r.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "pseudo_review")
    r.add_argument("--force", action="store_true", help="overwrite the decision CSVs")
    b = sub.add_parser("build", help="Mac: write merged-v1")
    b.add_argument("--zip", action="store_true", help="also write <out_dir>.zip")
    args = parser.parse_args()
    s = load_settings(args.config)

    if args.step == "predict":
        from ultralytics import YOLO  # only on the training machine

        with tempfile.TemporaryDirectory() as work:
            rows = predict(YOLO(str(args.weights)), args.data_root, Path(work), s)
            pack(Path(work), args.out)
        no_road = sum(r["n_road"] == 0 for r in rows)
        print(f"{len(rows)} images labelled, {no_road} without road; {LABELS_ZIP} on Drive")
    elif args.step == "review":
        if not args.no_fetch:
            n = fetch(args.drive_dir or _drive_pseudo_dir(), s.local_dir)
            print(f"Fetched {n} label files to {s.local_dir}")
        print(json.dumps(review(s.local_dir, s, args.out, args.force), indent=2))
    else:
        print(json.dumps(build(s.local_dir, s, args.zip), indent=2))


if __name__ == "__main__":
    main()
