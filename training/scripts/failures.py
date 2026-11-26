"""Failure analysis of a trained run on val (Mac only; Week 3, experiment 5).

Input: the predictions zip that training/scripts/evaluate.py wrote to
<drive>/runs/<run>/eval/preds_val_<imgsz>.zip, plus the local dataset (data/merged-v1).

What it does:
- keeps predictions with conf >= segmenter.conf (configs/pipeline.yaml), the masks the
  package will return,
- matches predicted potholes to labelled ones per image by mask IoU (greedy, highest
  confidence first, IoU >= failures.iou); masks are rasterised in normalised coords, which
  leaves IoU unchanged,
- reports precision, recall and recall by pothole size (the size buckets of
  docs/datasets.md), and road IoU on real-road images only (pseudo road is not truth),
- draws the worst images: most pothole errors (missed + false), and lowest road IoU,
- writes docs/failures_<run>.csv (one row per gallery image, `cause` filled in by hand
  from failures.causes in configs/eval.yaml) and docs/runs/<run>/failures_val.json.

Colours: missed pothole yellow, found pothole white, false pothole red, matched
prediction blue; predicted road tinted green.

Usage:
    python -m training.scripts.failures --run baseline            # fetch from Drive
    python -m training.scripts.failures --run baseline --no-fetch  # reuse local copy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.scripts.evaluate import fetch_preds, preds_zip_name
from training.scripts.inspect_dataset import SIZE_BUCKETS, parse_label_file, polygon_area
from training.scripts.merge_datasets import REPO_ROOT, _resolve
from training.scripts.pseudo_label import class_ids, write_csv

CSV_FIELDS = ["idx", "kind", "name", "real_road", "n_gt", "tp", "fn", "fp",
              "missed_area", "road_iou", "cause", "note"]  # fmt: skip
TILE_W, TILE_H, CAPTION_H = 480, 360, 24
SHEET_COLS, SHEET_ROWS = 4, 2
ROAD_COLOUR = (80, 200, 80)
MISSED = (0, 220, 255)
FOUND = (255, 255, 255)
FALSE = (40, 40, 230)
MATCHED = (255, 150, 0)


@dataclass
class FailureSettings:
    conf: float
    iou: float
    grid: int
    n_pothole: int
    n_road: int
    local_dir: Path
    out_dir: Path
    csv_dir: Path
    causes: list[str]


@dataclass
class Pred:
    cls: int
    conf: float
    polygon: np.ndarray


@dataclass
class ImageResult:
    name: str
    image: Path
    real_road: bool
    gt: list[np.ndarray]  # labelled pothole polygons
    preds: list[np.ndarray]  # predicted pothole polygons, conf >= threshold
    gt_found: list[bool]
    pred_matched: list[bool]
    ious: list[float] = field(default_factory=list)  # of matched pairs
    road_iou: float | None = None  # real-road images only
    road_pred: list[np.ndarray] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return sum(self.gt_found)

    @property
    def fn(self) -> int:
        return len(self.gt) - self.tp

    @property
    def fp(self) -> int:
        return len(self.preds) - sum(self.pred_matched)

    @property
    def missed_area(self) -> float:
        return sum(polygon_area(g) for g, f in zip(self.gt, self.gt_found, strict=True) if not f)


def load_settings(
    eval_path: Path = REPO_ROOT / "configs" / "eval.yaml",
    pipeline_path: Path = REPO_ROOT / "configs" / "pipeline.yaml",
) -> FailureSettings:
    f = yaml.safe_load(eval_path.read_text())["failures"]
    conf = yaml.safe_load(pipeline_path.read_text())["segmenter"]["conf"]
    return FailureSettings(
        conf=float(conf),
        iou=float(f["iou"]),
        grid=int(f["grid"]),
        n_pothole=int(f["n_pothole"]),
        n_road=int(f["n_road"]),
        local_dir=_resolve(f["local_dir"]),
        out_dir=_resolve(f["out_dir"]),
        csv_dir=_resolve(f["csv_dir"]),
        causes=list(f["causes"]),
    )


# ---------------------------------------------------------------- matching


def parse_preds(path: Path) -> list[Pred]:
    """Read one predictions file (`class conf x1 y1 ...` per line)."""
    preds = []
    for line in path.read_text().splitlines():
        v = line.split()
        if v:
            preds.append(Pred(int(v[0]), float(v[1]), np.array(v[2:], float).reshape(-1, 2)))
    return preds


def _mask(poly: np.ndarray, grid: int) -> np.ndarray:
    m = np.zeros((grid, grid), np.uint8)
    cv2.fillPoly(m, [np.round(poly * grid).astype(np.int32)], 1)
    return m.astype(bool)


def _union(polys: list[np.ndarray], grid: int) -> np.ndarray:
    m = np.zeros((grid, grid), bool)
    for p in polys:
        m |= _mask(p, grid)
    return m


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 0.0


def match(
    gt: list[np.ndarray], preds: list[np.ndarray], iou: float, grid: int
) -> tuple[list[bool], list[bool], list[float]]:
    """Greedy matching; `preds` must be sorted by confidence, highest first.

    Returns (gt found?, pred matched?, IoU of each matched pair).
    """
    gm = [_mask(g, grid) for g in gt]
    found, matched, ious = [False] * len(gt), [False] * len(preds), []
    for j, p in enumerate(preds):
        pm = _mask(p, grid)
        best, best_iou = -1, iou
        for i, g in enumerate(gm):
            if not found[i] and (v := mask_iou(pm, g)) >= best_iou:
                best, best_iou = i, v
        if best >= 0:
            found[best] = matched[j] = True
            ious.append(best_iou)
    return found, matched, ious


def _list_images(list_file: Path) -> dict[str, Path]:
    return {Path(n).stem: list_file.parent / n for n in list_file.read_text().split() if n}


def analyse(data_root: Path, preds_dir: Path, split: str, s: FailureSettings) -> list[ImageResult]:
    """Match predictions to labels for every image of `<split>.txt`."""
    road_id, pothole_id = class_ids(data_root)
    images = _list_images(data_root / f"{split}.txt")
    real_road = set(_list_images(data_root / f"{split}_real_road.txt"))
    results = []
    for name, image in images.items():
        pred_file = preds_dir / f"{name}.txt"
        if not pred_file.exists():
            raise SystemExit(f"No predictions for {name}: wrong zip for this split?")
        labels = parse_label_file(data_root / "labels" / f"{name}.txt")
        gt = [x.polygon for x in labels if x.class_id == pothole_id and not x.box_only]
        preds = sorted((p for p in parse_preds(pred_file) if p.conf >= s.conf),
                       key=lambda p: -p.conf)  # fmt: skip
        holes = [p.polygon for p in preds if p.cls == pothole_id]
        found, matched, ious = match(gt, holes, s.iou, s.grid)
        road_pred = [p.polygon for p in preds if p.cls == road_id]
        r = ImageResult(name, image, name in real_road, gt, holes, found, matched, ious,
                        road_pred=road_pred)  # fmt: skip
        if r.real_road:
            road_gt = [x.polygon for x in labels if x.class_id == road_id and not x.box_only]
            r.road_iou = mask_iou(_union(road_gt, s.grid), _union(road_pred, s.grid))
        results.append(r)
    return results


def _bucket(area: float) -> str:
    return next(name for name, upper in SIZE_BUCKETS if area < upper)


def summarise(results: list[ImageResult], s: FailureSettings) -> dict:
    """Precision, recall, recall by size, and road IoU over all images."""
    tp = sum(r.tp for r in results)
    fn = sum(r.fn for r in results)
    fp = sum(r.fp for r in results)
    by_size: dict[str, dict[str, float]] = {
        name: {"labels": 0, "found": 0} for name, _ in SIZE_BUCKETS
    }
    for r in results:
        for g, f in zip(r.gt, r.gt_found, strict=True):
            b = by_size[_bucket(polygon_area(g))]
            b["labels"] += 1
            b["found"] += int(f)
    for b in by_size.values():
        b["recall"] = round(b["found"] / b["labels"], 3) if b["labels"] else None
    road = [r.road_iou for r in results if r.road_iou is not None]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "conf": s.conf,
        "iou": s.iou,
        "images": len(results),
        "pothole": {
            "labels": tp + fn,
            "predictions": tp + fp,
            "found": tp,
            "missed": fn,
            "false": fp,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3) if tp else 0.0,
            "mean_matched_iou": round(float(np.mean([i for r in results for i in r.ious])), 3)
            if tp
            else None,
            "recall_by_size": by_size,
        },
        "road": {
            "real_road_images": len(road),
            "mean_iou": round(float(np.mean(road)), 3) if road else None,
            "images_below_0.8": sum(v < 0.8 for v in road),
        },
    }


def worst_pothole(results: list[ImageResult], n: int) -> list[ImageResult]:
    """Images with the most pothole errors (missed + false), then the most missed area."""
    bad = [r for r in results if r.fn + r.fp]
    return sorted(bad, key=lambda r: (-(r.fn + r.fp), -r.missed_area, r.name))[:n]


def worst_road(results: list[ImageResult], n: int) -> list[ImageResult]:
    road = [r for r in results if r.road_iou is not None]
    return sorted(road, key=lambda r: (r.road_iou, r.name))[:n]


# ---------------------------------------------------------------- drawing


def annotate(r: ImageResult) -> np.ndarray:
    """Full-size image with predicted road tinted and potholes outlined by outcome."""
    image = cv2.imread(str(r.image))
    if image is None:
        raise SystemExit(f"Cannot read {r.image}")
    h, w = image.shape[:2]

    def px(poly: np.ndarray) -> np.ndarray:
        return np.round(poly * [w, h]).astype(np.int32)

    overlay = image.copy()
    cv2.fillPoly(overlay, [px(p) for p in r.road_pred], ROAD_COLOUR)
    out = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    t = max(2, w // 300)
    for p, m in zip(r.preds, r.pred_matched, strict=True):
        cv2.polylines(out, [px(p)], True, MATCHED if m else FALSE, t)
    for g, f in zip(r.gt, r.gt_found, strict=True):
        cv2.polylines(out, [px(g)], True, FOUND if f else MISSED, t if f else 2 * t)
    return out


def _caption(idx: int, r: ImageResult) -> str:
    road = f" road {r.road_iou:.2f}" if r.road_iou is not None else ""
    return f"{idx} {r.name[:28]} miss {r.fn} false {r.fp}{road}"


def tile(image: np.ndarray, caption: str) -> np.ndarray:
    h, w = image.shape[:2]
    k = min(TILE_W / w, TILE_H / h)
    small = cv2.resize(image, (max(1, round(w * k)), max(1, round(h * k))))
    out = np.full((TILE_H + CAPTION_H, TILE_W, 3), 255, np.uint8)
    out[CAPTION_H : CAPTION_H + small.shape[0], : small.shape[1]] = small
    cv2.putText(out, caption, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def write_sheets(tiles: list[np.ndarray], out_dir: Path, prefix: str) -> list[Path]:
    per = SHEET_COLS * SHEET_ROWS
    paths = []
    for start in range(0, len(tiles), per):
        chunk = tiles[start : start + per]
        chunk += [np.full_like(tiles[0], 255)] * (per - len(chunk))
        rows = [np.hstack(chunk[i * SHEET_COLS : (i + 1) * SHEET_COLS]) for i in range(SHEET_ROWS)]
        path = out_dir / f"{prefix}_{start // per + 1:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows))
        paths.append(path)
    return paths


# ---------------------------------------------------------------- run


def _row(idx: int, kind: str, r: ImageResult) -> dict:
    return {
        "idx": idx,
        "kind": kind,
        "name": r.name,
        "real_road": int(r.real_road),
        "n_gt": len(r.gt),
        "tp": r.tp,
        "fn": r.fn,
        "fp": r.fp,
        "missed_area": round(r.missed_area, 4),
        "road_iou": "" if r.road_iou is None else round(r.road_iou, 3),
        "cause": "",
        "note": "",
    }


def run(
    run_name: str,
    data_root: Path,
    preds_dir: Path,
    s: FailureSettings,
    split: str = "val",
    force: bool = False,
    record_dir: Path = REPO_ROOT / "docs" / "runs",
) -> dict:
    """Analyse, draw the gallery, write the CSV and summary JSON; returns the summary."""
    if split == "test":
        raise SystemExit("Failure analysis shapes tuning, so it never looks at test")
    csv_path = s.csv_dir / f"failures_{run_name}.csv"
    if csv_path.exists() and not force:
        raise SystemExit(f"{csv_path} exists (it may hold your causes); use --force to redo")
    results = analyse(data_root, preds_dir, split, s)
    summary = {"run": run_name, "split": split, **summarise(results, s)}

    out = s.out_dir / run_name
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, sheets = [], []
    for kind, chosen in (("pothole", worst_pothole(results, s.n_pothole)),
                         ("road", worst_road(results, s.n_road))):  # fmt: skip
        tiles = []
        for r in chosen:
            idx = len(rows) + 1
            image = annotate(r)
            cv2.imwrite(str(out / "images" / f"{idx:02d}_{r.name}.jpg"), image)
            tiles.append(tile(image, _caption(idx, r)))
            rows.append(_row(idx, kind, r))
        if tiles:
            sheets += write_sheets(tiles, out, kind)
    write_csv(csv_path, CSV_FIELDS, rows)
    summary["gallery"] = {"csv": csv_path.name, "sheets": [p.name for p in sheets]}
    for folder in (record_dir / run_name, out):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"failures_{split}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="run folder under <drive>/runs")
    parser.add_argument("--imgsz", type=int, default=640, help="size the predictions were made at")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "merged-v1")
    parser.add_argument("--drive-dir", type=Path, help="default: the synced Drive project folder")
    parser.add_argument("--no-fetch", action="store_true", help="reuse the local predictions")
    parser.add_argument("--force", action="store_true", help="overwrite docs/failures_<run>.csv")
    args = parser.parse_args()
    s = load_settings()
    split = "val"
    preds_dir = s.local_dir / f"{args.run}_{split}_{args.imgsz}"
    if not args.no_fetch:
        from training.scripts.collect_runs import find_drive_root

        drive = args.drive_dir or find_drive_root()
        zip_path = drive / "runs" / args.run / "eval" / preds_zip_name(split, args.imgsz)
        if not zip_path.is_file():
            raise SystemExit(f"{zip_path.name} not found for run {args.run}: run evaluate first")
        print(f"Fetched {fetch_preds(zip_path, preds_dir)} prediction files")
    summary = run(args.run, args.data_root, preds_dir, s, split, args.force)
    print(json.dumps({k: summary[k] for k in ("pothole", "road", "gallery")}, indent=2))


if __name__ == "__main__":
    main()
