"""Check the package's ONNX Segmenter against Ultralytics (Mac, CPU; Week 4).

Usage (in a venv with `pip install -e ".[train]"`):
    python -m training.scripts.check_onnx --onnx models/baseline-640.onnx \
        --pt models/baseline.pt --data-root data/merged-v1

On `n_images` val images (seeded sample; never test), each in four shapes (original,
16:9 landscape crop, 9:16 portrait crop, small 400x300 resize), it compares:
- `same_model`: Ultralytics predicting with the same ONNX file. Same network and
  square input, so this isolates our letterbox, NMS and mask decoding. Must pass.
- `pytorch`: Ultralytics predicting with best.pt. It letterboxes to the smallest
  stride-multiple rectangle instead of a square, so non-square frames see a slightly
  different input; reported, not gated.

Pothole instances are matched greedily by mask IoU (highest reference score first).
Pass: mean matched IoU >= `min_iou` and no unmatched instance with score > `max_unmatched_conf`
on either side. Writes <out>/onnx_check.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from training.scripts.merge_datasets import REPO_ROOT, _resolve

VARIANTS = ("original", "landscape", "portrait", "small")


def load_settings(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)["check"]


def make_variants(frame: np.ndarray) -> dict[str, np.ndarray]:
    """The frame in four shapes: as is, centre-cropped to 16:9 and 9:16, resized to 400x300."""
    h, w = frame.shape[:2]

    def crop(aspect: float) -> np.ndarray:  # aspect = width / height
        cw, ch = (w, round(w / aspect)) if w / h < aspect else (round(h * aspect), h)
        cw, ch = min(cw, w), min(ch, h)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        return np.ascontiguousarray(frame[y0 : y0 + ch, x0 : x0 + cw])

    small = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA)
    return {"original": frame, "landscape": crop(16 / 9), "portrait": crop(9 / 16),
            "small": small}  # fmt: skip


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def match_instances(
    ours: list[np.ndarray], our_scores: list[float],
    ref: list[np.ndarray], ref_scores: list[float], min_match: float = 0.5,
) -> dict[str, Any]:  # fmt: skip
    """Greedy one-to-one matching by mask IoU, highest reference score first."""
    free = set(range(len(ours)))
    matched: list[dict[str, float]] = []
    unmatched_ref: list[float] = []
    for j in np.argsort(ref_scores)[::-1]:
        best, best_iou = None, min_match
        for i in free:
            iou = mask_iou(ours[i], ref[j])
            if iou >= best_iou:
                best, best_iou = i, iou
        if best is None:
            unmatched_ref.append(float(ref_scores[j]))
            continue
        free.discard(best)
        matched.append({"iou": best_iou, "score_diff": abs(our_scores[best] - ref_scores[j])})
    return {
        "matched": matched,
        "unmatched_ours": [float(our_scores[i]) for i in sorted(free)],
        "unmatched_ref": unmatched_ref,
    }


def summarise(records: list[dict[str, Any]], min_iou: float, max_conf: float) -> dict[str, Any]:
    """Aggregate per-frame comparisons (one reference) into totals and a pass flag."""
    ious = [m["iou"] for r in records for m in r["matched"]]
    diffs = [m["score_diff"] for r in records for m in r["matched"]]
    un_ours = [s for r in records for s in r["unmatched_ours"]]
    un_ref = [s for r in records for s in r["unmatched_ref"]]
    road = [r["road_iou"] for r in records]
    worst_unmatched = max(un_ours + un_ref, default=0.0)
    mean_iou = float(np.mean(ious)) if ious else 1.0
    return {
        "frames": len(records),
        "matched": len(ious),
        "mean_pothole_iou": round(mean_iou, 4),
        "min_pothole_iou": round(min(ious, default=1.0), 4),
        "max_score_diff": round(max(diffs, default=0.0), 4),
        "unmatched_ours": len(un_ours),
        "unmatched_ref": len(un_ref),
        "max_unmatched_score": round(worst_unmatched, 4),
        "mean_road_iou": round(float(np.mean(road)), 4) if road else None,
        "min_road_iou": round(min(road), 4) if road else None,
        "passed": mean_iou >= min_iou and worst_unmatched <= max_conf,
    }


def ultralytics_masks(
    model: Any, frame: np.ndarray, size: int, conf: float, iou: float, names: list[str]
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Road (union, no pothole fill) and pothole masks at frame size from Ultralytics predict."""
    from ultralytics.utils import ops

    r = model.predict(frame, imgsz=size, conf=conf, iou=iou, verbose=False)[0]
    road = np.zeros(frame.shape[:2], dtype=bool)
    if r.masks is None:
        return road, [], []
    masks = ops.scale_masks(r.masks.data[None].float(), frame.shape[:2])[0].numpy() > 0.5
    classes = r.boxes.cls.numpy().astype(int)
    scores = r.boxes.conf.numpy().astype(float)
    potholes, pothole_scores = [], []
    for m, c, s in zip(masks, classes, scores, strict=True):
        if names[c] == "road":
            road |= m
        elif names[c] == "pothole":
            potholes.append(m)
            pothole_scores.append(float(s))
    return road, potholes, pothole_scores


def check(onnx_path: Path, pt_path: Path, data_root: Path, cfg: dict[str, Any],
          pipeline: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    from ultralytics import YOLO

    from road_pothole_seg import Segmenter

    seg_cfg = {**pipeline["segmenter"], "fill_road": False}  # compare raw road to raw road
    ours = Segmenter(str(onnx_path), **seg_cfg)
    refs = {"same_model": YOLO(str(onnx_path), task="segment"), "pytorch": YOLO(str(pt_path))}

    lines = (data_root / "val.txt").read_text().split()
    sample = random.Random(cfg["seed"]).sample(sorted(lines), cfg["n_images"])
    records: dict[str, list[dict[str, Any]]] = {k: [] for k in refs}
    for line in sample:
        frame = cv2.imread(str(data_root / line))
        if frame is None:
            raise FileNotFoundError(data_root / line)
        for variant, f in make_variants(frame).items():
            out = ours(f)
            for key, model in refs.items():
                road, potholes, scores = ultralytics_masks(
                    model, f, ours.input_size, ours.conf, ours.iou, ours.names
                )
                rec = match_instances(out.pothole_masks, out.scores, potholes, scores)
                rec.update(image=Path(line).name, variant=variant, shape=list(f.shape[:2]),
                           road_iou=mask_iou(out.road_mask, road))  # fmt: skip
                records[key].append(rec)

    report: dict[str, Any] = {
        "onnx": onnx_path.name,
        "pt": pt_path.name,
        "images": len(sample),
        "variants": list(VARIANTS),
        "segmenter": seg_cfg,
        "min_iou": cfg["min_iou"],
        "max_unmatched_conf": cfg["max_unmatched_conf"],
    }
    for key, recs in records.items():
        report[key] = summarise(recs, cfg["min_iou"], cfg["max_unmatched_conf"])
        report[key]["by_variant"] = {
            v: summarise(
                [r for r in recs if r["variant"] == v], cfg["min_iou"], cfg["max_unmatched_conf"]
            )  # fmt: skip
            for v in VARIANTS
        }
    report["passed"] = report["same_model"]["passed"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "merged-v1")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "export.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "runs" / "baseline")
    args = parser.parse_args()

    cfg = load_settings(args.config)
    pipeline = yaml.safe_load((REPO_ROOT / "configs" / "pipeline.yaml").read_text())
    report = check(args.onnx, args.pt, _resolve(str(args.data_root)), cfg, pipeline)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "onnx_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("same_model", "pytorch")}))
    for key in ("same_model", "pytorch"):
        print(key, {k: v for k, v in report[key].items() if k != "by_variant"})


if __name__ == "__main__":
    main()
