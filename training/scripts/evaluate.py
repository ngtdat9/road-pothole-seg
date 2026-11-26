"""Score a trained best.pt on val (or, once, on test) at several input sizes (Colab only).

Usage (in training/notebooks/evaluate_colab.ipynb):
    python -m training.scripts.evaluate --weights <drive>/runs/baseline/weights/best.pt \
        --data-root /content/merged-v1 --out <drive>/runs/baseline/eval \
        --split val --sizes 640 480 320 --preds-imgsz 640

What it does:
- validates at every size in --sizes: pothole mask mAP on `<split>.txt`, road mask mAP on
  `<split>_real_road.txt` only (pothole-only images have no real road labels),
- with --preds-imgsz, predicts every image of `<split>.txt` at that size and writes ONE
  zip of per-image prediction files (`class conf x1 y1 ...`, normalised polygons; an
  empty file = nothing found) plus a manifest, for training/scripts/failures.py on the Mac,
- writes <out>/eval_<split>.json.

The test split needs --final: it is opened once, for the chosen model (Week 3, Day 6).
Ultralytics is imported inside `evaluate` only, so tests and CI never need it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from training.scripts.train import class_metrics, git_commit, package_version, write_data_yaml

EVAL_SPLITS = ("val", "test")
MANIFEST = "manifest.csv"
PRED_CONF_FLOOR = 0.05  # keep low-confidence masks too; failures.py applies the real threshold


def preds_zip_name(split: str, imgsz: int) -> str:
    return f"preds_{split}_{imgsz}.zip"


def _numpy(x: Any) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _images(list_file: Path) -> list[Path]:
    """Image paths of a YOLO list file (lines like ./images/x.jpg, relative to its folder)."""
    return [list_file.parent / line for line in list_file.read_text().split() if line]


def format_preds(classes: np.ndarray, confs: np.ndarray, polygons: list[np.ndarray]) -> str:
    """One `class conf x1 y1 x2 y2 ...` line per predicted polygon with 3+ points."""
    lines = []
    for cls, conf, poly in zip(classes, confs, polygons, strict=True):
        poly = np.asarray(poly, float).reshape(-1, 2)
        if len(poly) < 3:
            continue  # Ultralytics gives an empty segment for masks too small to trace
        coords = " ".join(f"{v:.5f}" for v in poly.reshape(-1))
        lines.append(f"{int(cls)} {float(conf):.4f} {coords}")
    return "\n".join(lines) + ("\n" if lines else "")


def predict_split(
    model: Any, data_root: Path, split: str, imgsz: int, out_zip: Path, conf: float
) -> int:
    """Predict every image of `<split>.txt`; write the zip (labels + manifest) to `out_zip`."""
    images = _images(data_root / f"{split}.txt")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rows = []
        for image in images:
            stem = image.stem
            result = model.predict(
                str(image), imgsz=imgsz, conf=conf, retina_masks=True, verbose=False
            )[0]
            text = ""
            if result.masks is not None:
                text = format_preds(
                    _numpy(result.boxes.cls), _numpy(result.boxes.conf), list(result.masks.xyn)
                )
            (work / f"{stem}.txt").write_text(text)
            rows.append({"name": stem, "n": text.count("\n")})
        with (work / MANIFEST).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "n"])
            writer.writeheader()
            writer.writerows(rows)
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(out_zip.with_suffix("")), "zip", root_dir=work)
    return len(images)


def fetch_preds(zip_path: Path, local_dir: Path) -> int:
    """Unpack a predictions zip into `local_dir` (flat .txt files only); returns the count."""
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if MANIFEST not in names:
            raise SystemExit(f"{zip_path.name} has no {MANIFEST}")
        (local_dir / MANIFEST).write_bytes(z.read(MANIFEST))
        labels = [n for n in names if n.endswith(".txt") and "/" not in n.strip("/")]
        for n in labels:  # never extract paths from the archive
            (local_dir / n).write_bytes(z.read(n))
    with (local_dir / MANIFEST).open() as f:
        expected = sum(1 for _ in csv.DictReader(f))
    if len(labels) != expected:
        raise SystemExit(f"{zip_path.name} has {len(labels)} files, the manifest lists {expected}")
    return len(labels)


def evaluate(
    weights: Path,
    data_root: Path,
    out_dir: Path,
    sizes: list[int],
    split: str = "val",
    final: bool = False,
    preds_imgsz: int | None = None,
    batch: int = 16,
    yolo_cls: Any = None,
) -> dict[str, Any]:
    """Validate `weights` at each size; optionally save predictions. `yolo_cls` is for tests."""
    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {EVAL_SPLITS}, got {split!r}")
    if split == "test" and not final:
        raise SystemExit("test is for the final evaluation only: pass --final to open it")
    pothole_list, road_list = f"{split}.txt", f"{split}_real_road.txt"
    for name in (pothole_list, road_list):
        if not (data_root / name).is_file():
            raise FileNotFoundError(f"{data_root / name} not found (is the dataset unzipped?)")
    if yolo_cls is None:
        from ultralytics import YOLO  # only on the training machine

        yolo_cls = YOLO
    model = yolo_cls(str(weights))
    pothole_yaml = write_data_yaml(
        data_root, {"train": "train.txt", "val": pothole_list}, f"eval_{split}"
    )
    road_yaml = write_data_yaml(
        data_root, {"train": "train.txt", "val": road_list}, f"eval_{split}_road"
    )

    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:  # Ultralytics' own val folders go here
        common = {"split": "val", "batch": batch, "plots": False, "project": tmp, "verbose": False}
        for imgsz in sizes:
            pv = model.val(data=str(pothole_yaml), imgsz=imgsz, name=f"p{imgsz}", **common)
            rv = model.val(data=str(road_yaml), imgsz=imgsz, name=f"r{imgsz}", **common)
            results[str(imgsz)] = {
                "pothole": class_metrics(pv.seg, dict(pv.names), "pothole"),
                "road": class_metrics(rv.seg, dict(rv.names), "road"),
            }

    summary: dict[str, Any] = {
        "date": dt.date.today().isoformat(),
        "weights": str(weights),
        "split": split,
        "commit": git_commit(),
        "ultralytics": package_version("ultralytics"),
        "metric_sources": {"pothole": pothole_list, "road": road_list},
        "sizes": results,
    }
    if preds_imgsz:
        zip_path = out_dir / preds_zip_name(split, preds_imgsz)
        n = predict_split(model, data_root, split, preds_imgsz, zip_path, PRED_CONF_FLOOR)
        summary["predictions"] = {"zip": zip_path.name, "imgsz": preds_imgsz, "images": n,
                                  "conf_floor": PRED_CONF_FLOOR}  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{split}.json").write_text(json.dumps(summary, indent=2))
    return summary


def table(summary: dict[str, Any]) -> str:
    """Markdown rows: one per input size."""

    def fmt(m: dict[str, float] | None, key: str) -> str:
        return f"{m[key]:.3f}" if m else "—"

    lines = ["| imgsz | pothole mAP50 | pothole mAP50-95 | road mAP50 | road mAP50-95 |",
             "|---|---|---|---|---|"]  # fmt: skip
    for size, m in summary["sizes"].items():
        p, r = m["pothole"], m["road"]
        lines.append(f"| {size} | {fmt(p, 'map50')} | {fmt(p, 'map50_95')} | "
                     f"{fmt(r, 'map50')} | {fmt(r, 'map50_95')} |")  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="unzipped dataset folder")
    parser.add_argument("--out", type=Path, required=True, help="e.g. <drive>/runs/<run>/eval")
    parser.add_argument("--split", choices=EVAL_SPLITS, default="val")
    parser.add_argument("--sizes", type=int, nargs="+", default=[640])
    parser.add_argument("--preds-imgsz", type=int, default=0, help="0 = no predictions zip")
    parser.add_argument("--final", action="store_true", help="allow --split test")
    args = parser.parse_args()
    summary = evaluate(args.weights, args.data_root, args.out, args.sizes, args.split,
                       args.final, args.preds_imgsz or None)  # fmt: skip
    print(table(summary))


if __name__ == "__main__":
    main()
