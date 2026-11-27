"""Export a trained best.pt to ONNX at several input sizes (Mac, CPU; Week 4).

Usage (in a venv with `pip install -e ".[train]"`):
    python -m training.scripts.export_onnx --weights <drive>/runs/baseline/weights/best.pt

What it does:
- exports one fixed-shape FP32 model per size in configs/export.yaml to
  <out_dir>/<name>-<size>.onnx (Ultralytics embeds class names and imgsz as metadata),
- opens each file with ONNX Runtime and records its input/output shapes,
- writes <out_dir>/<name>.json: weights hash, versions, commit and one entry per size.

Ultralytics is imported inside `export` only, so tests and CI never need it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from training.scripts.merge_datasets import REPO_ROOT, _resolve
from training.scripts.train import git_commit, package_version


@dataclass
class ExportSettings:
    name: str
    sizes: list[int]
    opset: int
    simplify: bool
    out_dir: Path


def load_settings(config_path: Path) -> ExportSettings:
    """The `export` section of configs/export.yaml."""
    with config_path.open() as f:
        raw = yaml.safe_load(f)["export"]
    sizes = [int(s) for s in raw["sizes"]]
    if not sizes or any(s <= 0 or s % 32 for s in sizes):
        raise ValueError(f"sizes must be positive multiples of 32 (model stride), got {sizes}")
    return ExportSettings(
        name=str(raw["name"]),
        sizes=sizes,
        opset=int(raw["opset"]),
        simplify=bool(raw["simplify"]),
        out_dir=_resolve(raw["out_dir"]),
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path) -> dict[str, Any]:
    """Input/output names and shapes, and the metadata Ultralytics stored, via ONNX Runtime."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    return {
        "inputs": {i.name: list(i.shape) for i in sess.get_inputs()},
        "outputs": {o.name: list(o.shape) for o in sess.get_outputs()},
        "metadata": {k: meta[k] for k in sorted(meta)},
    }


def export(
    weights: Path,
    s: ExportSettings,
    yolo_cls: Any = None,
    describe_fn: Callable[[Path], dict[str, Any]] = describe,
) -> dict[str, Any]:
    """Export every size and write <out_dir>/<name>.json; returns that report."""
    if yolo_cls is None:
        from ultralytics import YOLO as yolo_cls  # noqa: N813

    s.out_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    for size in s.sizes:
        made = yolo_cls(str(weights)).export(
            format="onnx", imgsz=size, opset=s.opset, simplify=s.simplify,
            dynamic=False, half=False, device="cpu",
        )  # fmt: skip
        dest = s.out_dir / f"{s.name}-{size}.onnx"
        shutil.move(str(made), dest)  # Ultralytics writes next to the weights
        models[str(size)] = {"file": dest.name, "sha256": sha256(dest), **describe_fn(dest)}

    report = {
        "date": dt.date.today().isoformat(),
        "weights": weights.name,
        "weights_sha256": sha256(weights),
        "commit": git_commit(),
        "ultralytics": package_version("ultralytics"),
        "onnx": package_version("onnx"),
        "opset": s.opset,
        "models": models,
    }
    (s.out_dir / f"{s.name}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "export.yaml")
    args = parser.parse_args()
    report = export(args.weights, load_settings(args.config))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
