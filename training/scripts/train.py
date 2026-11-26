"""Train a YOLO26-seg model from a named profile in configs/train.yaml (Colab only).

Usage (in the Colab notebook):
    python -m training.scripts.train --profile sanity \
        --data-root /content/merged-v1-base \
        --out /content/drive/MyDrive/road-pothole-seg/runs
    python -m training.scripts.train --profile baseline ... --resume

What it does:
- merges `defaults` with the profile and checks our rules (no vertical flip, no
  test list in training, lists exist),
- writes data_<profile>.yaml next to the dataset (paths to the profile's lists),
- trains with Ultralytics; checkpoints go to <out>/<name>/weights on Drive,
- validates best.pt: pothole metrics on the profile's val list, road metrics on
  val_real_road.txt only (pothole-only images have no real road labels),
- writes <out>/<name>/summary.json and prints a row for docs/results.md.

Ultralytics is imported inside `run` only, so tests and CI never need it.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NON_TRAIN_KEYS = ("model", "data", "notes")
ROAD_VAL_LIST = "val_real_road.txt"


# ---------------------------------------------------------------- config


def load_profile(config_path: Path, profile: str) -> dict[str, Any]:
    """`defaults` updated with one profile (nested `data` merged key by key)."""
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    if profile not in raw["profiles"]:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(raw['profiles'])}")
    merged = copy.deepcopy(raw["defaults"])
    for key, value in (raw["profiles"][profile] or {}).items():
        if key == "data":
            merged["data"] = {**merged.get("data", {}), **value}
        else:
            merged[key] = value
    return merged


def check_profile(cfg: dict[str, Any], data_root: Path) -> None:
    """Refuse settings that break the project's data and augmentation rules."""
    if float(cfg.get("flipud", 0)) != 0:
        raise ValueError("flipud must be 0: roads are never upside down")
    lists = cfg["data"]
    if set(lists) != {"train", "val"}:
        raise ValueError(f"data must name exactly a train and a val list, got {sorted(lists)}")
    for split, name in lists.items():
        if "test" in name:
            raise ValueError(f"data.{split} uses a test list ({name}); test is for final eval only")
        if not (data_root / name).is_file():
            raise FileNotFoundError(f"{data_root / name} not found (is the dataset unzipped?)")


def train_args(cfg: dict[str, Any]) -> dict[str, Any]:
    """The keys passed straight to Ultralytics' train()."""
    return {k: v for k, v in cfg.items() if k not in NON_TRAIN_KEYS}


def write_data_yaml(data_root: Path, lists: dict[str, str], name: str) -> Path:
    """data_<name>.yaml in the dataset folder, pointing at the chosen lists."""
    with (data_root / "data.yaml").open() as f:
        names = yaml.safe_load(f)["names"]
    data = {"path": str(data_root.resolve()), **lists, "names": names}
    path = data_root / f"data_{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# ---------------------------------------------------------------- summary


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def class_metrics(seg: Any, names: dict[int, str], cls: str) -> dict[str, float] | None:
    """Mask mAP50 and mAP50-95 of one class from an Ultralytics `metrics.seg` object."""
    ids = [i for i, n in names.items() if n == cls]
    indices = list(seg.ap_class_index)
    if not ids or ids[0] not in indices:
        return None  # class absent from this val set
    k = indices.index(ids[0])
    return {"map50": round(float(seg.ap50[k]), 4), "map50_95": round(float(seg.ap[k]), 4)}


def results_row(summary: dict[str, Any]) -> str:
    """One Markdown row for docs/results.md."""

    def fmt(cls: str, key: str) -> str:
        m = summary["metrics"].get(cls)
        return f"{m[key]:.3f}" if m else "—"

    cfg = summary["config"]
    return (
        f"| {summary['date']} | {summary['name']} | {Path(cfg['model']).stem} | "
        f"{cfg['data']['train']} | {cfg['epochs']} | {cfg['imgsz']} | "
        f"{fmt('pothole', 'map50')} | {fmt('pothole', 'map50_95')} | "
        f"{fmt('road', 'map50')} | {fmt('road', 'map50_95')} | {summary['commit']} | |"
    )


# ---------------------------------------------------------------- run


def run(
    profile: str,
    data_root: Path,
    out: Path,
    config_path: Path = REPO_ROOT / "configs" / "train.yaml",
    name: str | None = None,
    resume: bool = False,
    yolo_cls: Any = None,
    callbacks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train, validate best.pt, and write summary.json.

    `callbacks` maps Ultralytics event names to functions (e.g. monitor.Monitor's live
    panel). `yolo_cls` is for tests.
    """
    if yolo_cls is None:
        from ultralytics import YOLO  # only on the training machine

        yolo_cls = YOLO
    cfg = load_profile(config_path, profile)
    check_profile(cfg, data_root)
    name = name or profile
    data_yaml = write_data_yaml(data_root, cfg["data"], name)
    run_dir = out / name

    model = yolo_cls(str(run_dir / "weights" / "last.pt") if resume else cfg["model"])
    for event, fn in (callbacks or {}).items():
        model.add_callback(event, fn)
    if resume:
        model.train(resume=True)
    else:
        model.train(data=str(data_yaml), project=str(out), name=name, exist_ok=True,
                    **train_args(cfg))  # fmt: skip

    best = yolo_cls(str(run_dir / "weights" / "best.pt"))
    common = {"imgsz": cfg["imgsz"], "batch": cfg["batch"], "plots": False}
    pothole_val = best.val(data=str(data_yaml), split="val", **common)
    if cfg["data"]["val"] == ROAD_VAL_LIST:
        road_val = pothole_val  # already validated on real-road images
    else:
        road_lists = {"train": cfg["data"]["train"], "val": ROAD_VAL_LIST}
        road_yaml = write_data_yaml(data_root, road_lists, f"{name}_roadval")
        road_val = best.val(data=str(road_yaml), split="val", **common)

    summary = {
        "date": dt.date.today().isoformat(),
        "profile": profile,
        "name": name,
        "commit": git_commit(),
        "ultralytics": package_version("ultralytics"),
        "config": cfg,
        "metrics": {
            "pothole": class_metrics(pothole_val.seg, dict(pothole_val.names), "pothole"),
            "road": class_metrics(road_val.seg, dict(road_val.names), "road"),
        },
        "metric_sources": {"pothole": cfg["data"]["val"], "road": ROAD_VAL_LIST},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="unzipped dataset folder")
    parser.add_argument("--out", type=Path, required=True, help="runs folder, e.g. on Drive")
    parser.add_argument("--name", help="run name (default: the profile name)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt")
    args = parser.parse_args()
    summary = run(args.profile, args.data_root, args.out, args.config, args.name, args.resume)
    print(json.dumps(summary["metrics"], indent=2))
    print("\nRow for docs/results.md (fill in the conclusion):")
    print(results_row(summary))


if __name__ == "__main__":
    main()
