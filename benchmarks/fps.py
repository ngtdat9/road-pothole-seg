"""Per-stage latency of the full pipeline on this machine's CPU (Week 5, experiments 3 and 4).

Usage:
    python -m benchmarks.fps --source road.mp4 --name mac-m1
    python -m benchmarks.fps --source road.mp4 --name pc-openvino --provider openvino

Loads `--frames` frames from the video (resized to `--frame-width`, keeping the aspect),
then for each input size runs Segmenter -> analyse_potholes -> PotholeTracker on every
frame, `--warmup` frames first (not counted). Stages: preprocess, inference, postprocess
(mask decoding), analyse (clean + measure), track. Reports median and 90th percentile
per stage and FPS from the median total. Writes docs/runs/benchmark_<name>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from road_pothole_seg import PotholeTracker, Segmenter, analyse_potholes

REPO_ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}
STAGES = ("preprocess", "inference", "postprocess", "analyse", "track", "total")


def load_frames(source: str, n: int, width: int) -> list[np.ndarray]:
    """The first `n` frames of a video, resized to `width` px wide."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source}")
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frames.append(
            cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        )
    cap.release()
    if not frames:
        raise SystemExit(f"No frames read from {source}")
    return frames


def cpu_name() -> str:
    """CPU model name, best effort on macOS, Windows and Linux."""
    try:
        if platform.system() == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def summarise(times: dict[str, list[float]]) -> dict[str, Any]:
    """Median and p90 per stage (ms) and FPS from the median total."""
    out: dict[str, Any] = {
        s: {
            "median_ms": round(float(np.median(v)), 2),
            "p90_ms": round(float(np.percentile(v, 90)), 2),
        }
        for s, v in times.items()
    }
    out["fps"] = round(1000 / out["total"]["median_ms"], 1)
    return out


def bench_size(model: Path, frames: list[np.ndarray], config: dict[str, Any], providers: list[str],
               warmup: int) -> dict[str, Any]:  # fmt: skip
    seg = Segmenter(str(model), **{**config["segmenter"], "providers": providers})
    tracker = PotholeTracker(**config["tracker"])
    times: dict[str, list[float]] = {s: [] for s in STAGES}
    n_potholes = 0
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        s = seg(frame)
        t1 = time.perf_counter()
        measured = analyse_potholes(s, **config["postprocess"])
        t2 = time.perf_counter()
        tracker.update(measured)
        t3 = time.perf_counter()
        if i < warmup:
            continue
        for k, v in seg.last_timings_ms.items():
            times[k].append(v)
        times["analyse"].append((t2 - t1) * 1e3)
        times["track"].append((t3 - t2) * 1e3)
        times["total"].append((t3 - t0) * 1e3)
        n_potholes += len(measured)
    res = summarise(times)
    res["providers_used"] = seg.session.get_providers()
    res["mean_potholes_per_frame"] = round(n_potholes / max(len(frames) - warmup, 1), 2)
    return res


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="video file")
    parser.add_argument("--name", required=True, help="machine label, e.g. mac-m1, pc-cpu")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 480, 320])
    parser.add_argument("--models", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--frames", type=int, default=400, help="at most this many frames")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "runs")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    frames = load_frames(args.source, args.frames, args.frame_width)
    providers = PROVIDERS[args.provider]
    missing = set(providers) - set(ort.get_available_providers())
    if missing:
        raise SystemExit(f"{sorted(missing)} not available in this onnxruntime build")

    report: dict[str, Any] = {
        "date": dt.date.today().isoformat(),
        "name": args.name,
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "provider": args.provider,
        "source": Path(args.source).name,
        "frame_size": list(frames[0].shape[1::-1]),
        "frames": len(frames),
        "warmup": args.warmup,
        "sizes": {},
    }
    for size in args.sizes:
        r = bench_size(args.models / f"baseline-{size}.onnx", frames, config, providers,
                       args.warmup)  # fmt: skip
        report["sizes"][str(size)] = r
        stages = ", ".join(f"{s} {r[s]['median_ms']}" for s in STAGES)
        print(f"{size}: {r['fps']} FPS | median ms: {stages}")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"benchmark_{args.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
