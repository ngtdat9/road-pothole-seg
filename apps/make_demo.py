"""Make the README demo: overlay video, pothole CSV and a short GIF (Mac).

Usage:
    python -m apps.make_demo --source road.mp4 --start 2 --duration 6

Runs the full pipeline with `apps.run_video` (overlay MP4 + CSV in --out-dir), then cuts
`--duration` seconds from `--start` into a GIF (`--gif`, default docs/figures/demo.gif),
resized to `--width` px and sampled at `--gif-fps`. Needs Pillow (`pip install -e ".[demo]"`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from apps.run_video import DEFAULT_CONFIG, DEFAULT_MODEL, REPO_ROOT, run


def gif_frames(video: Path, start_s: float, duration_s: float, gif_fps: float,
               width: int) -> list[np.ndarray]:  # fmt: skip
    """RGB frames from `video` between start and start + duration, every 1/gif_fps s."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    first, last = round(start_s * src_fps), round((start_s + duration_s) * src_fps)
    step = max(src_fps / gif_fps, 1.0)
    wanted = {round(first + k * step) for k in range(int((last - first) / step) + 1)}
    frames, idx = [], 0
    try:
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                size = (width, round(frame.shape[0] * width / frame.shape[1]))
                small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            idx += 1
    finally:
        cap.release()
    return frames


def write_gif(frames: list[np.ndarray], path: Path, gif_fps: float) -> None:
    """Looping GIF with one adaptive palette per frame."""
    from PIL import Image

    if not frames:
        raise SystemExit("no frames in the chosen time range")
    images = [Image.fromarray(f).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
              for f in frames]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], loop=0,
                   duration=round(1000 / gif_fps), optimize=True)  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="input video")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "demo")
    parser.add_argument("--gif", type=Path, default=REPO_ROOT / "docs" / "figures" / "demo.gif")
    parser.add_argument("--start", type=float, default=0.0, help="GIF start, seconds")
    parser.add_argument("--duration", type=float, default=6.0, help="GIF length, seconds")
    parser.add_argument("--gif-fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args(argv)

    video = args.out_dir / "overlay.mp4"
    run(args.source, config_path=args.config, model_path=args.model,
        csv_path=args.out_dir / "potholes.csv", save_path=video, show=False)  # fmt: skip
    frames = gif_frames(video, args.start, args.duration, args.gif_fps, args.width)
    write_gif(frames, args.gif, args.gif_fps)
    size_mb = args.gif.stat().st_size / 1e6
    print(f"GIF: {args.gif} ({len(frames)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
