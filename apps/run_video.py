"""Run the pipeline on a video file or webcam: overlays, FPS counter and a CSV pothole log.

Usage:
    python -m apps.run_video --source road.mp4
    python -m apps.run_video --source 0                 # webcam 0
    python -m apps.run_video --source road.mp4 --save out.mp4 --no-show

Press q in the window to stop. The CSV has one row per track:
track_id, first_frame, last_frame, max_area_px, max_area_m2, severity (at max area).
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from road_pothole_seg import (
    Pothole,
    PotholeTracker,
    Segmentation,
    Segmenter,
    SeverityBins,
    analyse_potholes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"
DEFAULT_MODEL = REPO_ROOT / "models" / "baseline-640.onnx"
CSV_FIELDS = ["track_id", "first_frame", "last_frame", "max_area_px", "max_area_m2", "severity"]

# Overlay colours (BGR) and opacity — display only, not pipeline settings.
ROAD_COLOUR = (80, 200, 80)
POTHOLE_COLOUR = (40, 40, 230)
OVERLAY_ALPHA = 0.4


# ---------------------------------------------------------------- config


def load_config(path: Path) -> dict[str, Any]:
    """Read configs/pipeline.yaml."""
    with path.open() as f:
        return yaml.safe_load(f)


def build_pipeline(
    config: dict[str, Any], model_path: Path
) -> tuple[Segmenter, PotholeTracker, SeverityBins]:
    """Create the pipeline objects from the config sections."""
    segmenter = Segmenter(str(model_path), **config["segmenter"])
    tracker = PotholeTracker(**config["tracker"])
    bins = SeverityBins(
        frac=tuple(config["severity"]["frac"]),
        m2=tuple(config["severity"]["m2"]),
    )
    return segmenter, tracker, bins


# ---------------------------------------------------------------- CSV log


@dataclass
class TrackRecord:
    track_id: int
    first_frame: int
    last_frame: int
    max_area_px: int
    max_area_m2: float | None
    severity: str


class PotholeLog:
    """Collect one record per track across frames."""

    def __init__(self) -> None:
        self.records: dict[int, TrackRecord] = {}

    def update(self, frame_idx: int, potholes: list[Pothole]) -> None:
        """Add this frame's tracked potholes."""
        for p in potholes:
            rec = self.records.get(p.track_id)
            if rec is None:
                self.records[p.track_id] = TrackRecord(
                    p.track_id, frame_idx, frame_idx, p.area_px, p.area_m2, p.severity
                )
                continue
            rec.last_frame = frame_idx
            if p.area_px > rec.max_area_px:
                rec.max_area_px, rec.max_area_m2, rec.severity = p.area_px, p.area_m2, p.severity

    def write_csv(self, path: Path) -> None:
        """Write all tracks to `path`, sorted by track_id."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_FIELDS)
            for rec in sorted(self.records.values(), key=lambda r: r.track_id):
                m2 = "" if rec.max_area_m2 is None else f"{rec.max_area_m2:.4f}"
                row = [rec.track_id, rec.first_frame, rec.last_frame, rec.max_area_px, m2]
                writer.writerow([*row, rec.severity])


# ---------------------------------------------------------------- overlay


def _tint(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> None:
    """Blend `colour` into `image` where `mask` is True, in place."""
    image[mask] = (image[mask] * (1 - OVERLAY_ALPHA) + np.array(colour) * OVERLAY_ALPHA).astype(
        np.uint8
    )


def draw_overlay(
    frame: np.ndarray, seg: Segmentation, potholes: list[Pothole], fps: float | None
) -> np.ndarray:
    """Return a copy of `frame` with road tint, pothole outlines, labels and FPS."""
    out = frame.copy()
    if seg.road_mask is not None:
        _tint(out, seg.road_mask, ROAD_COLOUR)
    for p in potholes:
        _tint(out, p.mask, POTHOLE_COLOUR)
        contours, _ = cv2.findContours(
            p.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, POTHOLE_COLOUR, 2)
        x0, y0 = p.bbox_xyxy[:2]
        cv2.putText(
            out,
            f"#{p.track_id} {p.severity}",
            (x0, max(12, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if fps is not None:
        cv2.putText(
            out, f"{fps:5.1f} FPS", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
    return out


# ---------------------------------------------------------------- main loop


def open_source(source: str) -> cv2.VideoCapture:
    """Open a video file, or a webcam when `source` is an integer like "0"."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap


def run(
    source: str,
    config_path: Path = DEFAULT_CONFIG,
    model_path: Path = DEFAULT_MODEL,
    csv_path: Path | None = None,
    save_path: Path | None = None,
    show: bool = True,
    max_frames: int | None = None,
) -> PotholeLog:
    """Process `source` frame by frame and return the pothole log."""
    config = load_config(config_path)
    segmenter, tracker, bins = build_pipeline(config, model_path)
    cap = open_source(source)
    writer: cv2.VideoWriter | None = None
    log = PotholeLog()
    fps: float | None = None
    frame_idx = 0

    try:
        while max_frames is None or frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            start = time.perf_counter()
            seg = segmenter(frame)
            measured = analyse_potholes(seg, severity_bins=bins, **config["postprocess"])
            potholes = tracker.update(measured)
            elapsed = time.perf_counter() - start
            # Pipeline FPS (excludes reading and drawing), smoothed so the number is readable.
            inst = 1.0 / max(elapsed, 1e-6)
            fps = inst if fps is None else 0.9 * fps + 0.1 * inst

            log.update(frame_idx, potholes)
            out = draw_overlay(frame, seg, potholes, fps)

            if save_path is not None:
                if writer is None:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(
                        str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
                    )
                writer.write(out)
            if show:
                cv2.imshow("road-pothole-seg", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if csv_path is not None:
        log.write_csv(csv_path)
    print(f"Processed {frame_idx} frames, {len(log.records)} tracks", end="")
    print(f", pipeline {fps:.1f} FPS" if fps is not None else "")
    return log


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="video file path or webcam index")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "outputs" / "potholes.csv")
    parser.add_argument("--save", type=Path, default=None, help="write the overlay video here")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    run(
        args.source,
        config_path=args.config,
        model_path=args.model,
        csv_path=args.csv,
        save_path=args.save,
        show=not args.no_show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
