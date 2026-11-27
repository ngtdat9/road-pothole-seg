from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.fps import load_frames, summarise


def test_load_frames_resizes_and_stops_at_n(tmp_path: Path) -> None:
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 180))
    for _ in range(5):
        w.write(np.zeros((180, 320, 3), dtype=np.uint8))
    w.release()
    frames = load_frames(str(path), n=3, width=640)
    assert len(frames) == 3 and frames[0].shape == (360, 640, 3)


def test_summarise_reports_median_p90_and_fps() -> None:
    r = summarise({"total": [10.0, 20.0, 30.0, 40.0, 50.0]})
    assert r["total"]["median_ms"] == 30.0
    assert r["total"]["p90_ms"] == pytest.approx(46.0)
    assert r["fps"] == pytest.approx(33.3)
