import csv
from pathlib import Path

import cv2
import numpy as np
from conftest import FakeDet, FakeModel

from apps.run_video import PotholeLog, draw_overlay, run
from road_pothole_seg import Pothole, Segmentation


def _pothole(track_id: int, area_px: int, severity: str) -> Pothole:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    return Pothole(
        track_id=track_id,
        mask=mask,
        score=0.9,
        bbox_xyxy=(5, 5, 10, 10),
        area_px=area_px,
        area_frac=area_px / 400,
        area_m2=None,
        severity=severity,
        center_px=(7, 7),
    )


def _write_video(path: Path, n_frames: int, size: tuple[int, int] = (160, 120)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i * 10 % 255, dtype=np.uint8))
    writer.release()


def test_log_tracks_frame_range_and_max_area(tmp_path: Path) -> None:
    log = PotholeLog()
    log.update(0, [_pothole(1, 50, "small")])
    log.update(3, [_pothole(1, 400, "large"), _pothole(2, 10, "small")])
    log.update(5, [_pothole(1, 200, "medium")])

    out = tmp_path / "log.csv"
    log.write_csv(out)
    rows = list(csv.DictReader(out.open()))
    assert rows[0] == {
        "track_id": "1",
        "first_frame": "0",
        "last_frame": "5",
        "max_area_px": "400",
        "max_area_m2": "",
        "severity": "large",
    }
    assert rows[1]["track_id"] == "2" and rows[1]["first_frame"] == rows[1]["last_frame"] == "3"


def test_overlay_keeps_shape_and_does_not_touch_input() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    road = np.zeros((20, 20), dtype=bool)
    road[10:] = True
    p = _pothole(1, 25, "small")
    seg = Segmentation(road_mask=road, pothole_masks=[p.mask], scores=[0.9])

    out = draw_overlay(frame, seg, [p], fps=12.3)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert not frame.any()  # input unchanged
    assert out[15, 15].any()  # road tinted


def test_run_end_to_end_on_video_file(tmp_path: Path, fake_model: FakeModel) -> None:
    fake_model.dets = [FakeDet(1, 0.8, (300, 300, 360, 360))]  # one pothole per frame
    video = tmp_path / "in.avi"
    _write_video(video, n_frames=6)
    csv_path = tmp_path / "potholes.csv"
    save_path = tmp_path / "out.mp4"

    log = run(str(video), csv_path=csv_path, save_path=save_path, show=False)

    assert list(log.records) == [1]  # the same pothole in every frame: one track
    assert csv_path.exists() and save_path.exists() and save_path.stat().st_size > 0
    rows = list(csv.DictReader(csv_path.open()))
    assert [(r["first_frame"], r["last_frame"]) for r in rows] == [("1", "5")]  # ID from frame 2


def test_run_respects_max_frames(tmp_path: Path, fake_model: FakeModel) -> None:
    fake_model.dets = [FakeDet(1, 0.8, (300, 300, 360, 360))]
    video = tmp_path / "in.avi"
    _write_video(video, n_frames=10)
    log = run(str(video), show=False, max_frames=3)
    assert log.records[1].last_frame == 2
