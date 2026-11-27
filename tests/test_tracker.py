import numpy as np
import pytest

from road_pothole_seg.tracker import PotholeTracker, box_iou, greedy_match
from road_pothole_seg.types import Pothole


def _pothole(x: float, y: float, size: float = 40, score: float = 0.9) -> Pothole:
    box = (int(x), int(y), int(x + size), int(y + size))
    return Pothole(
        track_id=-1,
        mask=np.ones((2, 2), dtype=bool),
        score=score,
        bbox_xyxy=box,
        area_px=int(size * size),
        area_frac=0.01,
        area_m2=None,
        severity="medium",
        center_px=(int(x + size / 2), int(y + size / 2)),
    )


def _ids(tracker: PotholeTracker, frame: list[Pothole]) -> list[int]:
    return [p.track_id for p in tracker.update(frame)]


# ---------------------------------------------------------------- helpers


def test_box_iou_and_greedy_match() -> None:
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    b = np.array([[0, 0, 10, 5], [21, 21, 31, 31], [100, 100, 110, 110]], dtype=float)
    iou = box_iou(a, b)
    assert iou[0, 0] == pytest.approx(0.5) and iou[0, 2] == 0
    assert greedy_match(iou, 0.3) == [(1, 1), (0, 0)]
    assert greedy_match(iou, 0.9) == []
    assert box_iou(a, np.zeros((0, 4))).shape == (2, 0)


# ---------------------------------------------------------------- tracking


def test_new_track_gets_an_id_on_its_second_frame() -> None:
    t = PotholeTracker()
    assert _ids(t, [_pothole(100, 100)]) == []  # unconfirmed: no ID yet
    assert _ids(t, [_pothole(102, 104)]) == [1]


def test_one_frame_false_detection_never_gets_an_id() -> None:
    t = PotholeTracker()
    _ids(t, [_pothole(100, 100)])
    assert _ids(t, []) == []
    assert _ids(t, [_pothole(400, 400)]) == []  # a new candidate, not the old one
    assert _ids(t, [_pothole(401, 401)]) == [1]  # IDs have no gaps


def test_pothole_approaching_keeps_its_id() -> None:
    """Dashcam: a pothole slides down the frame and grows as the car gets closer."""
    t = PotholeTracker()
    ids = [_ids(t, [_pothole(300 - k, 200 + 8 * k, size=20 + 2 * k)]) for k in range(40)]
    assert ids[0] == [] and all(i == [1] for i in ids[1:])


def test_two_potholes_keep_separate_ids() -> None:
    t = PotholeTracker()
    seen: set[tuple[int, int]] = set()
    for k in range(20):
        left, right = _pothole(100 + 2 * k, 300 + 5 * k), _pothole(200 + 2 * k, 300 + 5 * k)
        swap = k % 2 == 1  # input order changes every frame
        ids = _ids(t, [right, left] if swap else [left, right])
        if k:
            seen.add((ids[1], ids[0]) if swap else (ids[0], ids[1]))  # (left, right)
    assert len(seen) == 1 and len(next(iter(seen))) == len(set(next(iter(seen))))


def test_short_gap_keeps_id_long_gap_gets_new_one() -> None:
    t = PotholeTracker(frame_rate=10.0, lost_seconds=1.0)  # 10 frames
    for k in range(3):
        _ids(t, [_pothole(100, 100 + 5 * k)])
    for _ in range(5):
        _ids(t, [])  # occluded for 0.5 s
    assert _ids(t, [_pothole(100, 140)]) == [1]  # Kalman predicted the motion
    for _ in range(12):
        _ids(t, [])  # gone for 1.2 s: forgotten
    _ids(t, [_pothole(100, 140)])
    assert _ids(t, [_pothole(100, 142)]) == [2]


def test_low_score_continues_a_track_but_never_starts_one() -> None:
    t = PotholeTracker(track_thresh=0.4)
    _ids(t, [_pothole(100, 100)])
    _ids(t, [_pothole(100, 102)])
    assert _ids(t, [_pothole(100, 104, score=0.3)]) == [1]  # score dip: same ID
    for _ in range(3):
        assert _ids(t, [_pothole(500, 500, score=0.3)]) == []


def test_lost_seconds_scales_with_frame_rate() -> None:
    assert PotholeTracker(frame_rate=30.0, lost_seconds=1.0).max_lost_frames == 30
    assert PotholeTracker(frame_rate=5.0, lost_seconds=1.0).max_lost_frames == 5


def test_min_hits_one_gives_ids_at_once() -> None:
    assert _ids(PotholeTracker(min_hits=1), [_pothole(0, 0), _pothole(100, 0)]) == [1, 2]


# ---------------------------------------------------------------- contract


def test_does_not_mutate_inputs_and_keeps_other_fields() -> None:
    t = PotholeTracker(min_hits=1)
    original = _pothole(10, 10)
    [tracked] = t.update([original])
    assert original.track_id == -1
    assert tracked.area_px == original.area_px and tracked.mask is original.mask


def test_empty_frame_returns_empty_list() -> None:
    assert PotholeTracker().update([]) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_rate": 0},
        {"track_thresh": 1.5},
        {"match_iou": -0.1},
        {"low_match_iou": 2},
        {"lost_seconds": -1},
        {"min_hits": 0},
    ],  # fmt: skip
)
def test_rejects_bad_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PotholeTracker(**kwargs)
