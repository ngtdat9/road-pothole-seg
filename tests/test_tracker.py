import numpy as np
import pytest

from road_pothole_seg.tracker import PotholeTracker
from road_pothole_seg.types import Pothole


def _pothole() -> Pothole:
    return Pothole(
        track_id=-1,
        mask=np.ones((2, 2), dtype=bool),
        score=0.9,
        bbox_xyxy=(0, 0, 2, 2),
        area_px=4,
        area_frac=1.0,
        area_m2=None,
        severity="large",
        center_px=(1, 1),
    )


def test_assigns_positive_unique_ids() -> None:
    tracked = PotholeTracker().update([_pothole(), _pothole()])
    ids = [p.track_id for p in tracked]
    assert all(i > 0 for i in ids) and len(set(ids)) == 2


def test_does_not_mutate_inputs() -> None:
    original = _pothole()
    PotholeTracker().update([original])
    assert original.track_id == -1


def test_keeps_other_fields() -> None:
    original = _pothole()
    [tracked] = PotholeTracker().update([original])
    assert tracked.area_px == original.area_px and tracked.mask is original.mask


def test_empty_frame_returns_empty_list() -> None:
    assert PotholeTracker().update([]) == []


def test_rejects_non_positive_frame_rate() -> None:
    with pytest.raises(ValueError):
        PotholeTracker(frame_rate=0)
