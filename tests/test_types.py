import numpy as np
import pytest

from road_pothole_seg import Pothole, Segmentation, SeverityBins


def _pothole(**overrides: object) -> Pothole:
    fields: dict[str, object] = {
        "track_id": -1,
        "mask": np.zeros((4, 6), dtype=bool),
        "score": 0.9,
        "bbox_xyxy": (0, 0, 2, 2),
        "area_px": 4,
        "area_frac": 4 / 24,
        "area_m2": None,
        "severity": "small",
        "center_px": (1, 1),
    }
    fields.update(overrides)
    return Pothole(**fields)  # type: ignore[arg-type]


def test_segmentation_defaults_to_no_potholes() -> None:
    seg = Segmentation(road_mask=np.zeros((4, 6), dtype=bool))
    assert seg.pothole_masks == [] and seg.scores == []


def test_segmentation_rejects_mismatched_scores() -> None:
    with pytest.raises(ValueError):
        Segmentation(road_mask=None, pothole_masks=[np.zeros((2, 2), bool)], scores=[])


def test_pothole_accepts_valid_severity() -> None:
    assert _pothole(severity="large").severity == "large"


def test_pothole_rejects_unknown_severity() -> None:
    with pytest.raises(ValueError):
        _pothole(severity="huge")


def test_severity_bins_defaults_are_valid() -> None:
    bins = SeverityBins()
    assert bins.frac[0] < bins.frac[1] and bins.m2[0] < bins.m2[1]


@pytest.mark.parametrize("bad", [(0.5, 0.1), (0.0, 0.1), (0.1, 0.1)])
def test_severity_bins_reject_bad_order(bad: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        SeverityBins(m2=bad)
