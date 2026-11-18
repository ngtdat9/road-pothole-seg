import numpy as np
import pytest

from road_pothole_seg.postprocess import analyse_potholes, classify_severity
from road_pothole_seg.types import Segmentation, SeverityBins


def _box_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_measures_box_area_and_centre() -> None:
    mask = _box_mask(100, 200, x0=10, y0=20, x1=30, y1=30)  # 20 x 10 px
    [p] = analyse_potholes(Segmentation(road_mask=None, pothole_masks=[mask], scores=[0.8]))
    assert p.bbox_xyxy == (10, 20, 30, 30)
    assert p.area_px == 200
    assert p.area_frac == pytest.approx(200 / 20_000)
    assert p.center_px == (20, 24)  # mean of pixel indices 10..29 and 20..29, rounded
    assert p.score == pytest.approx(0.8)
    assert p.track_id == -1
    assert p.area_m2 is None


def test_area_m2_only_with_metres_per_px() -> None:
    mask = _box_mask(100, 100, 0, 0, 10, 10)  # 100 px
    seg = Segmentation(road_mask=None, pothole_masks=[mask], scores=[0.9])
    [p] = analyse_potholes(seg, metres_per_px=0.02)
    assert p.area_m2 == pytest.approx(100 * 0.02**2)


def test_severity_uses_frac_bins_without_scale() -> None:
    bins = SeverityBins(frac=(0.01, 0.1), m2=(0.1, 0.5))
    small = _box_mask(100, 100, 0, 0, 5, 5)  # 0.0025
    medium = _box_mask(100, 100, 0, 0, 20, 20)  # 0.04
    large = _box_mask(100, 100, 0, 0, 50, 50)  # 0.25
    seg = Segmentation(road_mask=None, pothole_masks=[small, medium, large], scores=[1, 1, 1])
    assert [p.severity for p in analyse_potholes(seg, severity_bins=bins)] == [
        "small",
        "medium",
        "large",
    ]


def test_severity_uses_m2_bins_with_scale() -> None:
    # Same mask is "large" by frame fraction but "small" in m² at 1 cm per px.
    mask = _box_mask(100, 100, 0, 0, 50, 50)  # 2500 px -> 0.25 m² at 0.01 m/px
    seg = Segmentation(road_mask=None, pothole_masks=[mask], scores=[1.0])
    bins = SeverityBins(frac=(0.01, 0.1), m2=(0.3, 1.0))
    assert analyse_potholes(seg, severity_bins=bins)[0].severity == "large"
    assert analyse_potholes(seg, metres_per_px=0.01, severity_bins=bins)[0].severity == "small"


def test_drops_empty_masks() -> None:
    seg = Segmentation(road_mask=None, pothole_masks=[np.zeros((10, 10), dtype=bool)], scores=[0.9])
    assert analyse_potholes(seg) == []


def test_rejects_non_positive_scale() -> None:
    seg = Segmentation(road_mask=None)
    with pytest.raises(ValueError):
        analyse_potholes(seg, metres_per_px=0.0)


@pytest.mark.parametrize(
    ("value", "expected"), [(0.0, "small"), (0.1, "medium"), (0.49, "medium"), (0.5, "large")]
)
def test_classify_severity_boundaries(value: float, expected: str) -> None:
    assert classify_severity(value, (0.1, 0.5)) == expected
