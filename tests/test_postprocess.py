import numpy as np
import pytest

from road_pothole_seg.postprocess import analyse_potholes, classify_severity, clean_mask
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


# ---------------------------------------------------------------- clean_mask


def test_clean_mask_drops_specks_but_keeps_big_parts() -> None:
    mask = _box_mask(100, 100, 10, 10, 40, 40)  # main part, 900 px
    mask[60:70, 60:75] = True  # second part, 150 px >= 0.1 x 900: kept
    mask[90, 90] = True  # speck, 1 px: dropped
    out = clean_mask(mask, min_part=0.1)
    assert out[10:40, 10:40].all() and out[60:70, 60:75].all()
    assert not out[90, 90]
    assert out.sum() == 900 + 150


def test_clean_mask_fills_holes_not_open_bays() -> None:
    mask = _box_mask(50, 50, 10, 10, 30, 30)
    mask[15:20, 15:20] = False  # enclosed hole: filled
    mask[25:30, 0:30] = False  # bite open to the outside: stays empty
    out = clean_mask(mask)
    assert out[15:20, 15:20].all()
    assert not out[25:30].any()


def test_clean_mask_keeps_a_lone_tiny_pothole_and_handles_empty() -> None:
    tiny = np.zeros((100, 100), dtype=bool)
    tiny[5, 5] = True
    assert clean_mask(tiny).sum() == 1  # the largest part is always kept
    assert not clean_mask(np.zeros((10, 10), dtype=bool)).any()


def test_clean_mask_does_not_change_its_input() -> None:
    mask = _box_mask(20, 20, 2, 2, 8, 8)
    mask[4, 4] = False
    before = mask.copy()
    clean_mask(mask)
    assert (mask == before).all()


def test_analyse_potholes_measures_the_cleaned_mask() -> None:
    mask = _box_mask(100, 100, 10, 10, 20, 20)
    mask[12, 12] = False  # hole
    mask[80, 80] = True  # speck
    [p] = analyse_potholes(Segmentation(road_mask=None, pothole_masks=[mask], scores=[0.9]))
    assert p.area_px == 100 and p.bbox_xyxy == (10, 10, 20, 20)


def test_rejects_bad_min_part() -> None:
    with pytest.raises(ValueError):
        clean_mask(np.ones((3, 3), dtype=bool), min_part=1.5)


def test_mask_bbox() -> None:
    from road_pothole_seg.postprocess import mask_bbox

    assert mask_bbox(_box_mask(50, 60, 5, 7, 20, 30)) == (5, 7, 20, 30)
    assert mask_bbox(np.zeros((4, 4), dtype=bool)) is None
