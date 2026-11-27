"""Turn raw pothole masks into measured `Pothole` objects.

Each mask is cleaned first (`clean_mask`: drop specks, fill holes), then measured:
box, centre, area in pixels, as a fraction of the frame and, for top-down frames
with a known scale, in m²; severity comes from `SeverityBins`.
"""

from __future__ import annotations

import cv2
import numpy as np

from road_pothole_seg.types import Pothole, Segmentation, SeverityBins


def classify_severity(value: float, bounds: tuple[float, float]) -> str:
    """Bucket a size into "small", "medium" or "large" using (small_max, medium_max)."""
    small_max, medium_max = bounds
    if value < small_max:
        return "small"
    if value < medium_max:
        return "medium"
    return "large"


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the True pixels, end-exclusive; None for an empty mask.

    Uses row and column `any`, much faster than `np.nonzero` on a full-frame mask.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask[rows[0] : rows[-1] + 1].any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def clean_mask(mask: np.ndarray, min_part: float = 0.1) -> np.ndarray:
    """Return a cleaned copy of a bool mask: specks dropped, holes filled.

    The largest connected part is always kept; other parts are kept only if they are
    at least `min_part` times its area (a pothole can be split by a crack or a
    reflection). Holes (background not connected to the mask's box border) are filled,
    e.g. a water reflection inside a pothole. Works on the mask's bounding box only.
    """
    if not 0.0 <= min_part <= 1.0:
        raise ValueError(f"min_part must be in [0, 1], got {min_part}")
    out = np.zeros(mask.shape, dtype=bool)
    box = mask_bbox(mask)
    if box is None:
        return out
    x0, y0, x1, y1 = box
    crop = mask[y0:y1, x0:x1].astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]  # label 0 is background
    keep = 1 + np.flatnonzero(areas >= min_part * areas.max())
    kept = np.isin(labels, keep)

    # Fill holes: flood the background from a 1 px frame around the crop.
    padded = np.pad(kept.astype(np.uint8), 1)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)  # reaches all background connected to the border
    filled = (padded | (1 - flood)).astype(bool)[1:-1, 1:-1]
    out[y0:y1, x0:x1] = filled
    return out


def analyse_potholes(
    seg: Segmentation,
    metres_per_px: float | None = None,
    severity_bins: SeverityBins | None = None,
    min_part: float = 0.1,
) -> list[Pothole]:
    """Measure each pothole mask: box, centre, area (px, frame fraction, optional m²), severity.

    `metres_per_px` assumes a top-down camera; leave it None for perspective (dashcam) frames.
    Masks are cleaned with `clean_mask(mask, min_part)` first. Empty masks are dropped.
    `track_id` is -1 until `PotholeTracker` assigns one.
    """
    if metres_per_px is not None and metres_per_px <= 0:
        raise ValueError(f"metres_per_px must be positive, got {metres_per_px}")
    bins = severity_bins or SeverityBins()

    potholes: list[Pothole] = []
    for raw_mask, score in zip(seg.pothole_masks, seg.scores, strict=True):
        mask = clean_mask(raw_mask, min_part)
        box = mask_bbox(mask)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        xs, ys = xs + x0, ys + y0

        area_px = int(xs.size)
        area_frac = area_px / mask.size
        area_m2 = area_px * metres_per_px**2 if metres_per_px is not None else None
        severity = (
            classify_severity(area_m2, bins.m2)
            if area_m2 is not None
            else classify_severity(area_frac, bins.frac)
        )
        potholes.append(
            Pothole(
                track_id=-1,
                mask=mask,
                score=float(score),
                bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                area_px=area_px,
                area_frac=area_frac,
                area_m2=area_m2,
                severity=severity,
                center_px=(int(round(xs.mean())), int(round(ys.mean()))),
            )
        )
    return potholes
