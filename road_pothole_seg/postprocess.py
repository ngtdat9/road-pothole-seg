"""Turn raw pothole masks into measured `Pothole` objects.

Area, box, centre and severity are real. Mask cleanup (removing specks, filling
holes) is a pass-through for now and is added in Week 4.
"""

from __future__ import annotations

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


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Return a cleaned copy of a bool mask. Pass-through until Week 4."""
    return mask.astype(bool, copy=True)


def analyse_potholes(
    seg: Segmentation,
    metres_per_px: float | None = None,
    severity_bins: SeverityBins | None = None,
) -> list[Pothole]:
    """Measure each pothole mask: box, centre, area (px, frame fraction, optional m²), severity.

    `metres_per_px` assumes a top-down camera; leave it None for perspective (dashcam) frames.
    Empty masks are dropped. `track_id` is -1 until `PotholeTracker` assigns one.
    """
    if metres_per_px is not None and metres_per_px <= 0:
        raise ValueError(f"metres_per_px must be positive, got {metres_per_px}")
    bins = severity_bins or SeverityBins()

    potholes: list[Pothole] = []
    for raw_mask, score in zip(seg.pothole_masks, seg.scores, strict=True):
        mask = clean_mask(raw_mask)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue

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
