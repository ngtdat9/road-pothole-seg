"""Data classes returned by the public API (frozen at v1.0)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEVERITY_LEVELS: tuple[str, str, str] = ("small", "medium", "large")


@dataclass
class Segmentation:
    """Raw model output for one frame. All masks are HxW bool at the original frame size."""

    road_mask: np.ndarray | None
    pothole_masks: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.pothole_masks) != len(self.scores):
            raise ValueError(
                f"{len(self.pothole_masks)} pothole masks but {len(self.scores)} scores"
            )


@dataclass
class Pothole:
    """One analysed pothole. `track_id` is -1 until `PotholeTracker` assigns one."""

    track_id: int
    mask: np.ndarray
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    area_px: int
    area_frac: float
    area_m2: float | None
    severity: str
    center_px: tuple[int, int]

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_LEVELS:
            raise ValueError(f"severity must be one of {SEVERITY_LEVELS}, got {self.severity!r}")


@dataclass(frozen=True)
class SeverityBins:
    """Upper bounds of "small" and "medium"; anything above the second bound is "large".

    `frac` bins apply to `area_frac` and are used when no `metres_per_px` is given; the
    defaults are the tertiles of labelled pothole area in merged-v1 train (equal thirds).
    `m2` bins apply to `area_m2` and are used when it is. Defaults must equal
    `configs/pipeline.yaml`.
    """

    frac: tuple[float, float] = (0.0012, 0.0066)
    m2: tuple[float, float] = (0.1, 0.5)

    def __post_init__(self) -> None:
        for name in ("frac", "m2"):
            lo, hi = getattr(self, name)
            if not 0 < lo < hi:
                raise ValueError(f"{name} bins must satisfy 0 < small < medium, got {(lo, hi)}")
