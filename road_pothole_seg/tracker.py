"""Give each pothole one stable ID across frames.

STUB (Week 1): every detection gets a fresh ID, with no matching between frames.
ByteTrack (via supervision) replaces `update` in Week 4; the signature stays the same.
"""

from __future__ import annotations

import dataclasses
import itertools

from road_pothole_seg.types import Pothole


class PotholeTracker:
    """Assign `track_id` to potholes frame by frame. Create one tracker per video stream."""

    def __init__(self, frame_rate: float = 30.0) -> None:
        if frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {frame_rate}")
        self.frame_rate = frame_rate
        self._next_id = itertools.count(1)

    def update(self, potholes: list[Pothole]) -> list[Pothole]:
        """Return copies of `potholes` with `track_id` set (IDs start at 1); inputs unchanged."""
        # STUB: no matching; each detection is a new track.
        return [dataclasses.replace(p, track_id=next(self._next_id)) for p in potholes]
