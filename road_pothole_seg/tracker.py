"""Give each pothole one stable ID across frames: ByteTrack in numpy.

ByteTrack (Zhang et al., ECCV 2022) with a constant-velocity Kalman filter on the box
(centre x, centre y, aspect ratio, height). Each `update`:
1. predicts every track one frame ahead,
2. matches confident detections (score >= `track_thresh`) to confirmed and lost tracks
   by box IoU (>= `match_iou`),
3. matches the remaining low-score detections to still-tracked tracks (>= `low_match_iou`),
   so a pothole whose score dips for a frame keeps its ID,
4. matches leftover confident detections to unconfirmed tracks; anything else confident
   starts a new unconfirmed track,
5. marks unmatched tracks lost and forgets them after `lost_seconds`.

Boxes are grown by `match_buffer` x their size on each side before IoU (buffered IoU),
so small, fast-moving potholes and low frame rates still overlap between frames.
A track gets its ID once it has been matched in `min_hits` frames, so one-frame false
detections never get an ID and IDs count up 1, 2, 3 without gaps. Matching is greedy
by IoU (highest first), which equals the optimal assignment when overlaps are clear,
the usual case for a handful of potholes per frame.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from road_pothole_seg.types import Pothole

# Kalman noise, as a fraction of box height (ByteTrack's values).
STD_POSITION = 1.0 / 20
STD_VELOCITY = 1.0 / 160


class KalmanBox:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height) and their velocities."""

    def __init__(self, xyah: np.ndarray) -> None:
        self.mean = np.r_[xyah, np.zeros(4)]
        h = xyah[3]
        pos, vel = 2 * STD_POSITION * h, 10 * STD_VELOCITY * h
        std = [pos, pos, 1e-2, pos, vel, vel, 1e-5, vel]
        self.cov = np.diag(np.square(std))
        self._motion = np.eye(8)
        self._motion[:4, 4:] = np.eye(4)

    def predict(self) -> None:
        pos, vel = STD_POSITION * self.mean[3], STD_VELOCITY * self.mean[3]
        std = [pos, pos, 1e-2, pos, vel, vel, 1e-5, vel]
        self.mean = self._motion @ self.mean
        self.cov = self._motion @ self.cov @ self._motion.T + np.diag(np.square(std))

    def update(self, xyah: np.ndarray) -> None:
        h = self.mean[3]
        noise = np.diag(np.square([STD_POSITION * h, STD_POSITION * h, 1e-1, STD_POSITION * h]))
        proj_cov = self.cov[:4, :4] + noise
        gain = np.linalg.solve(proj_cov, self.cov[:4, :]).T  # cov H^T S^-1
        self.mean = self.mean + gain @ (xyah - self.mean[:4])
        self.cov = self.cov - gain @ proj_cov @ gain.T

    def box(self) -> np.ndarray:
        """Predicted box as xyxy."""
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def to_xyah(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between xyxy boxes a (N x 4) and b (M x 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod((rb - lt).clip(0), axis=2)
    area_a = np.prod((a[:, 2:] - a[:, :2]).clip(0), axis=1)
    area_b = np.prod((b[:, 2:] - b[:, :2]).clip(0), axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def expand(boxes: np.ndarray, buffer: float) -> np.ndarray:
    """Grow xyxy boxes by `buffer` x their width/height on each side (buffered IoU)."""
    if buffer == 0:
        return boxes
    wh = boxes[:, 2:] - boxes[:, :2]
    return np.hstack([boxes[:, :2] - buffer * wh, boxes[:, 2:] + buffer * wh])


def greedy_match(iou: np.ndarray, min_iou: float) -> list[tuple[int, int]]:
    """(row, col) pairs, highest IoU first, each row and column used once, IoU >= min_iou."""
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    used_r, used_c = set(), set()
    for flat in np.argsort(-iou, axis=None, kind="stable"):
        r, c = divmod(int(flat), iou.shape[1])
        if iou[r, c] < min_iou:
            break
        if r not in used_r and c not in used_c:
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
    return pairs


@dataclasses.dataclass
class _Track:
    kf: KalmanBox
    hits: int = 1
    track_id: int = -1  # -1 until confirmed
    lost_for: int = 0  # frames since the last match
    first_frame: int = 0  # update() call (0-based) that first saw it

    @property
    def confirmed(self) -> bool:
        return self.track_id > 0


class PotholeTracker:
    """Assign `track_id` to potholes frame by frame. Create one tracker per video stream.

    `frame_rate` converts `lost_seconds` into frames (dashcam ~30 FPS, drone 5-10 Hz).
    """

    def __init__(
        self,
        frame_rate: float = 30.0,
        track_thresh: float = 0.4,
        match_iou: float = 0.2,
        low_match_iou: float = 0.5,
        lost_seconds: float = 1.0,
        min_hits: int = 2,
        match_buffer: float = 0.3,
    ) -> None:
        if frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {frame_rate}")
        for name, value in (("track_thresh", track_thresh), ("match_iou", match_iou),
                            ("low_match_iou", low_match_iou)):  # fmt: skip
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if lost_seconds < 0:
            raise ValueError(f"lost_seconds must be >= 0, got {lost_seconds}")
        if min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {min_hits}")
        if match_buffer < 0:
            raise ValueError(f"match_buffer must be >= 0, got {match_buffer}")
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.match_iou = match_iou
        self.low_match_iou = low_match_iou
        self.lost_seconds = lost_seconds
        self.min_hits = min_hits
        self.match_buffer = match_buffer
        self.max_lost_frames = math.ceil(lost_seconds * frame_rate)
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._frame = -1  # index of the current update() call
        self._first_seen: dict[int, int] = {}  # track_id -> first frame seen

    def update(self, potholes: list[Pothole]) -> list[Pothole]:
        """Return copies of the potholes that belong to a confirmed track, with `track_id` set.

        The result can be shorter than the input: detections that start a new track wait
        `min_hits` frames for an ID, and low-score ones that continue no track are dropped.
        Order follows the input; the inputs are not changed.
        """
        self._frame += 1
        for t in self._tracks:
            t.kf.predict()
        boxes = np.array([p.bbox_xyxy for p in potholes], dtype=float).reshape(-1, 4)
        high = [i for i, p in enumerate(potholes) if p.score >= self.track_thresh]
        low = [i for i, p in enumerate(potholes) if p.score < self.track_thresh]
        matched: dict[int, _Track] = {}  # detection index -> track

        def match(tracks: list[_Track], dets: list[int], min_iou: float) -> tuple[list, list]:
            predicted = np.array([t.kf.box() for t in tracks]).reshape(-1, 4)
            ious = box_iou(
                expand(predicted, self.match_buffer), expand(boxes[dets], self.match_buffer)
            )
            pairs = greedy_match(ious, min_iou)
            for r, c in pairs:
                matched[dets[c]] = tracks[r]
            used_r, used_c = {r for r, _ in pairs}, {c for _, c in pairs}
            return ([t for r, t in enumerate(tracks) if r not in used_r],
                    [d for c, d in enumerate(dets) if c not in used_c])  # fmt: skip

        confirmed = [t for t in self._tracks if t.confirmed]
        unconfirmed = [t for t in self._tracks if not t.confirmed]
        left_tracks, left_high = match(confirmed, high, self.match_iou)
        still_tracked = [t for t in left_tracks if t.lost_for == 0]
        missed, _ = match(still_tracked, low, self.low_match_iou)
        missed += [t for t in left_tracks if t.lost_for > 0]
        stale_unconfirmed, new = match(unconfirmed, left_high, self.match_iou)

        for i, t in matched.items():
            t.kf.update(to_xyah(potholes[i].bbox_xyxy))
            t.hits += 1
            t.lost_for = 0
        for t in missed:
            t.lost_for += 1
        gone = {id(t) for t in stale_unconfirmed}
        gone |= {id(t) for t in missed if t.lost_for > self.max_lost_frames}
        self._tracks = [t for t in self._tracks if id(t) not in gone]
        for i in new:
            t = _Track(KalmanBox(to_xyah(potholes[i].bbox_xyxy)), first_frame=self._frame)
            self._tracks.append(t)
            matched[i] = t

        for i in sorted(matched):  # IDs in input order, so they count up predictably
            t = matched[i]
            if not t.confirmed and t.hits >= self.min_hits:
                t.track_id = self._next_id
                self._first_seen[t.track_id] = t.first_frame
                self._next_id += 1
        return [
            dataclasses.replace(p, track_id=matched[i].track_id)
            for i, p in enumerate(potholes)
            if i in matched and matched[i].confirmed
        ]

    def first_seen(self, track_id: int) -> int:
        """Frame index (count of `update` calls, from 0) where this track was first detected.

        Earlier than the frame where it got its ID when `min_hits` > 1. KeyError if unknown.
        """
        return self._first_seen[track_id]
