"""ONNX Runtime segmenter.

STUB (Week 1): returns fixed fake masks so the pipeline and demo run end to end.
The real ONNX inference, letterboxing and mask decoding replace the body in Week 4;
the signature and output contract stay the same.
"""

from __future__ import annotations

import numpy as np

from road_pothole_seg.types import Segmentation


def validate_frame(frame: np.ndarray) -> None:
    """Raise ValueError unless `frame` is an HxWx3 uint8 BGR image."""
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"frame must be a numpy array, got {type(frame).__name__}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame must be uint8, got {frame.dtype}")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"frame must be non-empty, got shape {frame.shape}")


class Segmenter:
    """Segment road and potholes in any HxWx3 BGR frame; masks come back at frame size."""

    def __init__(
        self,
        model_path: str,
        input_size: int | None = None,
        conf: float = 0.25,
        mask_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"conf must be in [0, 1], got {conf}")
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError(f"mask_threshold must be in [0, 1], got {mask_threshold}")
        if input_size is not None and input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        self.model_path = model_path
        self.input_size = input_size
        self.conf = conf
        self.mask_threshold = mask_threshold

    def __call__(self, frame: np.ndarray) -> Segmentation:
        validate_frame(frame)
        h, w = frame.shape[:2]

        # STUB: bottom half is road, one pothole centred in the lower quarter.
        road = np.zeros((h, w), dtype=bool)
        road[h // 2 :, :] = True

        pothole = np.zeros((h, w), dtype=bool)
        ph, pw = max(1, h // 10), max(1, w // 8)
        top, left = (h * 3) // 4 - ph // 2, (w - pw) // 2
        pothole[top : top + ph, left : left + pw] = True

        return Segmentation(road_mask=road, pothole_masks=[pothole], scores=[0.9])
