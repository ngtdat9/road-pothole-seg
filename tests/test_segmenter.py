import numpy as np
import pytest

from road_pothole_seg.segmenter import Segmenter


@pytest.mark.parametrize("shape", [(480, 640), (1080, 1920), (720, 405), (5, 7)])
def test_masks_match_frame_size(shape: tuple[int, int]) -> None:
    """Any frame size or aspect ratio returns masks at that exact size."""
    seg = Segmenter("unused.onnx")(np.zeros((*shape, 3), dtype=np.uint8))
    assert seg.road_mask is not None
    assert seg.road_mask.shape == shape and seg.road_mask.dtype == bool
    for mask in seg.pothole_masks:
        assert mask.shape == shape and mask.dtype == bool
        assert mask.any()


def test_returns_one_score_per_pothole() -> None:
    seg = Segmenter("unused.onnx")(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(seg.scores) == len(seg.pothole_masks)
    assert all(0.0 <= s <= 1.0 for s in seg.scores)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((100, 100), dtype=np.uint8),  # grayscale
        np.zeros((100, 100, 4), dtype=np.uint8),  # BGRA
        np.zeros((100, 100, 3), dtype=np.float32),  # wrong dtype
        np.zeros((0, 100, 3), dtype=np.uint8),  # empty
    ],
)
def test_rejects_bad_frames(frame: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Segmenter("unused.onnx")(frame)


@pytest.mark.parametrize(
    "kwargs", [{"conf": 1.5}, {"conf": -0.1}, {"mask_threshold": 2.0}, {"input_size": 0}]
)
def test_rejects_bad_settings(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Segmenter("unused.onnx", **kwargs)
