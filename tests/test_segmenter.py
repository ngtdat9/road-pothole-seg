from pathlib import Path

import numpy as np
import pytest
from conftest import FakeDet, FakeModel

from road_pothole_seg.segmenter import (
    Segmenter,
    class_names,
    letterbox,
    letterbox_params,
    nms,
    to_tensor,
)

REAL_MODEL = Path(__file__).resolve().parents[1] / "models" / "baseline-640.onnx"


# ---------------------------------------------------------------- letterbox


@pytest.mark.parametrize(
    ("hw", "expected"),
    [
        ((640, 640), (1.0, 640, 640, 0, 0)),
        ((720, 1280), (0.5, 640, 360, 0, 140)),
        ((1280, 720), (0.5, 360, 640, 140, 0)),
        ((300, 400), (1.6, 640, 480, 0, 80)),  # small frames are scaled up
    ],
)
def test_letterbox_params(hw: tuple[int, int], expected: tuple) -> None:
    assert letterbox_params(*hw, 640) == pytest.approx(expected)


def test_letterbox_pads_with_grey_and_keeps_content() -> None:
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    image, (r, new_w, new_h, left, top) = letterbox(frame, 640)
    assert image.shape == (640, 640, 3)
    assert (image[:top] == 114).all() and (image[top + new_h :] == 114).all()
    assert (image[top : top + new_h] == 200).all()


def test_to_tensor_is_rgb_nchw_unit_range() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    t = to_tensor(image)
    assert t.shape == (1, 3, 4, 4) and t.dtype == np.float32
    assert (t[0, 2] == 1.0).all() and (t[0, 0] == 0.0).all()  # blue is the last RGB channel


# ---------------------------------------------------------------- NMS and metadata


def test_nms_drops_overlaps_above_iou_and_keeps_score_order() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.5], dtype=np.float32)
    assert nms(boxes, scores, iou=0.5).tolist() == [1, 2]
    assert nms(boxes, scores, iou=0.9).tolist() == [1, 0, 2]  # IoU 0.68 is kept at 0.9


def test_class_names_from_metadata() -> None:
    assert class_names({"names": "{0: 'road', 1: 'pothole'}"}) == ["road", "pothole"]
    with pytest.raises(ValueError, match="names"):
        class_names({})


# ---------------------------------------------------------------- Segmenter (fake model)


def _frame(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("shape", [(480, 640), (1080, 1920), (720, 405), (5, 7)])
def test_masks_match_frame_size(fake_model: FakeModel, shape: tuple[int, int]) -> None:
    """Any frame size or aspect ratio returns masks at that exact size."""
    fake_model.dets = [FakeDet(0, 0.9, (0, 320, 640, 640)), FakeDet(1, 0.8, (300, 400, 360, 460))]
    seg = Segmenter("fake.onnx")(_frame(*shape))
    assert seg.road_mask.shape == shape and seg.road_mask.dtype == bool
    for mask in seg.pothole_masks:
        assert mask.shape == shape and mask.dtype == bool


def test_masks_map_back_through_the_letterbox(fake_model: FakeModel) -> None:
    """A 1280x720 frame sits at input rows 140..500; a box there maps to frame pixels x2."""
    fake_model.dets = [FakeDet(1, 0.8, (100, 240, 200, 300))]  # input pixels
    seg = Segmenter("fake.onnx", fill_road=False)(_frame(720, 1280))
    ys, xs = np.nonzero(seg.pothole_masks[0])
    assert (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) == (200, 400, 200, 320)
    assert seg.scores == [pytest.approx(0.8)]
    assert not seg.road_mask.any()  # no road detected: all False, not None


def test_mask_is_cropped_to_its_box(fake_model: FakeModel) -> None:
    fake_model.dets = [FakeDet(1, 0.8, (100, 100, 200, 200), region=(0, 0, 640, 640))]
    mask = Segmenter("fake.onnx")(_frame(640, 640)).pothole_masks[0]
    assert mask.sum() == 100 * 100 and mask[100:200, 100:200].all()


def test_road_is_union_of_road_instances_and_filled_with_potholes(fake_model: FakeModel) -> None:
    fake_model.dets = [
        FakeDet(0, 0.9, (0, 400, 320, 640)),
        FakeDet(0, 0.7, (320, 400, 640, 640)),
        FakeDet(1, 0.8, (300, 200, 340, 240)),  # pothole above the road
    ]
    filled = Segmenter("fake.onnx")(_frame(640, 640))
    raw = Segmenter("fake.onnx", fill_road=False)(_frame(640, 640))
    assert raw.road_mask[400:].all() and not raw.road_mask[:400].any()
    assert filled.road_mask[200:240, 300:340].all()
    assert len(filled.pothole_masks) == 1


def test_conf_and_nms_filter_detections(fake_model: FakeModel) -> None:
    fake_model.dets = [
        FakeDet(1, 0.9, (100, 100, 200, 200)),
        FakeDet(1, 0.6, (105, 105, 205, 205)),  # duplicate of the first: NMS drops it
        FakeDet(0, 0.6, (105, 105, 205, 205)),  # same box, other class: kept
        FakeDet(1, 0.2, (400, 400, 500, 500)),  # below conf
    ]
    seg = Segmenter("fake.onnx", fill_road=False)(_frame(640, 640))
    assert seg.scores == [pytest.approx(0.9)]
    assert seg.road_mask.any()


def test_empty_masks_are_dropped(fake_model: FakeModel) -> None:
    fake_model.dets = [FakeDet(1, 0.9, (100, 100, 200, 200), region=(400, 400, 500, 500))]
    assert Segmenter("fake.onnx")(_frame(640, 640)).pothole_masks == []


def test_records_stage_timings(fake_model: FakeModel) -> None:
    s = Segmenter("fake.onnx")
    s(_frame(100, 100))
    assert set(s.last_timings_ms) == {"preprocess", "inference", "postprocess"}
    assert fake_model.calls[0].shape == (1, 3, 640, 640)


def test_input_size_comes_from_the_model(fake_model: FakeModel) -> None:
    fake_model.size = 320
    assert Segmenter("fake.onnx").input_size == 320
    assert Segmenter("fake.onnx", input_size=320).input_size == 320
    with pytest.raises(ValueError, match="conflicts"):
        Segmenter("fake.onnx", input_size=640)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"shape": [1, 3, "h", "w"]}, "fixed shape"),
        ({"shape": [1, 3, 640, 480]}, "square"),
        ({"meta": {}}, "names"),
        ({"meta": {"names": "{0: 'car', 1: 'pothole'}"}}, "road"),
    ],
)
def test_rejects_unsuitable_models(fake_model: FakeModel, override: dict, match: str) -> None:
    for key, value in override.items():
        setattr(fake_model, key, value)
    with pytest.raises(ValueError, match=match):
        Segmenter("fake.onnx")


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((100, 100), dtype=np.uint8),  # grayscale
        np.zeros((100, 100, 4), dtype=np.uint8),  # BGRA
        np.zeros((100, 100, 3), dtype=np.float32),  # wrong dtype
        np.zeros((0, 100, 3), dtype=np.uint8),  # empty
    ],
)
def test_rejects_bad_frames(fake_model: FakeModel, frame: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Segmenter("fake.onnx")(frame)


@pytest.mark.parametrize(
    "kwargs",
    [{"conf": 1.5}, {"conf": -0.1}, {"mask_threshold": 2.0}, {"iou": 1.1}, {"input_size": 0}],
)
def test_rejects_bad_settings(fake_model: FakeModel, kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Segmenter("fake.onnx", **kwargs)


# ---------------------------------------------------------------- the real model


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="models/baseline-640.onnx not exported")
def test_real_model_runs_on_a_non_square_frame() -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    seg = Segmenter(str(REAL_MODEL))(frame)
    assert seg.road_mask.shape == (720, 1280)
    assert len(seg.pothole_masks) == len(seg.scores)
