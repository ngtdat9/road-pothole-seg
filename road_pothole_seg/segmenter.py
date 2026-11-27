"""ONNX Runtime segmenter: letterbox, inference, NMS and mask decoding.

Pre- and post-processing follow Ultralytics 8.4 `predict` so the masks match the
training framework (checked by training/scripts/check_onnx.py):
- letterbox: scale to fit, INTER_LINEAR, centred grey (114) padding, BGR -> RGB, / 255,
- output0 (1, 4 + classes + 32, anchors): centre-xywh boxes in input pixels, class
  scores already sigmoid, then mask coefficients; output1 (1, 32, mh, mw): prototypes,
- best class per anchor, `conf` filter, class-aware NMS at `iou`, at most 300 kept,
- mask = sigmoid(coefficients @ prototypes) upsampled to the input, padding removed,
  resized to the frame, thresholded, then cropped to the box.
"""

from __future__ import annotations

import ast
import math
import time

import cv2
import numpy as np
import onnxruntime as ort

from road_pothole_seg.types import Segmentation

ROAD, POTHOLE = "road", "pothole"
PAD_VALUE = 114  # Ultralytics' letterbox grey
MAX_DET = 300  # Ultralytics' cap on detections per image
MAX_WH = 7680  # class offset for class-aware NMS, as in Ultralytics
CV_MAX_CHANNELS = 512  # cv2.resize works on at most this many channels at once


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


# ---------------------------------------------------------------- pre-processing


def letterbox_params(h: int, w: int, size: int) -> tuple[float, int, int, int, int]:
    """(scale, resized_w, resized_h, left, top) for fitting an h x w frame into size x size."""
    r = min(size / h, size / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    left = round((size - new_w) / 2 - 0.1)
    top = round((size - new_h) / 2 - 0.1)
    return r, new_w, new_h, left, top


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
    """Fit `frame` into size x size with central grey padding; returns image and params."""
    h, w = frame.shape[:2]
    params = letterbox_params(h, w, size)
    _, new_w, new_h, left, top = params
    resized = frame if (w, h) == (new_w, new_h) else cv2.resize(
        frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )  # fmt: skip
    out = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    out[top : top + new_h, left : left + new_w] = resized
    return out, params


def to_tensor(image: np.ndarray) -> np.ndarray:
    """HxWx3 BGR uint8 -> 1x3xHxW RGB float32 in [0, 1]."""
    rgb = image[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb, dtype=np.float32)[None] / np.float32(255.0)


# ---------------------------------------------------------------- post-processing


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS on xyxy boxes; indices kept, highest score first (torchvision semantics)."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = (np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(0)
        ih = (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(0)
        inter = iw * ih
        ious = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[ious <= iou]
    return np.asarray(keep, dtype=np.int64)


def select_detections(
    output0: np.ndarray, n_classes: int, conf: float, iou: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw output0 (4 + classes + coefs, anchors) -> xyxy boxes, scores, classes, coefs."""
    pred = output0.T  # anchors x channels
    cls_scores = pred[:, 4 : 4 + n_classes]
    classes = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(pred)), classes]
    keep = scores > conf
    pred, scores, classes = pred[keep], scores[keep], classes[keep]

    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    kept = nms(boxes + (classes * MAX_WH)[:, None], scores, iou)[:MAX_DET]
    return boxes[kept], scores[kept], classes[kept], pred[kept, 4 + n_classes :]


def _resize_stack(stack: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize of an HxWxN float stack (any N) to height x width x N."""
    parts = [
        cv2.resize(stack[:, :, i : i + CV_MAX_CHANNELS], (width, height),
                   interpolation=cv2.INTER_LINEAR).reshape(height, width, -1)
        for i in range(0, stack.shape[2], CV_MAX_CHANNELS)
    ]  # fmt: skip
    return np.concatenate(parts, axis=2)


def decode_masks(
    coefs: np.ndarray,
    protos: np.ndarray,
    boxes: np.ndarray,
    params: tuple[float, int, int, int, int],
    size: int,
    frame_hw: tuple[int, int],
    mask_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """N masks (N x H x W bool, frame size) and N boxes (frame pixels, xyxy float)."""
    h, w = frame_hw
    r, new_w, new_h, left, top = params
    frame_boxes = (boxes - [left, top, left, top]) / r
    frame_boxes = frame_boxes.clip(0, [w, h, w, h])
    if len(coefs) == 0:
        return np.zeros((0, h, w), dtype=bool), frame_boxes

    c, mh, mw = protos.shape
    logits = (coefs @ protos.reshape(c, -1)).reshape(-1, mh, mw).transpose(1, 2, 0)
    logits = _resize_stack(np.ascontiguousarray(logits), size, size)  # to input size
    logits = logits[top : top + new_h, left : left + new_w]  # drop the padding
    logits = _resize_stack(np.ascontiguousarray(logits), w, h)  # to frame size

    t = min(max(mask_threshold, 1e-6), 1 - 1e-6)
    masks = (logits > math.log(t / (1 - t))).transpose(2, 0, 1)  # sigmoid(x) > t
    cols, rows = np.arange(w), np.arange(h)
    for m, (x1, y1, x2, y2) in zip(masks, frame_boxes, strict=True):
        m[:, (cols < x1) | (cols >= x2)] = False
        m[(rows < y1) | (rows >= y2), :] = False
    return masks, frame_boxes


def class_names(meta: dict[str, str]) -> list[str]:
    """Class names from Ultralytics ONNX metadata, in index order."""
    if "names" not in meta:
        raise ValueError("model has no 'names' metadata; export it with Ultralytics")
    names = ast.literal_eval(meta["names"])
    return [names[i] for i in sorted(names)]


# ---------------------------------------------------------------- the segmenter


class Segmenter:
    """Segment road and potholes in any HxWx3 BGR frame; masks come back at frame size.

    `input_size=None` reads the size from the model; an explicit size must match it.
    `iou` is the NMS overlap above which the weaker of two same-class boxes is dropped.
    With `fill_road`, potholes are added to the road mask (road = road ∪ potholes), so
    the road has no holes where it is damaged. With no road found, the road mask is
    all False. After each call, `last_timings_ms` holds per-stage latency.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int | None = None,
        conf: float = 0.25,
        mask_threshold: float = 0.5,
        iou: float = 0.5,
        fill_road: bool = True,
    ) -> None:
        for name, value in (("conf", conf), ("mask_threshold", mask_threshold), ("iou", iou)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if input_size is not None and input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        model_h, model_w = inp.shape[2], inp.shape[3]
        if not (isinstance(model_h, int) and isinstance(model_w, int)):
            raise ValueError(f"model input must have a fixed shape, got {inp.shape}")
        if model_h != model_w:
            raise ValueError(f"model input must be square, got {model_h}x{model_w}")
        if input_size is not None and input_size != model_h:
            raise ValueError(f"input_size={input_size} conflicts with the model's {model_h}")

        names = class_names(self.session.get_modelmeta().custom_metadata_map)
        if ROAD not in names or POTHOLE not in names:
            raise ValueError(f"model classes must include {ROAD!r} and {POTHOLE!r}, got {names}")
        self.names = names
        self._road, self._pothole = names.index(ROAD), names.index(POTHOLE)

        self.model_path = model_path
        self.input_name = inp.name
        self.input_size = model_h
        self.conf = conf
        self.mask_threshold = mask_threshold
        self.iou = iou
        self.fill_road = fill_road
        self.last_timings_ms: dict[str, float] = {}

    def __call__(self, frame: np.ndarray) -> Segmentation:
        validate_frame(frame)
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        image, params = letterbox(frame, self.input_size)
        tensor = to_tensor(image)
        t1 = time.perf_counter()
        output0, protos = self.session.run(None, {self.input_name: tensor})[:2]
        t2 = time.perf_counter()

        boxes, scores, classes, coefs = select_detections(
            output0[0], len(self.names), self.conf, self.iou
        )
        masks, _ = decode_masks(
            coefs, protos[0], boxes, params, self.input_size, (h, w), self.mask_threshold
        )
        nonempty = masks.any(axis=(1, 2))
        road = np.zeros((h, w), dtype=bool)
        pothole_masks: list[np.ndarray] = []
        pothole_scores: list[float] = []
        for mask, score, cls in zip(masks[nonempty], scores[nonempty], classes[nonempty],
                                    strict=True):  # fmt: skip
            if cls == self._road:
                road |= mask
            elif cls == self._pothole:
                pothole_masks.append(mask)
                pothole_scores.append(float(score))
        if self.fill_road:
            for mask in pothole_masks:
                road |= mask
        t3 = time.perf_counter()

        self.last_timings_ms = {
            "preprocess": (t1 - t0) * 1e3,
            "inference": (t2 - t1) * 1e3,
            "postprocess": (t3 - t2) * 1e3,
        }
        return Segmentation(road_mask=road, pothole_masks=pothole_masks, scores=pothole_scores)
