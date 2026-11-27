"""A fake ONNX Runtime session, so Segmenter tests run without a model file."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from road_pothole_seg import segmenter as segmenter_module

NAMES_META = "{0: 'road', 1: 'pothole'}"
N_PROTOS = 32


@dataclass
class FakeDet:
    """One detection in input-image pixels: its box and the region its mask covers."""

    cls: int
    score: float
    box: tuple[float, float, float, float]  # xyxy
    region: tuple[int, int, int, int] | None = None  # xyxy; default: box + 8 px (crop sets edges)


@dataclass
class FakeModel:
    size: int = 640
    dets: list[FakeDet] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=lambda: {"names": NAMES_META})
    shape: list[Any] | None = None  # input shape override
    calls: list[np.ndarray] = field(default_factory=list)

    def outputs(self) -> list[np.ndarray]:
        """output0 (1, 4+2+32, anchors) and protos (1, 32, size/4, size/4) for `dets`."""
        n_anchors = len(self.dets) + 3  # plus empty background anchors
        out = np.zeros((4 + 2 + N_PROTOS, n_anchors), dtype=np.float32)
        ph = self.size // 4
        protos = np.full((N_PROTOS, ph, ph), -10.0, dtype=np.float32)
        for i, d in enumerate(self.dets):
            x1, y1, x2, y2 = d.box
            out[:4, i] = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
            out[4 + d.cls, i] = d.score
            out[6 + i, i] = 1.0  # detection i uses prototype i
            rx1, ry1, rx2, ry2 = d.region or (
                max(int(x1) - 8, 0),
                max(int(y1) - 8, 0),
                int(x2) + 8,
                int(y2) + 8,
            )
            protos[i, ry1 // 4 : ry2 // 4, rx1 // 4 : rx2 // 4] = 10.0
        return [out[None], protos[None]]


class FakeSession:
    def __init__(self, model: FakeModel) -> None:
        self.model = model

    def get_inputs(self) -> list[SimpleNamespace]:
        s = self.model.size
        return [SimpleNamespace(name="images", shape=self.model.shape or [1, 3, s, s])]

    def get_modelmeta(self) -> SimpleNamespace:
        return SimpleNamespace(custom_metadata_map=self.model.meta)

    def run(self, _names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.model.calls.append(feeds["images"])
        return self.model.outputs()


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    """Every Segmenter created in the test uses this model instead of a real ONNX file."""
    model = FakeModel()
    monkeypatch.setattr(
        segmenter_module.ort, "InferenceSession", lambda *_a, **_k: FakeSession(model)
    )
    return model
