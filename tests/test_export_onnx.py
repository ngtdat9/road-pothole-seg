import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from training.scripts.export_onnx import REPO_ROOT, ExportSettings, export, load_settings


class FakeYOLO:
    """Writes a dummy .onnx next to the weights, like Ultralytics' export."""

    calls: list[dict[str, Any]] = []

    def __init__(self, weights: str) -> None:
        self.weights = Path(weights)

    def export(self, **kwargs: Any) -> str:
        FakeYOLO.calls.append(kwargs)
        out = self.weights.with_suffix(".onnx")
        out.write_bytes(f"model {kwargs['imgsz']}".encode())
        return str(out)


def _describe(path: Path) -> dict[str, Any]:
    size = int(path.stem.split("-")[-1])
    return {"inputs": {"images": [1, 3, size, size]}, "outputs": {}, "metadata": {}}


def test_export_writes_one_model_per_size_and_a_report(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    s = ExportSettings(name="baseline", sizes=[640, 320], opset=17, simplify=True,
                       out_dir=tmp_path / "models")  # fmt: skip
    FakeYOLO.calls = []
    report = export(weights, s, yolo_cls=FakeYOLO, describe_fn=_describe)

    assert [c["imgsz"] for c in FakeYOLO.calls] == [640, 320]
    assert all(c["dynamic"] is False and c["opset"] == 17 for c in FakeYOLO.calls)
    assert (s.out_dir / "baseline-640.onnx").read_bytes() == b"model 640"
    assert not (tmp_path / "best.onnx").exists()  # moved, not copied
    saved = json.loads((s.out_dir / "baseline.json").read_text())
    assert saved == report
    assert saved["models"]["320"]["inputs"]["images"] == [1, 3, 320, 320]
    assert len(saved["weights_sha256"]) == 64


def test_repo_config_loads() -> None:
    s = load_settings(REPO_ROOT / "configs" / "export.yaml")
    assert s.sizes == [640, 480, 320]
    assert s.out_dir == REPO_ROOT / "models"


def test_sizes_must_be_stride_multiples(tmp_path: Path) -> None:
    cfg = tmp_path / "export.yaml"
    cfg.write_text(yaml.safe_dump({"export": {"name": "m", "sizes": [600], "opset": 17,
                                              "simplify": True, "out_dir": "models"}}))  # fmt: skip
    with pytest.raises(ValueError, match="multiples of 32"):
        load_settings(cfg)
