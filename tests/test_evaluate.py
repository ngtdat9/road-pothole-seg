import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from training.scripts.evaluate import evaluate, fetch_preds, format_preds, preds_zip_name, table

NAMES = {0: "road", 1: "pothole"}
SQUARE = np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]])


class FakeYOLO:
    """Records val/predict calls; mimics the bits of Ultralytics that evaluate uses."""

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.vals: list[dict[str, Any]] = []
        self.predicted: list[str] = []
        FakeYOLO.last = self

    def val(self, **kwargs: Any) -> SimpleNamespace:
        val_list = yaml.safe_load(Path(kwargs["data"]).read_text())["val"]
        self.vals.append({"list": val_list, "imgsz": kwargs["imgsz"]})
        score = kwargs["imgsz"] / 1000  # smaller input, lower score
        seg = SimpleNamespace(
            ap_class_index=[0, 1], ap50=[score, score / 2], ap=[score / 2, score / 4]
        )
        return SimpleNamespace(seg=seg, names=NAMES)

    def predict(self, source: str, **kwargs: Any) -> list[SimpleNamespace]:
        self.predicted.append(Path(source).name)
        if Path(source).stem == "b":
            return [SimpleNamespace(masks=None, boxes=None)]
        masks = SimpleNamespace(xyn=[SQUARE, np.empty((0, 2))])  # 2nd mask too small to trace
        boxes = SimpleNamespace(cls=np.array([1.0, 0.0]), conf=np.array([0.8, 0.3]))
        return [SimpleNamespace(masks=masks, boxes=boxes)]


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "merged-v1"
    (root / "images").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": NAMES}))
    (root / "train.txt").write_text("./images/t.jpg\n")
    for split in ("val", "test"):
        (root / f"{split}.txt").write_text("./images/a.jpg\n./images/b.jpg\n")
        (root / f"{split}_real_road.txt").write_text("./images/a.jpg\n")
    return root


def test_format_preds_skips_untraceable_masks() -> None:
    text = format_preds(np.array([1, 0]), np.array([0.8, 0.3]), [SQUARE, np.empty((0, 2))])
    assert text.count("\n") == 1 and text.startswith("1 0.8000 0.10000 0.10000 ")
    assert format_preds(np.array([]), np.array([]), []) == ""


def test_evaluate_sizes_lists_and_predictions(data_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "eval"
    summary = evaluate(Path("best.pt"), data_root, out, [640, 320], preds_imgsz=640,
                       yolo_cls=FakeYOLO)  # fmt: skip
    model = FakeYOLO.last
    # Pothole on the whole split, road on real-road images only, at every size.
    assert model.vals == [
        {"list": "val.txt", "imgsz": 640},
        {"list": "val_real_road.txt", "imgsz": 640},
        {"list": "val.txt", "imgsz": 320},
        {"list": "val_real_road.txt", "imgsz": 320},
    ]
    assert summary["sizes"]["640"]["pothole"] == {"map50": 0.32, "map50_95": 0.16}
    assert summary["sizes"]["320"]["road"] == {"map50": 0.32, "map50_95": 0.16}
    assert model.predicted == ["a.jpg", "b.jpg"]
    assert json.loads((out / "eval_val.json").read_text())["predictions"]["images"] == 2
    assert "| 320 | 0.160 | 0.080 | 0.320 | 0.160 |" in table(summary)

    local = tmp_path / "local"
    assert fetch_preds(out / preds_zip_name("val", 640), local) == 2
    assert (local / "b.txt").read_text() == ""  # nothing found is still a file
    assert (local / "a.txt").read_text().startswith("1 0.8000 ")


def test_test_split_needs_final(data_root: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="final"):
        evaluate(Path("best.pt"), data_root, tmp_path, [640], split="test", yolo_cls=FakeYOLO)
    summary = evaluate(Path("best.pt"), data_root, tmp_path, [640], split="test", final=True,
                       yolo_cls=FakeYOLO)  # fmt: skip
    assert summary["metric_sources"] == {"pothole": "test.txt", "road": "test_real_road.txt"}
    with pytest.raises(SystemExit, match="final"):  # the naive set's clean test, too
        evaluate(Path("best.pt"), data_root, tmp_path, [640], split="test_clean",
                 yolo_cls=FakeYOLO)  # fmt: skip
    with pytest.raises(ValueError):
        evaluate(Path("best.pt"), data_root, tmp_path, [640], split="train", yolo_cls=FakeYOLO)


def test_fetch_preds_rejects_incomplete_zip(tmp_path: Path) -> None:
    path = tmp_path / "p.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.csv", "name,n\na,1\nb,0\n")
        z.writestr("a.txt", "")
    with pytest.raises(SystemExit, match="manifest"):
        fetch_preds(path, tmp_path / "local")
