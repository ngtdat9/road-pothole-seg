import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from training.scripts.train import (
    REPO_ROOT,
    check_profile,
    class_metrics,
    load_profile,
    results_row,
    run,
    train_args,
    write_data_yaml,
)

NAMES = {0: "road", 1: "pothole"}


def _seg(classes: list[int], ap50: list[float], ap: list[float]) -> SimpleNamespace:
    return SimpleNamespace(ap_class_index=classes, ap50=ap50, ap=ap)


class FakeYOLO:
    """Records calls instead of training; mimics the bits of Ultralytics we use."""

    calls: list[tuple[str, str, dict[str, Any]]] = []

    def __init__(self, weights: str) -> None:
        self.weights = weights

    def train(self, **kwargs: Any) -> None:
        FakeYOLO.calls.append(("train", self.weights, kwargs))

    def val(self, **kwargs: Any) -> SimpleNamespace:
        FakeYOLO.calls.append(("val", self.weights, kwargs))
        data = yaml.safe_load(Path(kwargs["data"]).read_text())
        if data["val"] == "val_real_road.txt":
            seg = _seg([0, 1], [0.9, 0.5], [0.7, 0.3])
        else:  # mixed val: road AP is polluted by unlabelled road, must not be used
            seg = _seg([0, 1], [0.1, 0.6], [0.05, 0.35])
        return SimpleNamespace(seg=seg, names=NAMES)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "merged-v1-base"
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"names": NAMES}))
    for name in ("train", "val", "test", "train_real_road", "val_real_road"):
        (root / f"{name}.txt").write_text("./images/a.jpg\n")
    return root


@pytest.fixture
def config(tmp_path: Path) -> Path:
    cfg = {
        "defaults": {
            "model": "yolo26n-seg.pt",
            "data": {"train": "train.txt", "val": "val.txt"},
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "flipud": 0.0,
        },
        "profiles": {
            "sanity": {"epochs": 5, "notes": "check"},
            "teacher": {"data": {"train": "train_real_road.txt", "val": "val_real_road.txt"}},
            "bad_flip": {"flipud": 0.5},
            "bad_test": {"data": {"val": "test.txt"}},
        },
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_load_profile_merges_defaults_and_nested_data(config: Path) -> None:
    sanity = load_profile(config, "sanity")
    assert sanity["epochs"] == 5 and sanity["imgsz"] == 640
    teacher = load_profile(config, "teacher")
    assert teacher["data"] == {"train": "train_real_road.txt", "val": "val_real_road.txt"}
    with pytest.raises(ValueError, match="Unknown profile"):
        load_profile(config, "nope")


def test_train_args_drop_our_own_keys(config: Path) -> None:
    args = train_args(load_profile(config, "sanity"))
    assert "model" not in args and "data" not in args and "notes" not in args
    assert args["epochs"] == 5


@pytest.mark.parametrize(("profile", "match"), [("bad_flip", "flipud"), ("bad_test", "test")])
def test_check_profile_rejects_rule_breaks(
    config: Path, data_root: Path, profile: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        check_profile(load_profile(config, profile), data_root)


def test_check_profile_needs_existing_lists(config: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_profile(load_profile(config, "sanity"), tmp_path)


def test_write_data_yaml_points_at_lists(data_root: Path) -> None:
    path = write_data_yaml(data_root, {"train": "train_real_road.txt", "val": "val.txt"}, "t")
    data = yaml.safe_load(path.read_text())
    assert data["path"] == str(data_root.resolve())
    assert data["train"] == "train_real_road.txt"
    assert data["names"] == NAMES


def test_class_metrics_finds_class_or_none() -> None:
    seg = _seg([1], [0.6], [0.35])
    assert class_metrics(seg, NAMES, "pothole") == {"map50": 0.6, "map50_95": 0.35}
    assert class_metrics(seg, NAMES, "road") is None


def test_run_trains_validates_and_writes_summary(config: Path, data_root: Path, tmp_path: Path):
    FakeYOLO.calls = []
    out = tmp_path / "runs"
    summary = run("sanity", data_root, out, config, yolo_cls=FakeYOLO)

    kind, weights, kwargs = FakeYOLO.calls[0]
    assert (kind, weights) == ("train", "yolo26n-seg.pt")
    assert kwargs["epochs"] == 5 and kwargs["project"] == str(out) and kwargs["name"] == "sanity"
    assert yaml.safe_load(Path(kwargs["data"]).read_text())["val"] == "val.txt"
    assert [c[1] for c in FakeYOLO.calls[1:]] == [str(out / "sanity/weights/best.pt")] * 2

    # Pothole from the mixed val list, road only from real-road val images.
    assert summary["metrics"]["pothole"] == {"map50": 0.6, "map50_95": 0.35}
    assert summary["metrics"]["road"] == {"map50": 0.9, "map50_95": 0.7}
    saved = json.loads((out / "sanity" / "summary.json").read_text())
    assert saved["config"]["epochs"] == 5

    row = results_row(summary)
    assert row.startswith(f"| {summary['date']} | sanity | yolo26n-seg | train.txt | 5 | 640 |")
    assert "0.600 | 0.350 | 0.900 | 0.700" in row


def test_run_teacher_validates_once(config: Path, data_root: Path, tmp_path: Path) -> None:
    FakeYOLO.calls = []
    run("teacher", data_root, tmp_path / "runs", config, yolo_cls=FakeYOLO)
    assert [c[0] for c in FakeYOLO.calls] == ["train", "val"]


def test_resume_continues_from_last(config: Path, data_root: Path, tmp_path: Path) -> None:
    FakeYOLO.calls = []
    out = tmp_path / "runs"
    run("sanity", data_root, out, config, resume=True, yolo_cls=FakeYOLO)
    assert FakeYOLO.calls[0] == ("train", str(out / "sanity/weights/last.pt"), {"resume": True})


def test_repo_train_config_is_valid() -> None:
    for profile in ("sanity", "teacher", "baseline", "small", "tune1"):
        cfg = load_profile(REPO_ROOT / "configs" / "train.yaml", profile)
        assert cfg["flipud"] == 0
        assert all("test" not in v for v in cfg["data"].values())
        # RAM caching can break deterministic runs; decode-once disk cache is the choice.
        assert cfg["cache"] == "disk"
