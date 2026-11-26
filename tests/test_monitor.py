import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.scripts.monitor import (
    EPOCHS_CSV,
    Monitor,
    line_chart,
    load_reference,
    read_epochs,
    render,
    summarise,
)

NAMES = {0: "road", 1: "pothole"}
LOSSES = ("box_loss", "seg_loss", "cls_loss", "sem_loss")


class FakeTrainer:
    """The attributes of an Ultralytics SegmentationTrainer that the monitor reads."""

    def __init__(self, save_dir: Path, epochs: int = 10, start_epoch: int = 0) -> None:
        self.save_dir = save_dir
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.epoch = start_epoch
        self.args = SimpleNamespace(model="yolo26n-seg.pt", data="/c/data_baseline.yaml",
                                    imgsz=640, patience=3, close_mosaic=2)  # fmt: skip
        self.train_loader = SimpleNamespace(dataset=list(range(200)))
        self.data = {"names": NAMES}
        self.lr = {"lr/pg0": 0.001}
        self.tloss = None
        self.metrics: dict[str, float] = {}
        self.validator = None

    def label_loss_items(self, loss: list[float], prefix: str) -> dict[str, float]:
        return {f"{prefix}/{n}": v for n, v in zip(LOSSES, loss, strict=True)}

    def finish_epoch(self, pothole: float, road: float) -> None:
        self.tloss = [1.0, 2.0 - self.epoch * 0.1, 3.0, 0.5]
        self.metrics = {"metrics/mAP50-95(M)": (pothole + road) / 2, "val/seg_loss": 2.2,
                        "val/box_loss": 1.4, "val/sem_loss": 0.0}  # fmt: skip
        seg = SimpleNamespace(ap_class_index=[0, 1], ap50=[0.9, 0.7], ap=[road, pothole])
        self.validator = SimpleNamespace(metrics=SimpleNamespace(seg=seg))


def _train(m: Monitor, trainer: FakeTrainer, scores: list[float]) -> None:
    for pothole in scores:
        m.on_train_epoch_start(trainer)
        m.on_train_epoch_end(trainer)
        trainer.finish_epoch(pothole, 0.7)
        m.on_fit_epoch_end(trainer)
        trainer.epoch += 1


def test_monitor_records_epochs_and_redraws(tmp_path: Path) -> None:
    shown: list[str] = []
    m = Monitor(show=shown.append, reference={"name": "baseline", "pothole": 0.40, "road": 0.79})
    trainer = FakeTrainer(tmp_path / "baseline")
    m.on_train_start(trainer)
    _train(m, trainer, [0.30, 0.42, 0.41, 0.40])
    m.on_train_end(trainer)

    assert len(shown) == 6  # start, 4 epochs, end
    rec = m.records[1]
    assert rec["epoch"] == 2 and rec["pothole_map50_95"] == 0.42 and rec["road_map50_95"] == 0.7
    assert rec["train_seg_loss"] == pytest.approx(1.9) and rec["val_seg_loss"] == 2.2
    assert rec["lr"] == 0.001

    last = shown[-1]
    assert "Epoch 4 of 10" in last and "stopped early" in last
    assert "best, epoch 2" in last and "baseline 0.400 (+0.020)" in last
    assert "early stop in 1" in last  # 2 epochs since best, patience 3
    assert "mosaic off from epoch 9" in last
    assert "Box loss" not in last and "box loss" in last  # seg loss is the big chart

    saved = read_epochs(tmp_path / "baseline" / EPOCHS_CSV, up_to=99)
    assert [r["epoch"] for r in saved] == [1, 2, 3, 4]


def test_resume_reads_back_earlier_epochs(tmp_path: Path) -> None:
    first = Monitor()
    trainer = FakeTrainer(tmp_path / "run")
    first.on_train_start(trainer)
    _train(first, trainer, [0.3, 0.35, 0.36])  # epoch 3 was lost: resume starts after epoch 2

    again = Monitor()
    trainer = FakeTrainer(tmp_path / "run", start_epoch=2)
    again.on_train_start(trainer)
    assert [r["epoch"] for r in again.records] == [1, 2]
    _train(again, trainer, [0.37])
    assert [r["epoch"] for r in again.records] == [1, 2, 3]
    assert again.records[-1]["pothole_map50_95"] == 0.37


def test_summarise_eta_and_speed() -> None:
    records = [{"epoch": i, "train_s": 50.0, "val_s": 10.0, "pothole_map50_95": 0.1 * i}
               for i in range(1, 4)]  # fmt: skip
    s = summarise(records, {"epochs": 10, "train_images": 1000, "close_mosaic": 10})
    assert s["eta"] == 7 * 60 and s["elapsed"] == 180 and s["img_per_s"] == 20
    assert s["best"]["epoch"] == 3 and s["since_best"] == 0 and s["mosaic_off"] == 1


def test_render_before_any_epoch_and_escapes_names() -> None:
    info = {"name": "<b>x</b>", "epochs": 100, "patience": 30, "close_mosaic": 10}
    page = render([], info)
    assert "Epoch 0 of 100" in page and "Waiting for the first epoch" in page
    assert "<b>x</b>" not in page and "&lt;b&gt;" in page


def test_line_chart_skips_missing_values() -> None:
    svg = line_chart([("a", [None, 0.2, 0.3], "#000", False)], 10, mark_epoch=3, vline=9)
    assert svg.count("<path") == 1 and "best, epoch 3" in svg


def test_load_reference(tmp_path: Path) -> None:
    (tmp_path / "baseline").mkdir()
    metrics = {"pothole": {"map50_95": 0.408}, "road": {"map50_95": 0.79}}
    (tmp_path / "baseline" / "summary.json").write_text(json.dumps({"metrics": metrics}))
    assert load_reference("baseline", tmp_path) == {
        "name": "baseline",
        "pothole": 0.408,
        "road": 0.79,
    }
    assert load_reference("missing", tmp_path) is None
