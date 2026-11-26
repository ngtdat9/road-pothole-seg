import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from training.scripts.failures import (
    FailureSettings,
    analyse,
    load_settings,
    match,
    run,
    summarise,
    worst_pothole,
    worst_road,
)
from training.scripts.merge_datasets import format_label


def sq(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


ROAD = sq(0, 0.5, 1, 1)
HOLE_A = sq(0.4, 0.7, 0.6, 0.9)  # medium (4%)
BIG_B = sq(0.1, 0.5, 0.5, 0.9)  # large (16%)
TINY_B = sq(0.8, 0.1, 0.83, 0.13)  # tiny (0.09%)
HOLE_C = sq(0.2, 0.2, 0.4, 0.4)  # medium (4%)


def _preds(rows: list[tuple[int, float, np.ndarray]]) -> str:
    return "".join(
        f"{c} {conf:.4f} " + " ".join(f"{v:.5f}" for v in p.reshape(-1)) + "\n"
        for c, conf, p in rows
    )


def _settings(tmp_path: Path) -> FailureSettings:
    return FailureSettings(
        conf=0.25, iou=0.5, grid=640, n_pothole=20, n_road=8,
        local_dir=tmp_path / "preds", out_dir=tmp_path / "out", csv_dir=tmp_path / "docs",
        causes=["shadow", "other"],
    )  # fmt: skip


@pytest.fixture
def data(tmp_path: Path) -> tuple[Path, Path]:
    """a: real road, pothole found + a duplicate; b: big found, tiny missed, one false;
    c: pothole-only image, pothole missed."""
    root = tmp_path / "merged-v1"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"names": {0: "road", 1: "pothole"}}))
    labels = {
        "a": [(0, ROAD), (1, HOLE_A)],
        "b": [(0, ROAD), (1, BIG_B), (1, TINY_B)],
        "c": [(1, HOLE_C)],
    }
    for name, inst in labels.items():
        cv2.imwrite(str(root / "images" / f"{name}.jpg"), np.full((60, 80, 3), 90, np.uint8))
        (root / "labels" / f"{name}.txt").write_text(format_label(inst))
    (root / "val.txt").write_text("./images/a.jpg\n./images/b.jpg\n./images/c.jpg\n")
    (root / "val_real_road.txt").write_text("./images/a.jpg\n")

    preds = tmp_path / "preds" / "baseline_val_640"
    preds.mkdir(parents=True)
    (preds / "a.txt").write_text(_preds([
        (0, 0.9, ROAD), (1, 0.8, HOLE_A),
        (1, 0.6, HOLE_A),  # duplicate: the pothole is already taken -> false
        (1, 0.1, HOLE_C),  # below conf: ignored
    ]))  # fmt: skip
    (preds / "b.txt").write_text(
        _preds([(1, 0.7, sq(0.12, 0.52, 0.5, 0.9)), (1, 0.5, sq(0.6, 0.1, 0.7, 0.2))])
    )
    (preds / "c.txt").write_text("")
    return root, preds


def test_match_is_greedy_by_confidence() -> None:
    left, right = sq(0, 0, 0.5, 1), sq(0.5, 0, 1, 1)
    found, matched, ious = match([left, right], [sq(0, 0, 0.55, 1), left], 0.5, 200)
    assert found == [True, False] and matched == [True, False] and ious[0] > 0.85
    assert match([], [left], 0.5, 200) == ([], [False], [])


def test_analyse_and_summary(tmp_path: Path, data: tuple[Path, Path]) -> None:
    root, preds = data
    s = _settings(tmp_path)
    results = {r.name: r for r in analyse(root, preds, "val", s)}
    assert [(r.tp, r.fn, r.fp) for r in results.values()] == [(1, 0, 1), (1, 1, 1), (0, 1, 0)]
    assert results["a"].road_iou == pytest.approx(1.0)
    assert results["b"].road_iou is None  # only real-road images are scored for road

    summary = summarise(list(results.values()), s)
    p = summary["pothole"]
    assert (p["found"], p["missed"], p["false"]) == (2, 2, 2) and p["recall"] == 0.5
    by_size = p["recall_by_size"]
    assert by_size["tiny <0.1%"] == {"labels": 1, "found": 0, "recall": 0.0}
    assert by_size["medium 1-5%"]["recall"] == 0.5 and by_size["large >=5%"]["recall"] == 1.0
    assert by_size["small 0.1-1%"]["recall"] is None
    assert summary["road"]["real_road_images"] == 1

    assert [r.name for r in worst_pothole(list(results.values()), 5)] == ["b", "c", "a"]
    assert [r.name for r in worst_road(list(results.values()), 5)] == ["a"]


def test_run_writes_gallery_csv_and_summary(tmp_path: Path, data: tuple[Path, Path]) -> None:
    root, preds = data
    s = _settings(tmp_path)
    summary = run("baseline", root, preds, s, record_dir=tmp_path / "runs")
    with (s.csv_dir / "failures_baseline.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["kind"], r["name"]) for r in rows] == [
        ("pothole", "b"), ("pothole", "c"), ("pothole", "a"), ("road", "a"),
    ]  # fmt: skip
    assert all(r["cause"] == "" for r in rows)
    assert summary["gallery"]["sheets"] == ["pothole_01.jpg", "road_01.jpg"]
    assert (s.out_dir / "baseline" / "pothole_01.jpg").exists()
    assert len(list((s.out_dir / "baseline" / "images").glob("*.jpg"))) == 4
    saved = json.loads((tmp_path / "runs" / "baseline" / "failures_val.json").read_text())
    assert saved["pothole"]["missed"] == 2

    with pytest.raises(SystemExit, match="force"):  # never overwrite hand-filled causes
        run("baseline", root, preds, s, record_dir=tmp_path / "runs")
    with pytest.raises(SystemExit, match="test"):
        run("baseline", root, preds, s, split="test", force=True, record_dir=tmp_path / "runs")


def test_missing_predictions_file_is_an_error(tmp_path: Path, data: tuple[Path, Path]) -> None:
    root, preds = data
    (preds / "c.txt").unlink()
    with pytest.raises(SystemExit, match="c"):
        analyse(root, preds, "val", _settings(tmp_path))


def test_repo_config_uses_pipeline_conf() -> None:
    s = load_settings()
    assert s.conf == 0.25 and 0 < s.iou < 1
    assert "label_error" in s.causes and s.n_pothole == 20
