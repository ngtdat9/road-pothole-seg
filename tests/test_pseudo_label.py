import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from training.scripts.pseudo_label import (
    REPO_ROOT,
    PseudoSettings,
    analyse,
    build,
    choose_review,
    fetch,
    load_settings,
    load_targets,
    pack,
    pothole_cover,
    predict,
    review,
    road_polygons,
)

H = W = 64
# A pothole in the lower half (on the road) ...
POTHOLE = "1 0.4 0.7 0.6 0.7 0.6 0.9 0.4 0.9\n"
# ... and a large "pothole" up in the scene: how GNSoft2 sometimes labels a vehicle.
VEHICLE = "1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n"


def _settings(tmp_path: Path, base: Path) -> PseudoSettings:
    return PseudoSettings(
        base_dir=base,
        out_dir=tmp_path / "merged-v1",
        local_dir=tmp_path / "local",
        review_csv=tmp_path / "docs" / "pseudo_review.csv",
        offroad_csv=tmp_path / "docs" / "pseudo_offroad.csv",
        imgsz=640,
        conf=0.5,
        min_area_frac=0.01,
        min_pothole_cover=0.5,
        flag_min_area=0.002,
        offroad_max_cover=0.2,
        offroad_min_area=0.01,
        review_n=3,
        seed=0,
    )


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """rs_a real road (train); gn_b, gn_c, gn_d pothole-only; gn_t in test."""
    root = tmp_path / "merged-v1-base"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    rows = [
        ("rs_a", "train", 1, "0 0 0.5 1 0.5 1 1 0 1\n"),
        ("gn_b", "train", 0, POTHOLE + VEHICLE),
        ("gn_c", "train", 0, POTHOLE),
        ("gn_d", "val", 0, POTHOLE),
        ("gn_t", "test", 0, POTHOLE),
    ]
    lists: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    with (root / "split.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "split", "source", "group", "real_road"])
        for name, split, real, label in rows:
            cv2.imwrite(str(root / "images" / f"{name}.jpg"), np.zeros((H, W, 3), np.uint8))
            (root / "labels" / f"{name}.txt").write_text(label)
            writer.writerow([name, split, "src", name, real])
            lists[split].append(f"./images/{name}.jpg")
    for split, names in lists.items():
        (root / f"{split}.txt").write_text("\n".join(names) + "\n")
    (root / "val_real_road.txt").write_text("\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"names": {0: "road", 1: "pothole"}}))
    return root


class FakeTeacher:
    """gn_b: road on the lower half. gn_c: road only from y = 0.82 (cuts the pothole).
    gn_d: no road at all."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def predict(self, source: str, **kwargs: object) -> list[SimpleNamespace]:
        stem = Path(source).stem
        self.seen.append(stem)
        if stem == "gn_d":
            return [SimpleNamespace(masks=None, boxes=None)]
        mask = np.zeros((H, W), np.float32)
        mask[round(H * (0.82 if stem == "gn_c" else 0.5)) :] = 1
        masks = SimpleNamespace(data=mask[None])
        boxes = SimpleNamespace(cls=np.array([0.0]), conf=np.array([0.9]))
        return [SimpleNamespace(masks=masks, boxes=boxes)]


@pytest.fixture
def local(tmp_path: Path, base: Path) -> Path:
    """Predict on 'Colab', pack to 'Drive', fetch to the local copy."""
    s = _settings(tmp_path, base)
    work = tmp_path / "work"
    predict(FakeTeacher(), base, work, s)
    pack(work, tmp_path / "drive")
    assert fetch(tmp_path / "drive", s.local_dir) == 3
    return s.local_dir


def test_targets_are_pothole_only_train_and_val(base: Path) -> None:
    assert [(t.name, t.split) for t in load_targets(base)] == [
        ("gn_b", "train"),
        ("gn_c", "train"),
        ("gn_d", "val"),
    ]


def test_predict_never_reads_test_or_real_road(tmp_path: Path, base: Path) -> None:
    teacher = FakeTeacher()
    predict(teacher, base, tmp_path / "work", _settings(tmp_path, base))
    assert teacher.seen == ["gn_b", "gn_c", "gn_d"]
    lines = (tmp_path / "work" / "labels" / "gn_b.txt").read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("0 ")


def test_road_polygons_filter_class_conf_and_area(tmp_path: Path, base: Path) -> None:
    s = _settings(tmp_path, base)
    masks = np.zeros((3, H, W), np.float32)
    masks[0, H // 2 :] = 1  # road, confident
    masks[1, :] = 1  # pothole class: ignored
    masks[2, 0:2, 0:2] = 1  # road, but tiny
    polys = road_polygons(masks, np.array([0, 1, 0]), np.array([0.9, 0.9, 0.9]), 0, s)
    assert len(polys) == 1 and polys[0][1] == pytest.approx(0.9)
    assert road_polygons(masks[:1], np.array([0]), np.array([0.3]), 0, s) == []


def test_pothole_cover() -> None:
    lower = [np.array([[0, 0.5], [1, 0.5], [1, 1], [0, 1]])]
    hole = [np.array([[0.4, 0.7], [0.6, 0.7], [0.6, 0.9], [0.4, 0.9]])]
    assert pothole_cover(lower, hole) == pytest.approx(1.0)
    assert pothole_cover([], hole) == 0.0
    assert pothole_cover(lower, []) is None


def test_fetch_rejects_incomplete_zip(tmp_path: Path, base: Path, local: Path) -> None:
    drive = tmp_path / "drive"
    rows = (drive / "pseudo_road.csv").read_text()
    (drive / "pseudo_road.csv").write_text(rows + "gn_x,gn_x.jpg,train,1,0.5,0.9\n")
    with pytest.raises(SystemExit, match="labels"):
        fetch(drive, tmp_path / "again")


def test_analyse_flags_and_offroad(tmp_path: Path, base: Path, local: Path) -> None:
    rows, holes = analyse(local, _settings(tmp_path, base))
    flags = {r["name"]: r["flagged"] for r in rows}
    # gn_b: the vehicle is off-road (not a flag); gn_c: road cuts a real pothole; gn_d: no road.
    assert flags == {"gn_b": 0, "gn_c": 1, "gn_d": 1}
    assert [round(h.cover, 1) for h in holes["gn_b"]] == [1.0, 0.0]


def test_choose_review_takes_all_flagged_then_samples() -> None:
    rows = [{"name": f"n{i}", "flagged": 1 if i < 2 else 0} for i in range(10)]
    chosen = choose_review(rows, 5, seed=0)
    assert [r["name"] for r in chosen[:2]] == ["n0", "n1"] and len(chosen) == 5
    assert len(choose_review(rows, 1, seed=0)) == 2  # flagged are never cut


def test_review_then_build(tmp_path: Path, base: Path, local: Path) -> None:
    s = _settings(tmp_path, base)
    summary = review(local, s, tmp_path / "sheets")
    assert summary["offroad_labels"] == 1 and summary["flagged"] == 2
    assert (tmp_path / "sheets" / "review_01.jpg").exists()
    assert (tmp_path / "sheets" / "offroad_01.jpg").exists()
    with pytest.raises(SystemExit):  # never overwrite decisions silently
        review(local, s, tmp_path / "sheets")

    # Reject gn_d (no road).
    text = s.review_csv.read_text().splitlines()
    text = [line[:-1] + "0" if ",gn_d," in line else line for line in text]
    s.review_csv.write_text("\n".join(text) + "\n")
    result = build(local, s)

    out = s.out_dir
    assert result["rejected"] == ["gn_d"] and result["offroad_pothole_labels_dropped"] == 1
    lines = (out / "labels" / "gn_b.txt").read_text().splitlines()
    assert lines[0] == POTHOLE.strip() and len(lines) == 2 and lines[1].startswith("0 ")
    assert "gn_d" not in (out / "val.txt").read_text()
    assert not (out / "images" / "gn_d.jpg").exists()
    assert (out / "labels" / "gn_t.txt").read_text() == POTHOLE  # test untouched
    assert (out / "labels" / "rs_a.txt").read_text() == (base / "labels" / "rs_a.txt").read_text()
    assert json.loads((out / "report.json").read_text())["pseudo_road"]["images"]["val"] == 0


def test_build_needs_review(tmp_path: Path, base: Path, local: Path) -> None:
    with pytest.raises(SystemExit, match="review"):
        build(local, _settings(tmp_path, base))


def test_repo_config_loads() -> None:
    s = load_settings(REPO_ROOT / "configs" / "merge.yaml")
    assert s.out_dir.name == "merged-v1" and 0 < s.conf < 1
    assert s.offroad_max_cover < s.min_pothole_cover
