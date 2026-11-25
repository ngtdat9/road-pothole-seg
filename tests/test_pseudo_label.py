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
    _raster,
    analyse,
    build,
    choose_review,
    fetch,
    fill_road,
    load_settings,
    load_targets,
    pack,
    predict,
    review,
    ring_cover,
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
        offroad_max_cover=0.5,
        offroad_min_area=0.002,
        fill_close_frac=0.01,
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
    """Road on the lower half, drawn AROUND the pothole like the real teacher (a notch
    from the bottom edge up to the pothole's top). gn_d: no road at all."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def predict(self, source: str, **kwargs: object) -> list[SimpleNamespace]:
        stem = Path(source).stem
        self.seen.append(stem)
        if stem == "gn_d":
            return [SimpleNamespace(masks=None, boxes=None)]
        mask = np.zeros((H, W), np.float32)
        mask[H // 2 :] = 1
        mask[round(0.7 * H) :, round(0.4 * W) : round(0.6 * W)] = 0  # pothole left out
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


def _square(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def test_ring_cover_sees_road_around_a_pothole_gap() -> None:
    # Road everywhere below 0.5 except a gap where the pothole is: the pothole's own
    # area is not road, but the ring around it is.
    road = _raster([_square(0, 0.5, 1, 1)]) & ~_raster([_square(0.4, 0.7, 0.6, 0.9)])
    assert ring_cover(road, _square(0.4, 0.7, 0.6, 0.9)) > 0.9
    assert ring_cover(road, _square(0.1, 0.1, 0.3, 0.3)) == 0.0  # up in the scene


def test_fill_road_covers_kept_potholes(tmp_path: Path, base: Path) -> None:
    s = _settings(tmp_path, base)
    road = [np.array([[0, 0.5], [1, 0.5], [1, 1], [0.6, 1], [0.6, 0.7], [0.4, 0.7],
                      [0.4, 1], [0, 1]])]  # fmt: skip
    filled = fill_road(road, [_square(0.4, 0.7, 0.6, 0.9)], (100, 100), s)
    assert _raster(filled)[round(0.8 * 512), round(0.5 * 512)]  # pothole centre is road
    assert fill_road([], [_square(0.4, 0.7, 0.6, 0.9)], (100, 100), s) == []


def test_fetch_rejects_incomplete_zip(tmp_path: Path, base: Path, local: Path) -> None:
    drive = tmp_path / "drive"
    rows = (drive / "pseudo_road.csv").read_text()
    (drive / "pseudo_road.csv").write_text(rows + "gn_x,gn_x.jpg,train,1,0.5,0.9\n")
    with pytest.raises(SystemExit, match="labels"):
        fetch(drive, tmp_path / "again")


def test_analyse_flags_and_offroad(tmp_path: Path, base: Path, local: Path) -> None:
    rows, holes = analyse(local, _settings(tmp_path, base))
    flags = {r["name"]: r["flagged"] for r in rows}
    # gn_b: a vehicle labelled as pothole is off-road; gn_c: fine; gn_d: no road.
    assert flags == {"gn_b": 1, "gn_c": 0, "gn_d": 1}
    real, vehicle = holes["gn_b"]
    assert real.cover > 0.5 and vehicle.cover == 0.0


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
    road = np.array(lines[1].split()[1:], float).reshape(-1, 2)
    assert _raster([road])[round(0.8 * 512), round(0.5 * 512)]  # pothole filled into road
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
    assert 0 < s.offroad_max_cover < 1 and s.fill_close_frac < 0.05
