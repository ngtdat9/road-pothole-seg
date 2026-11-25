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
    build,
    choose_review,
    load_settings,
    load_targets,
    pothole_cover,
    predict,
    review,
    road_polygons,
)

H = W = 64
# Pothole in the lower half; the fake teacher predicts road on the lower half.
POTHOLE = "1 0.4 0.7 0.6 0.7 0.6 0.9 0.4 0.9\n"


def _settings(tmp_path: Path, base: Path) -> PseudoSettings:
    return PseudoSettings(
        base_dir=base,
        out_dir=tmp_path / "merged-v1",
        review_csv=tmp_path / "docs" / "pseudo_review.csv",
        imgsz=640,
        conf=0.5,
        min_area_frac=0.01,
        min_pothole_cover=0.5,
        review_n=3,
        seed=0,
    )


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """Tiny split: rs_a real road (train), gn_b/gn_c/gn_d pothole-only, gn_t in test."""
    root = tmp_path / "merged-v1-base"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    rows = [
        ("rs_a", "train", 1, "0 0 0.5 1 0.5 1 1 0 1\n"),
        ("gn_b", "train", 0, POTHOLE),
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
    """Road on the lower half; for gn_c only the top quarter (misses the pothole)."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def predict(self, source: str, **kwargs: object) -> list[SimpleNamespace]:
        self.seen.append(Path(source).stem)
        mask = np.zeros((H, W), np.float32)
        if Path(source).stem == "gn_c":
            mask[: H // 4] = 1
        else:
            mask[H // 2 :] = 1
        masks = SimpleNamespace(data=mask[None])
        boxes = SimpleNamespace(cls=np.array([0.0]), conf=np.array([0.9]))
        return [SimpleNamespace(masks=masks, boxes=boxes)]


def test_targets_are_pothole_only_train_and_val(base: Path) -> None:
    targets = load_targets(base)
    assert [(t.name, t.split) for t in targets] == [
        ("gn_b", "train"),
        ("gn_c", "train"),
        ("gn_d", "val"),
    ]


def test_road_polygons_filter_class_conf_and_area(tmp_path: Path, base: Path) -> None:
    s = _settings(tmp_path, base)
    masks = np.zeros((3, H, W), np.float32)
    masks[0, H // 2 :] = 1  # road, confident
    masks[1, :] = 1  # pothole class: ignored
    masks[2, 0:2, 0:2] = 1  # road, but tiny
    polys = road_polygons(masks, np.array([0, 1, 0]), np.array([0.9, 0.9, 0.9]), 0, s)
    assert len(polys) == 1
    assert polys[0][1] == pytest.approx(0.9)
    assert road_polygons(masks[:1], np.array([0]), np.array([0.3]), 0, s) == []


def test_pothole_cover() -> None:
    lower = [np.array([[0, 0.5], [1, 0.5], [1, 1], [0, 1]])]
    hole = [np.array([[0.4, 0.7], [0.6, 0.7], [0.6, 0.9], [0.4, 0.9]])]
    assert pothole_cover(lower, hole) == pytest.approx(1.0)
    assert pothole_cover([], hole) == 0.0
    assert pothole_cover(lower, []) is None


def test_predict_writes_labels_and_flags_misses(tmp_path: Path, base: Path) -> None:
    teacher = FakeTeacher()
    out = tmp_path / "pseudo"
    rows = predict(teacher, base, out, _settings(tmp_path, base))
    assert "gn_t" not in teacher.seen and "rs_a" not in teacher.seen
    by_name = {r["name"]: r for r in rows}
    assert by_name["gn_b"]["flagged"] == 0 and by_name["gn_c"]["flagged"] == 1
    label = (out / "labels" / "gn_b.txt").read_text().splitlines()
    assert len(label) == 1 and label[0].startswith("0 ")
    assert (out / "pseudo_road.csv").exists()


def test_choose_review_takes_all_flagged_then_samples() -> None:
    rows = [{"name": f"n{i}", "flagged": "1" if i < 2 else "0"} for i in range(10)]
    chosen = choose_review(rows, 5, seed=0)
    assert [r["name"] for r in chosen[:2]] == ["n0", "n1"] and len(chosen) == 5
    assert len(choose_review(rows, 1, seed=0)) == 2  # flagged are never cut


def test_review_and_build(tmp_path: Path, base: Path) -> None:
    s = _settings(tmp_path, base)
    pseudo = tmp_path / "pseudo"
    predict(FakeTeacher(), base, pseudo, s)
    records = review(pseudo, s, tmp_path / "sheets")
    assert len(records) == 3 and (tmp_path / "sheets" / "sheet_01.jpg").exists()
    with pytest.raises(SystemExit):  # never overwrite decisions silently
        review(pseudo, s, tmp_path / "sheets")

    # Reject gn_c (road missed the pothole).
    text = s.review_csv.read_text().splitlines()
    text = [line[:-1] + "0" if ",gn_c," in line else line for line in text]
    s.review_csv.write_text("\n".join(text) + "\n")
    summary = build(pseudo, s)

    out = s.out_dir
    assert summary["pseudo_labelled"] == 2 and summary["rejected"] == ["gn_c"]
    lines = (out / "labels" / "gn_b.txt").read_text().splitlines()
    assert lines[0] == POTHOLE.strip() and lines[1].startswith("0 ")  # pothole kept, road added
    assert "gn_c" not in (out / "train.txt").read_text()
    assert not (out / "images" / "gn_c.jpg").exists()
    assert (out / "labels" / "gn_t.txt").read_text() == POTHOLE  # test untouched
    assert (out / "labels" / "rs_a.txt").read_text() == (base / "labels" / "rs_a.txt").read_text()
    assert json.loads((out / "report.json").read_text())["pseudo_road"]["images"]["train"] == 2


def test_build_needs_review(tmp_path: Path, base: Path) -> None:
    s = _settings(tmp_path, base)
    predict(FakeTeacher(), base, tmp_path / "pseudo", s)
    with pytest.raises(SystemExit, match="review"):
        build(tmp_path / "pseudo", s)


def test_repo_config_loads() -> None:
    s = load_settings(REPO_ROOT / "configs" / "merge.yaml")
    assert s.out_dir.name == "merged-v1" and 0 < s.conf < 1
