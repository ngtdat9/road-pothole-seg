import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from training.scripts.inspect_dataset import Instance
from training.scripts.merge_datasets import (
    REPO_ROOT,
    box_iou,
    choose_copy,
    group_of,
    load_config,
    merge,
    split_box_only,
)

# Raw class ids follow each fake dataset's data.yaml (see _make_dataset calls).
ROAD = "1 0 0.5 1 0.5 1 1 0 1"  # raw id 1 = road in the two-class set; area 0.5
POTHOLE = "0 0.4 0.6 0.6 0.6 0.6 0.8 0.4 0.8"  # raw id 0 = pothole; bbox (0.4,0.6)-(0.6,0.8)
POTHOLE_DUP_BOX = "0 0.5 0.7 0.2 0.2"  # same pothole as a box -> drop the line
ORPHAN_BOX = "0 0.1 0.1 0.05 0.05"  # pothole with no mask -> drop the image


def _make_dataset(root: Path, names: list[str], files: dict[str, dict[str, str]]) -> None:
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}))
    for split, images in files.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for name, label in images.items():
            cv2.imwrite(str(root / split / "images" / name), np.zeros((32, 32, 3), np.uint8))
            (root / split / "labels" / f"{Path(name).stem}.txt").write_text(label)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    _make_dataset(
        raw / "two-class-v1",
        ["0", "road"],
        {
            "train": {
                # Two augmentation copies; the second is zoomed in (bigger road area).
                "vid-1_jpg.rf.aaaa.jpg": f"{ROAD}\n{POTHOLE}\n{POTHOLE_DUP_BOX}",
                "vid-1_jpg.rf.bbbb.jpg": f"{ROAD}\n0 0.3 0.5 0.7 0.5 0.7 0.9 0.3 0.9\n"
                "1 0 0.2 1 0.2 1 1 0 1",
                "vid-2_jpg.rf.cccc.jpg": f"{ROAD}\n{POTHOLE}",
                "orphan_jpg.rf.dddd.jpg": f"{ROAD}\n{POTHOLE}\n{ORPHAN_BOX}",
                "noroad_jpg.rf.eeee.jpg": POTHOLE,
                "DJI_0001_jpg.rf.ffff.jpg": f"{ROAD}\n{POTHOLE}",
            },
            "valid": {"single_jpg.rf.1111.jpg": ROAD},
        },
    )
    _make_dataset(
        raw / "pothole-only-v2",
        ["manhole", "pothole", "vehicle"],
        {
            "train": {
                # raw ids: 0 manhole, 1 pothole, 2 vehicle (a vehicle box is harmless).
                "AM_x_1_jpg.rf.aaaa.jpg": "0 0.1 0.1 0.2 0.1 0.2 0.2\n"
                "1 0.4 0.6 0.6 0.6 0.6 0.8 0.4 0.8\n2 0.5 0.5 0.1 0.1",
                "AM_x_2_jpg.rf.bbbb.jpg": "0 0.1 0.1 0.2 0.1 0.2 0.2",  # manhole only
            }
        },
    )
    cfg = {
        "raw_root": str(raw),
        "out_dir": str(tmp_path / "curated"),
        "classes": ["road", "pothole"],
        "box_dup_iou": 0.5,
        "sources": [
            {
                "slug": "two-class-v1",
                "prefix": "tc",
                "real_road": True,
                "class_map": {"0": "pothole", "road": "road"},
                "exclude": ["^DJI_"],
                "groups": [r"^(vid)-\d+"],
            },
            {
                "slug": "pothole-only-v2",
                "prefix": "po",
                "real_road": False,
                "class_map": {"pothole": "pothole", "manhole": None, "vehicle": None},
                "exclude": [],
                "groups": [r"^(AM_x)_\d+"],
            },
        ],
    }
    path = tmp_path / "merge.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _manifest(out_dir: Path) -> dict[str, dict[str, str]]:
    with (out_dir / "manifest.csv").open() as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def test_box_iou() -> None:
    assert box_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert box_iou((0, 0, 2, 1), (1, 0, 3, 1)) == pytest.approx(1 / 3)


def test_split_box_only_separates_duplicates_from_orphans() -> None:
    poly = np.array([[0.4, 0.6], [0.6, 0.6], [0.6, 0.8], [0.4, 0.8]])
    instances = [
        Instance(1, poly, box_only=False),
        Instance(1, np.empty((0, 2)), True, (0.5, 0.7, 0.2, 0.2)),  # duplicate
        Instance(1, np.empty((0, 2)), True, (0.1, 0.1, 0.05, 0.05)),  # orphan
        Instance(0, np.empty((0, 2)), True, (0.5, 0.7, 0.2, 0.2)),  # other class -> orphan
    ]
    assert split_box_only(instances, 0.5) == (1, 2)


def test_group_of_uses_first_match_else_image() -> None:
    assert group_of("vid-12_jpg", [r"^(vid)-\d+"], "tc") == "tc:vid"
    assert group_of("lonely_jpg", [r"^(vid)-\d+"], "tc") == "tc:lonely_jpg"


def test_choose_copy_prefers_unaugmented_split_then_smallest_area() -> None:
    small = [Instance(0, np.array([[0, 0], [0.1, 0], [0.1, 0.1]]), False)]
    big = [Instance(0, np.array([[0, 0], [0.9, 0], [0.9, 0.9]]), False)]
    train_small = ("train", Path("a.rf.1.jpg"), small)
    train_big = ("train", Path("a.rf.2.jpg"), big)
    valid_big = ("valid", Path("a.rf.3.jpg"), big)
    assert choose_copy([train_big, train_small]) is train_small
    assert choose_copy([train_small, valid_big]) is valid_big


def test_merge_applies_every_rule(config_path: Path) -> None:
    cfg = load_config(config_path)
    report = merge(cfg)
    rows = _manifest(cfg.out_dir)

    two = report["sources"]["two-class-v1"]
    assert two["originals"] == 6
    assert two["excluded"] == 1  # DJI
    assert two["dropped_box_only_image"] == 1  # orphan
    assert two["dropped_no_road"] == 1  # noroad
    assert two["box_lines_duplicate"] == 1
    assert two["kept"] == 3
    assert sorted(rows) == ["po_AM_x_1_jpg", "po_AM_x_2_jpg", "tc_single_jpg", "tc_vid-1_jpg",
                            "tc_vid-2_jpg"]  # fmt: skip

    only = report["sources"]["pothole-only-v2"]
    assert only["labels_deleted"] == {"manhole": 2, "vehicle": 1}
    assert only["dropped_box_only_image"] == 0  # the vehicle box didn't cost the image
    assert report["total_kept"] == 5
    assert report["total_real_road"] == 3

    # Groups: both vid frames share one group; unmatched images are their own.
    assert rows["tc_vid-1_jpg"]["group"] == rows["tc_vid-2_jpg"]["group"] == "tc:vid"
    assert rows["tc_single_jpg"]["group"] == "tc:single_jpg"
    assert rows["po_AM_x_1_jpg"]["real_road"] == "0"


def test_merge_writes_remapped_polygon_labels(config_path: Path) -> None:
    cfg = load_config(config_path)
    merge(cfg)
    # The less zoomed-in copy (rf.aaaa) was chosen; its box line is gone; road -> 0.
    label = (cfg.out_dir / "labels" / "tc_vid-1_jpg.txt").read_text().splitlines()
    assert [line.split()[0] for line in label] == ["0", "1"]
    assert all(len(line.split()) >= 7 for line in label)
    assert label[0].split()[1:3] == ["0.000000", "0.500000"]
    # Pothole-only: manhole deleted, pothole remapped from raw id 1 to output id 1.
    po = (cfg.out_dir / "labels" / "po_AM_x_1_jpg.txt").read_text().splitlines()
    assert [line.split()[0] for line in po] == ["1"]
    assert (cfg.out_dir / "labels" / "po_AM_x_2_jpg.txt").read_text() == ""  # hard negative
    assert (cfg.out_dir / "images" / "tc_vid-1_jpg.jpg").exists()
    assert json.loads((cfg.out_dir / "report.json").read_text())["total_kept"] == 5


def test_load_config_rejects_unknown_target_class(tmp_path: Path, config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    cfg["sources"][0]["class_map"]["road"] = "lane"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="unknown classes"):
        load_config(bad)


def test_repo_merge_config_loads() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "merge.yaml")
    assert cfg.classes == ["road", "pothole"]
    assert [s.prefix for s in cfg.sources] == ["pr", "rs", "gn"]
