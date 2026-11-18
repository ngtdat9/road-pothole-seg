from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from training.scripts.inspect_dataset import (
    BOX_COLOUR,
    draw_labels,
    make_contact_sheets,
    parse_label_file,
    polygon_area,
    source_stem,
    summarize,
    to_markdown,
)

ROAD = "0 0 0.5 1 0.5 1 1 0 1"  # bottom half: area 0.5
BIG_POTHOLE = "1 0.4 0.6 0.6 0.6 0.6 0.8 0.4 0.8"  # 0.2 x 0.2 = 4%
TINY_POTHOLE = "1 0.1 0.9 0.12 0.9 0.12 0.92 0.1 0.92"  # 0.02 x 0.02 = 0.04%
BOX_ONLY = "1 0.5 0.5 0.1 0.1"


def _make_dataset(root: Path, files: dict[str, dict[str, str | None]]) -> Path:
    """files: split -> {image filename: label text or None for no label file}."""
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["road", "pothole"], "nc": 2}))
    for split, images in files.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for name, label in images.items():
            cv2.imwrite(str(root / split / "images" / name), np.zeros((64, 96, 3), np.uint8))
            if label is not None:
                (root / split / "labels" / f"{Path(name).stem}.txt").write_text(label)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return _make_dataset(
        tmp_path / "potholes-v1",
        {
            "train": {
                # Two Roboflow augmentation copies of one original.
                "a_jpg.rf.0123abcd.jpg": f"{ROAD}\n{BIG_POTHOLE}",
                "a_jpg.rf.89abcdef.jpg": f"{ROAD}\n{BIG_POTHOLE}",
                "b_jpg.rf.aaaa1111.jpg": TINY_POTHOLE,
                "c_jpg.rf.bbbb2222.jpg": None,  # no label file
            },
            "valid": {
                "a_jpg.rf.cccc3333.jpg": ROAD,  # same original as train -> leakage
                "d_jpg.rf.dddd4444.jpg": BOX_ONLY,
            },
        },
    )


def test_source_stem_strips_roboflow_suffix() -> None:
    assert source_stem(Path("frame_012_jpg.rf.5f3e9a1b2c.jpg")) == "frame_012_jpg"
    assert source_stem(Path("plain.jpg")) == "plain"


def test_polygon_area_is_image_fraction() -> None:
    square = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    assert polygon_area(square) == pytest.approx(0.25)
    assert polygon_area(np.empty((0, 2))) == 0.0


def test_parse_label_file_handles_polygons_and_boxes(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text(f"{BIG_POTHOLE}\n{BOX_ONLY}\n\n")
    poly, box = parse_label_file(path)
    assert poly.class_id == 1 and poly.polygon.shape == (4, 2) and not poly.box_only
    assert box.box_only and box.box == pytest.approx((0.5, 0.5, 0.1, 0.1))
    assert parse_label_file(tmp_path / "missing.txt") == []


def test_parse_label_file_rejects_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("1 0.1 0.2 0.3")
    with pytest.raises(ValueError):
        parse_label_file(path)


def test_summarize_counts(dataset: Path) -> None:
    s = summarize(dataset)
    assert s.images_per_split == {"train": 4, "valid": 2}
    assert s.originals_per_split == {"train": 3, "valid": 2}
    assert s.originals_in_several_splits == 1  # "a_jpg" is in train and valid
    assert s.annotations_per_class == {"road": 3, "pothole": 4}
    assert s.images_per_class == {"road": 3, "pothole": 4}
    assert s.box_only_labels == 1
    assert s.images_without_labels == 1
    assert s.size_buckets_per_class["pothole"]["tiny <0.1%"] == 1
    assert s.size_buckets_per_class["pothole"]["medium 1-5%"] == 2
    assert s.median_area_frac_per_class["road"] == pytest.approx(0.5)


def test_markdown_mentions_key_numbers(dataset: Path) -> None:
    md = to_markdown(summarize(dataset))
    assert "Original images (Roboflow `.rf.` copies removed): 5" in md
    assert "| `pothole` | 4 |" in md


def test_draw_labels_shows_box_only_labels(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text(BOX_ONLY)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_labels(image, parse_label_file(path), ["road", "pothole"])
    assert tuple(out[45, 50]) == BOX_COLOUR  # top edge of the box at y = 0.45
    assert not image.any()


def test_contact_sheets_use_one_copy_per_original(dataset: Path, tmp_path: Path) -> None:
    sheets = make_contact_sheets(dataset, tmp_path / "out", samples=50, seed=0)
    assert len(sheets) == 1  # 4 originals fit on one 10-image sheet
    sheet = cv2.imread(str(sheets[0]))
    assert sheet is not None and sheet.shape[1] == 5 * 320
