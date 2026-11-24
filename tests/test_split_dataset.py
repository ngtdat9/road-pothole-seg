import csv
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from training.scripts.split_dataset import (
    SPLITS,
    Item,
    assign_groups,
    check_no_leakage,
    load_settings,
    split,
)

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def _items() -> list[Item]:
    """Two sources: 'a' with videos of 1-5 frames, 'b' with single images."""
    items = []
    for g in range(40):
        for k in range(1 + g % 5):
            items.append(Item(f"a_{g}_{k}", f"a_{g}_{k}.jpg", "a", f"a:{g}", True, 1))
    for g in range(60):
        items.append(Item(f"b_{g}", f"b_{g}.jpg", "b", f"b:{g}", False, 2))
    return items


def test_assign_groups_hits_ratios_per_source() -> None:
    items = _items()
    groups = assign_groups(items, RATIOS, seed=0)
    for source in ("a", "b"):
        src = [it for it in items if it.source == source]
        for s in SPLITS:
            share = sum(groups[it.group] == s for it in src) / len(src)
            assert share == pytest.approx(RATIOS[s], abs=0.04)


def test_assign_groups_is_deterministic_and_seed_dependent() -> None:
    items = _items()
    assert assign_groups(items, RATIOS, 0) == assign_groups(items, RATIOS, 0)
    assert assign_groups(items, RATIOS, 0) != assign_groups(items, RATIOS, 1)


def test_check_no_leakage_catches_split_group() -> None:
    items = [Item("x1", "x1.jpg", "a", "a:x", True, 0), Item("x2", "x2.jpg", "a", "a:x", True, 0)]
    check_no_leakage(items, {"x1": "train", "x2": "train"})
    with pytest.raises(AssertionError):
        check_no_leakage(items, {"x1": "train", "x2": "test"})


def test_load_settings_rejects_bad_ratios(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        yaml.safe_dump({"split": {"out_dir": "x", "ratios": {"train": 0.9}, "seed": 0}})
    )
    with pytest.raises(ValueError):
        load_settings(path)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    pool = tmp_path / "curated"
    (pool / "images").mkdir(parents=True)
    (pool / "labels").mkdir(parents=True)
    manifest = [["name", "file", "source", "group", "real_road", "n_pothole"]]
    dedup = [["name", "dhash", "removed", "duplicate_of", "group"]]
    for it in _items():
        (pool / "images" / it.file).write_bytes(b"jpg")
        (pool / "labels" / f"{it.name}.txt").write_text("1 0 0 1 0 1 1\n")
        manifest.append([it.name, it.file, it.source, "unused", int(it.real_road), it.n_pothole])
        dedup.append([it.name, "0", 0, "", it.group])
    # One removed duplicate that must not appear anywhere.
    (pool / "images" / "dup.jpg").write_bytes(b"jpg")
    manifest.append(["dup", "dup.jpg", "a", "a:0", 1, 1])
    dedup.append(["dup", "0", 1, "a_0_0", "a:0"])
    for name, rows in (("manifest.csv", manifest), ("dedup.csv", dedup)):
        with (pool / name).open("w", newline="") as f:
            csv.writer(f).writerows(rows)
    cfg = {
        "raw_root": str(tmp_path / "raw"),
        "out_dir": str(pool),
        "classes": ["road", "pothole"],
        "box_dup_iou": 0.5,
        "sources": [],
        "split": {"out_dir": str(tmp_path / "merged-base"), "ratios": RATIOS, "seed": 0},
    }
    path = tmp_path / "merge.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _list(out: Path, name: str) -> list[str]:
    return (out / f"{name}.txt").read_text().split()


def test_split_writes_dataset_lists_and_zip(config_path: Path) -> None:
    report = split(config_path, make_zip=True)
    out = config_path.parent / "merged-base"

    lists = {s: _list(out, s) for s in SPLITS}
    all_images = [p for s in SPLITS for p in lists[s]]
    assert len(all_images) == len(set(all_images)) == len(_items())
    assert "./images/dup.jpg" not in all_images
    assert all(p.startswith("./images/") for p in all_images)
    for s in SPLITS:
        real = _list(out, f"{s}_real_road")
        assert set(real) <= set(lists[s])
        assert all(Path(p).name.startswith("a_") for p in real)

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "road", 1: "pothole"}
    assert data["train"] == "train.txt"
    assert (out / "labels" / "a_0_0.txt").exists()
    assert sum(report[s]["images"] for s in SPLITS) == len(_items())
    assert json.loads((out / "report.json").read_text())["seed"] == 0

    with zipfile.ZipFile(f"{out}.zip") as z:
        assert "merged-base/data.yaml" in z.namelist()
