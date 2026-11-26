import csv
import json
from pathlib import Path

import pytest

from training.scripts.merge_datasets import REPO_ROOT
from training.scripts.naive_split import NaiveSettings, assign, build, collect, load_settings

ROAD_LINE = "0 0 0.5 1 0.5 1 1 0 1"
HOLE_LINE = "1 0.4 0.7 0.6 0.7 0.6 0.9 0.4 0.9"


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def settings(tmp_path: Path) -> NaiveSettings:
    """Pool: kept k0..k19 (pr, real road) and g0..g9 (gn, pseudo road in merged-v1);
    duplicates: dg1 of g1, dk3 of k3 (real road), dt of the test image k0, dn of g9
    (g9 was rejected in merged-v1, so dn has no road to borrow). k1 is test-grouped
    with k0. g8 was rejected in merged-v1."""
    pool, clean = tmp_path / "pool", tmp_path / "merged-v1"
    for d in (pool, clean):
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir()
    kept = [(f"k{i}", "pr", f"pr:{i // 2 if i < 2 else i}", 1) for i in range(20)]
    kept += [(f"g{i}", "gn", f"gn:{i}", 0) for i in range(10)]
    dups = [("dg1", "gn", "gn:1", 0, "g1"), ("dk3", "pr", "pr:3", 1, "k3"),
            ("dt", "pr", "pr:0", 1, "k0"), ("dn", "gn", "gn:9", 0, "g9")]  # fmt: skip
    manifest, dedup = [], []
    for name, src, group, real in kept:
        manifest.append([name, f"{name}.jpg", src, group, real])
        dedup.append([name, 0, "", group])
    for name, src, group, real, twin in dups:
        manifest.append([name, f"{name}.jpg", src, group, real])
        dedup.append([name, 1, twin, group])
    for name, *_ in manifest:
        (pool / "images" / f"{name}.jpg").write_bytes(b"jpg")
        own = f"{ROAD_LINE}\n{HOLE_LINE}\n" if name.startswith(("k", "dk", "dt")) else HOLE_LINE
        (pool / "labels" / f"{name}.txt").write_text(own + "\n")
    _write_csv(pool / "manifest.csv", ["name", "file", "source", "group", "real_road"], manifest)
    _write_csv(pool / "dedup.csv", ["name", "removed", "duplicate_of", "group"], dedup)

    split_rows = []
    for name, src, group, real in kept:
        if name in ("g8", "g9"):
            continue
        split = "test" if name in ("k0", "k1") else "train"
        split_rows.append([name, split, src, group, real])
        (clean / "images" / f"{name}.jpg").write_bytes(b"jpg")
        label = f"{ROAD_LINE}\n{HOLE_LINE}\n"
        (clean / "labels" / f"{name}.txt").write_text(label)
    _write_csv(clean / "split.csv", ["name", "split", "source", "group", "real_road"], split_rows)
    (clean / "test.txt").write_text("./images/k0.jpg\n./images/k1.jpg\n")
    (clean / "test_real_road.txt").write_text("./images/k0.jpg\n./images/k1.jpg\n")
    (clean / "data.yaml").write_text("names: {0: road, 1: pothole}\n")
    return NaiveSettings(pool, clean, tmp_path / "naive",
                         {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)  # fmt: skip


def test_collect_holds_out_clean_test_groups_and_fixes_labels(settings: NaiveSettings) -> None:
    items, test_names, left_out = collect(settings)
    by_name = {it.name: it for it in items}
    assert test_names == ["k0", "k1"]
    assert not {"k0", "k1", "dt"} & set(by_name)  # test images and their duplicate
    assert left_out == {"clean_test_group": 3, "rejected_in_merged_v1": 2,
                        "duplicate_without_road": 1}  # fmt: skip
    dg1 = by_name["dg1"]  # pothole-only duplicate: own pothole + twin's road
    assert dg1.label == f"{ROAD_LINE}\n{HOLE_LINE}\n" and not dg1.real_road
    assert by_name["dk3"].real_road and by_name["dk3"].duplicate_of == "k3"
    assert len(items) == 18 + 8 + 2


def test_assign_is_random_per_image_and_seeded() -> None:
    from types import SimpleNamespace

    items = [SimpleNamespace(name=f"n{i}") for i in range(100)]
    a = assign(items, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    assert sorted(a.values()).count("train") == 70 and list(a.values()).count("test") == 15
    assert a == assign(items, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)


def test_build_writes_both_test_lists(settings: NaiveSettings) -> None:
    report = build(settings)
    out = settings.out_dir
    assert (out / "test_clean.txt").read_text() == "./images/k0.jpg\n./images/k1.jpg\n"
    assert (out / "labels" / "k0.txt").exists() and (out / "images" / "k1.jpg").exists()
    listed = sum(len((out / f"{s}.txt").read_text().split()) for s in ("train", "val", "test"))
    assert listed == sum(report["images"].values()) == 28
    assert report["pool_images"] == 34 and report["test_clean"]["images"] == 2
    assert sum(report["duplicates"].values()) == 2
    assert json.loads((out / "report.json").read_text())["seed"] == 0


def test_repo_config() -> None:
    s = load_settings(REPO_ROOT / "configs" / "merge.yaml")
    assert s.clean_dir.name == "merged-v1" and s.out_dir.name == "merged-v1-naive"
    assert sum(s.ratios.values()) == pytest.approx(1)
