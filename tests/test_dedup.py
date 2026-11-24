import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from training.scripts.dedup import (
    BANDS,
    DedupSettings,
    PoolImage,
    UnionFind,
    apply,
    close_pairs,
    dhash,
    hamming,
    load_settings,
    make_gallery,
    resolve,
)


def _scene(seed: int) -> np.ndarray:
    """A random blocky picture; different seeds give unrelated pictures."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return cv2.resize(blocks, (128, 96), interpolation=cv2.INTER_NEAREST)


def _img(name: str, group: str, real_road: bool = True, n_pothole: int = 1) -> PoolImage:
    return PoolImage(name, Path(f"{name}.jpg"), name[:2], group, real_road, n_pothole)


def test_dhash_identical_and_brightened_copies_are_close() -> None:
    scene = _scene(1)
    brighter = cv2.convertScaleAbs(scene, alpha=1.0, beta=20)
    assert dhash(scene) == dhash(scene.copy())
    a, b, c = (np.array([dhash(x)], dtype=np.uint64) for x in (scene, brighter, _scene(2)))
    assert hamming(a, b)[0] <= 4
    assert hamming(a, c)[0] > 12


def test_hamming_counts_bits() -> None:
    a = np.array([0, 0b1011, 2**64 - 1], dtype=np.uint64)
    b = np.array([0, 0, 0], dtype=np.uint64)
    assert hamming(a, b).tolist() == [0, 3, 64]


def test_close_pairs_finds_all_pairs_within_distance() -> None:
    hashes = np.array([0b0, 0b1, 0b11, 2**64 - 1], dtype=np.uint64)
    assert close_pairs(hashes, 2) == [(0, 1, 1), (0, 2, 2), (1, 2, 1)]


def test_union_find_joins_transitively() -> None:
    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(2) == uf.find(0) != uf.find(3)


def test_resolve_removes_duplicates_and_links_groups() -> None:
    images = [
        _img("gn_a", "gn:a", real_road=False, n_pothole=3),
        _img("pr_b", "pr:b", real_road=True, n_pothole=1),  # duplicate of gn_a; kept (real road)
        _img("pr_c", "pr:c"),  # near-duplicate of pr_b -> same group
        _img("pr_d", "pr:vid"),
        _img("pr_e", "pr:vid"),  # same filename group as pr_d
        _img("pr_f", "pr:f"),  # unrelated
    ]
    pairs = [(0, 1, 2), (1, 2, 8), (0, 5, 30)]
    result = resolve(images, pairs, DedupSettings(remove_max=4, link_max=10))
    assert result.removed == {0: 1}
    assert result.group[1] == result.group[2] == result.group[0]
    assert result.group[3] == result.group[4] == "pr:vid"
    assert result.group[5] == "pr:f"


def test_load_settings_rejects_bad_order(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"dedup": {"remove_max": 8, "link_max": 4}}))
    with pytest.raises(ValueError):
        load_settings(path)


@pytest.fixture
def pool_config(tmp_path: Path) -> Path:
    pool = tmp_path / "curated"
    (pool / "images").mkdir(parents=True)
    rows = []
    pictures = {
        "pr_x": _scene(1),
        "pr_x_copy": _scene(1),
        "gn_y": _scene(2),
        "gn_z": _scene(3),
    }
    for name, pixels in pictures.items():
        cv2.imwrite(str(pool / "images" / f"{name}.png"), pixels)
        rows.append([name, f"{name}.png", name[:2], f"{name[:2]}:{name}", int(name[:2] == "pr"), 1])
    with (pool / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "file", "source", "group", "real_road", "n_pothole"])
        writer.writerows(rows)
    (pool / "report.json").write_text(json.dumps({"total_kept": 4}))
    cfg = {
        "raw_root": str(tmp_path / "raw"),
        "out_dir": str(pool),
        "classes": ["road", "pothole"],
        "box_dup_iou": 0.5,
        "sources": [],
        "dedup": {"remove_max": 4, "link_max": 10},
    }
    path = tmp_path / "merge.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_apply_writes_dedup_csv_and_report(pool_config: Path) -> None:
    summary = apply(pool_config)
    pool = pool_config.parent / "curated"
    with (pool / "dedup.csv").open() as f:
        rows = {r["name"]: r for r in csv.DictReader(f)}
    assert rows["pr_x_copy"]["removed"] == "1"
    assert rows["pr_x_copy"]["duplicate_of"] == "pr_x"
    assert rows["gn_y"]["removed"] == "0"
    assert summary["removed_total"] == 1
    report = json.loads((pool / "report.json").read_text())
    assert report["total_kept"] == 4  # earlier keys are kept
    assert report["dedup"]["kept_total"] == 3


def test_gallery_writes_sheets_and_counts(pool_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "gallery"
    counts = make_gallery(pool_config, out)
    assert [c["band"] for c in counts] == [f"{lo}-{hi}" for lo, hi in BANDS]
    assert counts[0]["pairs"] == 1  # the identical copy
    assert (out / "gallery_d00-02.jpg").exists()
    assert (out / "counts.json").exists()
