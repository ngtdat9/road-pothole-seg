import numpy as np
import pytest

from training.scripts.check_onnx import (
    REPO_ROOT,
    VARIANTS,
    load_settings,
    make_variants,
    mask_iou,
    match_instances,
    summarise,
)


def _box(y0: int, x0: int, size: int = 10) -> np.ndarray:
    m = np.zeros((50, 50), dtype=bool)
    m[y0 : y0 + size, x0 : x0 + size] = True
    return m


def test_variants_have_the_expected_shapes() -> None:
    v = make_variants(np.zeros((480, 640, 3), dtype=np.uint8))
    assert list(v) == list(VARIANTS)
    assert v["original"].shape[:2] == (480, 640)
    assert v["landscape"].shape[:2] == (360, 640)
    assert v["portrait"].shape[:2] == (480, 270)
    assert v["small"].shape[:2] == (300, 400)


def test_mask_iou() -> None:
    assert mask_iou(_box(0, 0), _box(0, 5)) == pytest.approx(50 / 150)
    assert mask_iou(np.zeros((5, 5), bool), np.zeros((5, 5), bool)) == 1.0


def test_matching_is_one_to_one_and_reports_leftovers() -> None:
    ours, our_scores = [_box(0, 0), _box(30, 30)], [0.9, 0.4]
    ref, ref_scores = [_box(0, 1), _box(0, 0), _box(20, 0)], [0.5, 0.8, 0.35]
    r = match_instances(ours, our_scores, ref, ref_scores)
    assert [m["iou"] for m in r["matched"]] == [1.0]  # best reference first takes the exact twin
    assert r["unmatched_ours"] == [0.4]
    assert sorted(r["unmatched_ref"]) == [0.35, 0.5]


def test_summary_passes_only_with_high_iou_and_no_confident_misses() -> None:
    good = {"matched": [{"iou": 0.99, "score_diff": 0.001}], "unmatched_ours": [0.26],
            "unmatched_ref": [], "road_iou": 0.995}  # fmt: skip
    assert summarise([good], 0.98, 0.3)["passed"]
    missed = {**good, "unmatched_ref": [0.6]}
    assert not summarise([good, missed], 0.98, 0.3)["passed"]
    low = {**good, "matched": [{"iou": 0.9, "score_diff": 0.0}]}
    assert not summarise([low], 0.98, 0.3)["passed"]


def test_repo_config_loads() -> None:
    cfg = load_settings(REPO_ROOT / "configs" / "export.yaml")
    assert cfg["n_images"] == 20 and cfg["min_iou"] == 0.98
