"""Naive merge for the leakage experiment (Week 3, experiment 1; Mac only).

The question: how much do near-duplicates inflate the scores of a naive merge?
A naive merge keeps every duplicate and splits images at random, so copies of one
photo land in both train and test. To measure it fairly:

1. The clean test split of merged-v1 is held out first, together with every pool
   image in the same dedup group (its duplicates and same-scene frames), so the naive
   model never trains on a copy of a clean test image.
2. The rest of the curated pool, duplicates included, is split 70/15/15 at random
   per image (seeded), with no grouping.
3. Labels: images that are in merged-v1 keep their merged-v1 labels (pseudo road and
   dropped off-road labels included). A removed duplicate keeps its own pothole labels;
   if its source has no real road, it borrows the road lines of its kept twin. A
   duplicate whose twin is not in merged-v1 is dropped (its road would be unlabelled),
   as are kept images that merged-v1 rejected.

The naive model is scored on `test.txt` (naive test, leaky) and `test_clean.txt`
(merged-v1's test, copied unchanged). Road uses the `*_real_road.txt` lists only.

Writes <out_dir>/images, labels, data.yaml, the lists, split.csv and report.json.
`--zip` also writes <out_dir>.zip for upload to Google Drive.

Usage:
    python -m training.scripts.naive_split --zip
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from training.scripts.merge_datasets import REPO_ROOT, _resolve
from training.scripts.split_dataset import SPLITS

ROAD, POTHOLE = "0", "1"
CLEAN_TEST = "test_clean"


@dataclass
class NaiveSettings:
    pool_dir: Path  # curated pool: every image, duplicates included, with dedup.csv
    clean_dir: Path  # merged-v1: the clean split with final labels
    out_dir: Path
    ratios: dict[str, float]
    seed: int


@dataclass
class NaiveItem:
    name: str
    file: str
    source: str
    group: str
    duplicate_of: str  # "" for a kept image
    real_road: bool
    label: str  # label file text


def load_settings(config_path: Path) -> NaiveSettings:
    """The `naive` section of configs/merge.yaml (ratios come from `split`)."""
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    n = raw["naive"]
    return NaiveSettings(
        pool_dir=_resolve(raw["out_dir"]),
        clean_dir=_resolve(n["clean_dir"]),
        out_dir=_resolve(n["out_dir"]),
        ratios={k: float(v) for k, v in raw["split"]["ratios"].items()},
        seed=int(n["seed"]),
    )


def _lines(text: str, cls: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.split()[:1] == [cls]]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def collect(s: NaiveSettings) -> tuple[list[NaiveItem], list[str], Counter]:
    """Pool items for the naive split, the clean test names, and what was left out."""
    dedup = {r["name"]: r for r in _csv(s.pool_dir / "dedup.csv")}
    clean = {r["name"]: r for r in _csv(s.clean_dir / "split.csv")}
    test_names = sorted(n for n, r in clean.items() if r["split"] == "test")
    test_groups = {dedup[n]["group"] for n in test_names}
    clean_label = {n: (s.clean_dir / "labels" / f"{n}.txt").read_text() for n in clean}

    items: list[NaiveItem] = []
    left_out: Counter = Counter()
    for row in _csv(s.pool_dir / "manifest.csv"):
        name, d = row["name"], dedup[row["name"]]
        twin = d["duplicate_of"] if d["removed"] == "1" else ""
        if d["group"] in test_groups or (twin and dedup[twin]["group"] in test_groups):
            left_out["clean_test_group"] += 1  # includes the clean test images themselves
            continue
        if not twin:
            if name not in clean:
                left_out["rejected_in_merged_v1"] += 1
                continue
            label, real = clean_label[name], clean[name]["real_road"] == "1"
        else:
            own = (s.pool_dir / "labels" / f"{name}.txt").read_text()
            real = row["real_road"] == "1"
            if real:
                label = own
            elif twin in clean and _lines(clean_label[twin], ROAD):
                lines = _lines(clean_label[twin], ROAD) + _lines(own, POTHOLE)
                label = "\n".join(lines) + "\n"
            else:
                left_out["duplicate_without_road"] += 1
                continue
        items.append(NaiveItem(name, row["file"], row["source"], d["group"], twin, real, label))
    return items, test_names, left_out


def assign(items: list[NaiveItem], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Name -> split: shuffled per image, no grouping (the naive way)."""
    names = sorted(it.name for it in items)
    random.Random(seed).shuffle(names)
    n_train = round(ratios["train"] * len(names))
    n_val = round(ratios["val"] * len(names))
    cuts = {"train": names[:n_train], "val": names[n_train : n_train + n_val],
            "test": names[n_train + n_val :]}  # fmt: skip
    return {n: split for split, members in cuts.items() for n in members}


def leakage(items: list[NaiveItem], split_of: dict[str, str]) -> dict[str, int]:
    """Per eval split: images sharing a dedup group with at least one train image."""
    train_groups = {it.group for it in items if split_of[it.name] == "train"}
    out = {}
    for split in ("val", "test"):
        members = [it for it in items if split_of[it.name] == split]
        out[split] = sum(it.group in train_groups for it in members)
    return out


def build(s: NaiveSettings, make_zip: bool = False) -> dict:
    """Write the naive dataset and return its report."""
    items, test_names, left_out = collect(s)
    split_of = assign(items, s.ratios, s.seed)

    out = s.out_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    lists: dict[str, list[str]] = defaultdict(list)
    for it in sorted(items, key=lambda i: i.name):
        shutil.copy2(s.pool_dir / "images" / it.file, out / "images" / it.file)
        (out / "labels" / f"{it.name}.txt").write_text(it.label)
        lists[split_of[it.name]].append(f"./images/{it.file}")
        if it.real_road:
            lists[f"{split_of[it.name]}_real_road"].append(f"./images/{it.file}")
    for name in (CLEAN_TEST, f"{CLEAN_TEST}_real_road"):  # merged-v1's test, unchanged
        src = (s.clean_dir / f"{name.replace(CLEAN_TEST, 'test')}.txt").read_text().split()
        lists[name] = src
        for line in src:
            file = Path(line).name
            shutil.copy2(s.clean_dir / "images" / file, out / "images" / file)
            stem = Path(file).stem
            shutil.copy2(s.clean_dir / "labels" / f"{stem}.txt", out / "labels" / f"{stem}.txt")
    for name, paths in lists.items():
        (out / f"{name}.txt").write_text("\n".join(paths) + "\n")
    shutil.copy2(s.clean_dir / "data.yaml", out / "data.yaml")
    with (out / "split.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "split", "source", "group", "duplicate_of", "real_road"])
        for it in sorted(items, key=lambda i: i.name):
            w.writerow([it.name, split_of[it.name], it.source, it.group, it.duplicate_of,
                        int(it.real_road)])  # fmt: skip
        for n in test_names:
            w.writerow([n, CLEAN_TEST, "", "", "", ""])

    report = {
        "seed": s.seed,
        "ratios": s.ratios,
        "pool_images": len(items) + sum(left_out.values()),
        "left_out": dict(left_out),
        "images": {sp: sum(v == sp for v in split_of.values()) for sp in SPLITS},
        "duplicates": {
            sp: sum(split_of[it.name] == sp and bool(it.duplicate_of) for it in items)
            for sp in SPLITS
        },  # fmt: skip
        "real_road_images": {sp: len(lists[f"{sp}_real_road"]) for sp in SPLITS},
        "shares_group_with_train": leakage(items, split_of),
        CLEAN_TEST: {
            "images": len(test_names),
            "real_road_images": len(lists[f"{CLEAN_TEST}_real_road"]),
        },  # fmt: skip
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    if make_zip:
        shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "merge.yaml")
    parser.add_argument("--zip", action="store_true", help="also write <out_dir>.zip")
    args = parser.parse_args()
    print(json.dumps(build(load_settings(args.config), args.zip), indent=2))


if __name__ == "__main__":
    main()
