"""Grouped, seeded train/val/test split of the de-duplicated pool (pipeline stage 4).

Reads <pool>/manifest.csv and <pool>/dedup.csv (from merge_datasets and dedup),
skips removed duplicates, and splits EACH SOURCE on its own so every split has
every kind of road. Whole groups (one video, or near-duplicate frames) go to one
split, so the same scene never appears on both sides: no train/test leakage.

Group assignment per source: shuffle groups with the seed, place the largest
first, and give each group to the split that is furthest below its target.

Writes <out_dir>/images, labels, data.yaml, the image lists
(train/val/test.txt and *_real_road.txt), split.csv and report.json.
`--zip` also writes <out_dir>.zip for upload to Google Drive.

Usage:
    python -m training.scripts.split_dataset --zip
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

from training.scripts.merge_datasets import REPO_ROOT, _resolve, load_config

SPLITS = ("train", "val", "test")


@dataclass
class SplitSettings:
    out_dir: Path
    ratios: dict[str, float]
    seed: int


@dataclass
class Item:
    name: str
    file: str
    source: str
    group: str
    real_road: bool
    n_pothole: int


def load_settings(config_path: Path) -> SplitSettings:
    """The `split` section of configs/merge.yaml."""
    with config_path.open() as f:
        raw = yaml.safe_load(f)["split"]
    ratios = {k: float(v) for k, v in raw["ratios"].items()}
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1) > 1e-6:
        raise ValueError(f"ratios must have keys {SPLITS} and sum to 1, got {ratios}")
    return SplitSettings(_resolve(raw["out_dir"]), ratios, int(raw["seed"]))


def load_items(pool_dir: Path) -> list[Item]:
    """Kept images of the pool, with their final (linked) group from dedup.csv."""
    with (pool_dir / "dedup.csv").open() as f:
        dedup = {r["name"]: r for r in csv.DictReader(f)}
    items = []
    with (pool_dir / "manifest.csv").open() as f:
        for row in csv.DictReader(f):
            d = dedup[row["name"]]
            if d["removed"] == "1":
                continue
            items.append(
                Item(
                    name=row["name"],
                    file=row["file"],
                    source=row["source"],
                    group=d["group"],
                    real_road=row["real_road"] == "1",
                    n_pothole=int(row["n_pothole"]),
                )
            )
    return items


def assign_groups(items: list[Item], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Group -> split, per source; whole groups only, largest first, to the neediest split."""
    rng = random.Random(seed)
    by_source: dict[str, dict[str, int]] = defaultdict(Counter)
    for it in items:
        by_source[it.source][it.group] += 1

    assignment: dict[str, str] = {}
    for source in sorted(by_source):
        sizes = by_source[source]
        total = sum(sizes.values())
        target = {s: ratios[s] * total for s in SPLITS}
        filled = dict.fromkeys(SPLITS, 0)
        groups = sorted(sizes)
        rng.shuffle(groups)
        groups.sort(key=lambda g: -sizes[g])  # stable: equal sizes keep the shuffled order
        for g in groups:
            split = max(SPLITS, key=lambda s: (target[s] - filled[s]) / target[s])
            assignment[g] = split
            filled[split] += sizes[g]
    return assignment


def check_no_leakage(items: list[Item], split_of: dict[str, str]) -> None:
    """Raise if any group's images ended up in more than one split."""
    seen: dict[str, set[str]] = defaultdict(set)
    for it in items:
        seen[it.group].add(split_of[it.name])
    leaked = [g for g, s in seen.items() if len(s) > 1]
    if leaked:
        raise AssertionError(f"Groups in several splits: {leaked[:5]}")


def build_report(items: list[Item], split_of: dict[str, str]) -> dict:
    """Images, real-road images, potholes and groups per split and source."""
    report: dict = {}
    for split in SPLITS:
        members = [it for it in items if split_of[it.name] == split]
        per_source = {}
        for source in sorted({it.source for it in items}):
            src = [it for it in members if it.source == source]
            per_source[source] = {
                "images": len(src),
                "potholes": sum(it.n_pothole for it in src),
                "groups": len({it.group for it in src}),
            }
        report[split] = {
            "images": len(members),
            "share": round(len(members) / len(items), 4),
            "real_road_images": sum(it.real_road for it in members),
            "potholes": sum(it.n_pothole for it in members),
            "per_source": per_source,
        }
    return report


def write_split(
    items: list[Item], split_of: dict[str, str], pool_dir: Path, out_dir: Path, classes: list[str]
) -> None:
    """Copy images and labels, and write data.yaml, image lists and split.csv."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)
    lists: dict[str, list[str]] = defaultdict(list)
    for it in sorted(items, key=lambda i: i.name):
        shutil.copy2(pool_dir / "images" / it.file, out_dir / "images" / it.file)
        shutil.copy2(pool_dir / "labels" / f"{it.name}.txt", out_dir / "labels" / f"{it.name}.txt")
        split = split_of[it.name]
        # "./" paths are resolved relative to the list file by Ultralytics.
        lists[split].append(f"./images/{it.file}")
        if it.real_road:
            lists[f"{split}_real_road"].append(f"./images/{it.file}")
    for name in [*SPLITS, *(f"{s}_real_road" for s in SPLITS)]:
        (out_dir / f"{name}.txt").write_text("\n".join(lists[name]) + "\n")
    data = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": dict(enumerate(classes)),
    }
    (out_dir / "data.yaml").write_text(
        "# Paths are relative to this file. Road on pothole-only (gn_) images is not\n"
        "# labelled in this base set; see docs/datasets.md.\n"
        + yaml.safe_dump(data, sort_keys=False)
    )
    with (out_dir / "split.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "split", "source", "group", "real_road"])
        for it in sorted(items, key=lambda i: i.name):
            writer.writerow([it.name, split_of[it.name], it.source, it.group, int(it.real_road)])


def split(config_path: Path, make_zip: bool = False) -> dict:
    """Run the whole split and return the report."""
    merge_cfg = load_config(config_path)
    settings = load_settings(config_path)
    items = load_items(merge_cfg.out_dir)
    group_split = assign_groups(items, settings.ratios, settings.seed)
    split_of = {it.name: group_split[it.group] for it in items}
    check_no_leakage(items, split_of)
    write_split(items, split_of, merge_cfg.out_dir, settings.out_dir, merge_cfg.classes)
    report = {"seed": settings.seed, "ratios": settings.ratios, **build_report(items, split_of)}
    (settings.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    if make_zip:
        out = settings.out_dir
        shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "merge.yaml")
    parser.add_argument("--zip", action="store_true", help="also write <out_dir>.zip")
    args = parser.parse_args()
    print(json.dumps(split(args.config, args.zip), indent=2))


if __name__ == "__main__":
    main()
