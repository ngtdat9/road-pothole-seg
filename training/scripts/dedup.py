"""Find duplicates and near-duplicates in the curated pool (pipeline stage 2).

Every image gets a 64-bit dHash (a perceptual hash: similar pictures get similar
bits). The Hamming distance between two hashes (number of differing bits, 0-64)
says how alike two images are. Two thresholds from configs/merge.yaml:

- distance <= remove_max: the same picture -> keep one, mark the others removed
- distance <= link_max: the same scene or video -> keep both, but give them one
  group so the split puts them on the same side (no train/test leakage)

Commands:
    python -m training.scripts.dedup gallery   # sample pairs per distance band, to pick thresholds
    python -m training.scripts.dedup apply     # write <pool>/dedup.csv and update report.json

`apply` never deletes files; the split step skips rows marked removed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.scripts.merge_datasets import REPO_ROOT, load_config

# Distance bands shown in the gallery (inclusive).
BANDS: tuple[tuple[int, int], ...] = ((0, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12))
MAX_DISTANCE = BANDS[-1][1]
PAIRS_PER_SHEET = 10
THUMB = 240  # px, each image in a pair is fitted into a THUMB x THUMB square
CAPTION_H = 34
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass
class PoolImage:
    name: str
    file: Path
    source: str
    group: str
    real_road: bool
    n_pothole: int


@dataclass
class DedupSettings:
    remove_max: int
    link_max: int


# ---------------------------------------------------------------- loading


def load_pool(pool_dir: Path) -> list[PoolImage]:
    """Rows of the curated pool's manifest.csv, in file order."""
    with (pool_dir / "manifest.csv").open() as f:
        return [
            PoolImage(
                name=row["name"],
                file=pool_dir / "images" / row["file"],
                source=row["source"],
                group=row["group"],
                real_road=row["real_road"] == "1",
                n_pothole=int(row["n_pothole"]),
            )
            for row in csv.DictReader(f)
        ]


def load_settings(config_path: Path) -> DedupSettings:
    """The `dedup` section of configs/merge.yaml."""
    with config_path.open() as f:
        raw = yaml.safe_load(f)["dedup"]
    settings = DedupSettings(int(raw["remove_max"]), int(raw["link_max"]))
    if not 0 <= settings.remove_max <= settings.link_max <= 64:
        raise ValueError(f"Need 0 <= remove_max <= link_max <= 64, got {settings}")
    return settings


# ---------------------------------------------------------------- hashing


def dhash(image: np.ndarray) -> int:
    """64-bit difference hash: is each pixel brighter than its right neighbour (9x8 grid)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hash_images(images: list[PoolImage]) -> np.ndarray:
    """dHash of every image as a uint64 array."""
    hashes = np.empty(len(images), dtype=np.uint64)
    for i, img in enumerate(images):
        pixels = cv2.imread(str(img.file), cv2.IMREAD_GRAYSCALE)
        if pixels is None:
            raise FileNotFoundError(f"Cannot read {img.file}")
        hashes[i] = dhash(pixels)
    return hashes


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise Hamming distance between uint64 arrays (works on numpy 1.x and 2.x)."""
    xor = np.bitwise_xor(a, b).astype(np.uint64)
    return _POPCOUNT[xor.reshape(-1, 1).view(np.uint8)].sum(axis=1).reshape(xor.shape)


def close_pairs(hashes: np.ndarray, max_distance: int) -> list[tuple[int, int, int]]:
    """All index pairs (i < j) with distance <= max_distance, as (i, j, distance)."""
    pairs: list[tuple[int, int, int]] = []
    for i in range(len(hashes) - 1):
        d = hamming(hashes[i + 1 :], hashes[i])
        for k in np.flatnonzero(d <= max_distance):
            pairs.append((i, i + 1 + int(k), int(d[k])))
    return pairs


# ---------------------------------------------------------------- union-find


class UnionFind:
    """Disjoint sets over 0..n-1; `union` joins two sets, `find` names a set."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def sets(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for x in range(len(self.parent)):
            out.setdefault(self.find(x), []).append(x)
        return out


# ---------------------------------------------------------------- apply


@dataclass
class DedupResult:
    removed: dict[int, int]  # removed index -> index of the kept duplicate
    group: list[str]  # final group per image


def keep_rank(img: PoolImage) -> tuple[bool, int, str]:
    """Sort key: the first image of a duplicate set is kept (real road, more potholes, name)."""
    return (not img.real_road, -img.n_pothole, img.name)


def resolve(
    images: list[PoolImage], pairs: list[tuple[int, int, int]], s: DedupSettings
) -> DedupResult:
    """Mark duplicates removed and merge filename groups with near-duplicate links."""
    dup = UnionFind(len(images))
    for i, j, d in pairs:
        if d <= s.remove_max:
            dup.union(i, j)
    removed: dict[int, int] = {}
    for members in dup.sets().values():
        keep = min(members, key=lambda m: keep_rank(images[m]))
        removed.update({m: keep for m in members if m != keep})

    link = UnionFind(len(images))
    first_in_group: dict[str, int] = {}
    for idx, img in enumerate(images):  # same filename group -> same set
        link.union(idx, first_in_group.setdefault(img.group, idx))
    for i, j, d in pairs:  # distance <= remove_max is also <= link_max
        if d <= s.link_max:
            link.union(i, j)
    group = [""] * len(images)
    for members in link.sets().values():
        name = min(images[m].group for m in members)
        for m in members:
            group[m] = name
    return DedupResult(removed, group)


def summarize(images: list[PoolImage], result: DedupResult, s: DedupSettings) -> dict:
    """Counts for report.json: removed per source, groups, and the largest groups."""
    kept = [i for i in range(len(images)) if i not in result.removed]
    sizes = Counter(result.group[i] for i in kept)
    largest = []
    for name, size in sizes.most_common(10):
        sources = Counter(images[i].source for i in kept if result.group[i] == name)
        largest.append({"group": name, "images": size, "sources": dict(sources)})
    return {
        "remove_max": s.remove_max,
        "link_max": s.link_max,
        "removed_per_source": dict(Counter(images[i].source for i in result.removed)),
        "removed_total": len(result.removed),
        "kept_total": len(kept),
        "groups_before_linking": len({images[i].group for i in kept}),
        "groups_after_linking": len(sizes),
        "largest_groups": largest,
    }


def write_dedup(
    pool_dir: Path, images: list[PoolImage], hashes: np.ndarray, result: DedupResult
) -> None:
    with (pool_dir / "dedup.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "dhash", "removed", "duplicate_of", "group"])
        for idx, img in enumerate(images):
            keep = result.removed.get(idx)
            writer.writerow(
                [
                    img.name,
                    f"{int(hashes[idx]):016x}",
                    int(keep is not None),
                    images[keep].name if keep is not None else "",
                    result.group[idx],
                ]
            )


def apply(config_path: Path) -> dict:
    """Hash the pool, resolve duplicates and groups, write dedup.csv, update report.json."""
    pool_dir = load_config(config_path).out_dir
    settings = load_settings(config_path)
    images = load_pool(pool_dir)
    hashes = hash_images(images)
    result = resolve(images, close_pairs(hashes, settings.link_max), settings)
    write_dedup(pool_dir, images, hashes, result)
    summary = summarize(images, result, settings)
    report_path = pool_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["dedup"] = summary
    report_path.write_text(json.dumps(report, indent=2))
    return summary


# ---------------------------------------------------------------- gallery


def _fit(image: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping aspect ratio, pad to a size x size grey square."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = np.full((size, size, 3), 64, np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def draw_pair(a: PoolImage, b: PoolImage, distance: int) -> np.ndarray:
    """Two images side by side with their names and distance underneath."""
    tiles = [_fit(cv2.imread(str(p.file)), THUMB) for p in (a, b)]
    pair = np.hstack([tiles[0], np.full((THUMB, 4, 3), 255, np.uint8), tiles[1]])
    caption = np.full((CAPTION_H, pair.shape[1], 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(caption, f"d={distance}", (4, 14), font, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    for k, img in enumerate((a, b)):
        cv2.putText(caption, img.name[:34], (4 + k * (THUMB + 4), 29), font, 0.35, (0, 0, 0), 1,
                    cv2.LINE_AA)  # fmt: skip
    framed = np.vstack([pair, caption])
    return cv2.copyMakeBorder(framed, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def band_counts(pairs: list[tuple[int, int, int]], images: list[PoolImage]) -> list[dict]:
    """Pairs per distance band, and how many of them join two different sources."""
    out = []
    for lo, hi in BANDS:
        band = [(i, j) for i, j, d in pairs if lo <= d <= hi]
        cross = sum(images[i].source != images[j].source for i, j in band)
        out.append({"band": f"{lo}-{hi}", "pairs": len(band), "cross_source": cross})
    return out


def make_gallery(config_path: Path, out_dir: Path, seed: int = 0) -> list[dict]:
    """One sheet of sampled pairs per distance band; returns the pair counts."""
    images = load_pool(load_config(config_path).out_dir)
    pairs = close_pairs(hash_images(images), MAX_DISTANCE)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for lo, hi in BANDS:
        band = [p for p in pairs if lo <= p[2] <= hi]
        sample = rng.sample(band, min(PAIRS_PER_SHEET, len(band)))
        path = out_dir / f"gallery_d{lo:02d}-{hi:02d}.jpg"
        if not sample:
            path.unlink(missing_ok=True)
            continue
        tiles = [draw_pair(images[i], images[j], d) for i, j, d in sample]
        blank = np.full_like(tiles[0], 255)
        tiles += [blank] * (len(tiles) % 2)
        rows = [np.hstack(tiles[k : k + 2]) for k in range(0, len(tiles), 2)]
        cv2.imwrite(str(path), np.vstack(rows))
    counts = band_counts(pairs, images)
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["gallery", "apply"])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "merge.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "dedup")
    args = parser.parse_args()
    if args.command == "gallery":
        result: object = make_gallery(args.config, args.out)
    else:
        result = apply(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
