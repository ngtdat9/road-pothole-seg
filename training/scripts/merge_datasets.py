"""Merge the raw Roboflow exports into one curated pool (pipeline stage 2).

Applies the per-source rules in configs/merge.yaml (decided in docs/datasets.md):
- keep ONE copy per original image (Roboflow `.rf.<hash>` augmentation copies),
- map raw class names to `road` / `pothole`, deleting the rest,
- drop box-only label lines that duplicate a polygon, and drop images that still
  have a box-only object (Ultralytics would otherwise drop every polygon),
- drop excluded images (e.g. drone frames reserved for Project 2),
- drop real-road images that have no road label (visible road must be labelled),
- assign each image a video/sequence group for the grouped split.

Writes <out_dir>/images, <out_dir>/labels (YOLO-seg, polygons only),
<out_dir>/manifest.csv and <out_dir>/report.json, and prints the report.
De-duplication and the split run on this pool in the next steps.

Usage:
    python -m training.scripts.merge_datasets --config configs/merge.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from training.scripts.inspect_dataset import (
    Instance,
    find_images,
    label_path,
    load_class_names,
    parse_label_file,
    polygon_area,
    source_stem,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Copies from these splits are never augmented by Roboflow, so they are preferred.
UNAUGMENTED_SPLITS = ("valid", "val", "test")


@dataclass
class SourceRules:
    slug: str
    prefix: str
    real_road: bool
    class_map: dict[str, str | None]
    exclude: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)


@dataclass
class MergeConfig:
    raw_root: Path
    out_dir: Path
    classes: list[str]
    box_dup_iou: float
    sources: list[SourceRules]


@dataclass
class CuratedImage:
    name: str  # output file stem, e.g. "pr_img-12_jpg"
    source: str
    original: str  # original stem without the `.rf.` suffix
    raw_image: Path
    group: str
    real_road: bool
    instances: list[tuple[int, np.ndarray]]  # (output class id, (N, 2) polygon)

    def area_frac(self, class_id: int) -> float:
        return sum(polygon_area(p) for c, p in self.instances if c == class_id)


@dataclass
class SourceReport:
    images: int = 0
    originals: int = 0
    excluded: int = 0
    dropped_box_only_image: int = 0
    dropped_no_road: int = 0
    kept: int = 0
    box_lines_duplicate: int = 0
    labels_deleted: dict[str, int] = field(default_factory=dict)
    instances_kept: dict[str, int] = field(default_factory=dict)
    groups: int = 0
    largest_group: int = 0


# ---------------------------------------------------------------- config


def load_config(path: Path) -> MergeConfig:
    """Read configs/merge.yaml; relative paths are resolved from the repo root."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    sources = [SourceRules(**s) for s in raw["sources"]]
    prefixes = [s.prefix for s in sources]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"Source prefixes must be unique: {prefixes}")
    for s in sources:
        unknown = {v for v in s.class_map.values() if v is not None} - set(raw["classes"])
        if unknown:
            raise ValueError(f"{s.slug}: class_map targets unknown classes {sorted(unknown)}")
    return MergeConfig(
        raw_root=_resolve(raw["raw_root"]),
        out_dir=_resolve(raw["out_dir"]),
        classes=list(raw["classes"]),
        box_dup_iou=float(raw["box_dup_iou"]),
        sources=sources,
    )


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


# ---------------------------------------------------------------- rules


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _cxcywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _polygon_xyxy(poly: np.ndarray) -> tuple[float, float, float, float]:
    return (float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()),
            float(poly[:, 1].max()))  # fmt: skip


def split_box_only(instances: list[Instance], iou: float) -> tuple[int, int]:
    """Count box-only lines that duplicate a same-class polygon, and those that don't."""
    duplicates = orphans = 0
    polygons = [p for p in instances if not p.box_only]
    for inst in instances:
        if not inst.box_only:
            continue
        box = _cxcywh_to_xyxy(inst.box)
        same_class = [p.polygon for p in polygons if p.class_id == inst.class_id]
        if any(box_iou(box, _polygon_xyxy(p)) > iou for p in same_class):
            duplicates += 1
        else:
            orphans += 1
    return duplicates, orphans


def group_of(stem: str, patterns: list[str], prefix: str) -> str:
    """Video/sequence group of an original: first pattern match, else the image itself."""
    for pattern in patterns:
        m = re.match(pattern, stem)
        if m:
            return f"{prefix}:{m.group(1)}"
    return f"{prefix}:{stem}"


def choose_copy(copies: list[tuple[str, Path, list[Instance]]]) -> tuple[str, Path, list[Instance]]:
    """Pick one copy of an original: an unaugmented split first, then the least zoomed-in.

    Roboflow crop augmentation enlarges every object, so the copy with the smallest
    total polygon area is the closest to the full original frame. Ties go to the
    first file name, so the choice is deterministic.
    """

    def key(c: tuple[str, Path, list[Instance]]) -> tuple[bool, float, str]:
        split, path, instances = c
        area = sum(polygon_area(i.polygon) for i in instances if not i.box_only)
        return (split not in UNAUGMENTED_SPLITS, round(area, 6), path.name)

    return min(copies, key=key)


# ---------------------------------------------------------------- curation


def curate_source(rules: SourceRules, cfg: MergeConfig) -> tuple[list[CuratedImage], SourceReport]:
    """Apply one source's rules; return the kept images and the counts removed."""
    root = cfg.raw_root / rules.slug
    raw_names = load_class_names(root)
    missing = set(raw_names) - set(rules.class_map)
    if missing:
        raise ValueError(f"{rules.slug}: class_map has no entry for {sorted(missing)}")
    road_id = cfg.classes.index("road")
    report = SourceReport()

    copies: dict[str, list[tuple[str, Path, list[Instance]]]] = {}
    for split, images in find_images(root).items():
        for image in images:
            report.images += 1
            instances = parse_label_file(label_path(image))
            copies.setdefault(source_stem(image), []).append((split, image, instances))
    report.originals = len(copies)

    deleted: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()
    group_sizes: Counter[str] = Counter()
    kept: list[CuratedImage] = []
    for stem in sorted(copies):
        if any(re.match(p, stem) for p in rules.exclude):
            report.excluded += 1
            continue
        _, image, instances = choose_copy(copies[stem])

        # Boxes of deleted classes don't matter: they are removed with their class.
        relevant = [i for i in instances if rules.class_map[raw_names[i.class_id]] is not None]
        duplicates, orphans = split_box_only(relevant, cfg.box_dup_iou)
        if orphans:
            report.dropped_box_only_image += 1
            continue
        report.box_lines_duplicate += duplicates

        out: list[tuple[int, np.ndarray]] = []
        for inst in instances:
            raw = raw_names[inst.class_id]
            target = rules.class_map[raw]
            if target is None:
                deleted[raw] += 1
                continue
            if inst.box_only:
                continue  # a duplicate of a polygon (orphans were handled above)
            out.append((cfg.classes.index(target), inst.polygon))

        if rules.real_road and not any(c == road_id for c, _ in out):
            report.dropped_no_road += 1
            continue

        group = group_of(stem, rules.groups, rules.prefix)
        group_sizes[group] += 1
        kept_counts.update(cfg.classes[c] for c, _ in out)
        kept.append(
            CuratedImage(
                name=f"{rules.prefix}_{stem}",
                source=rules.slug,
                original=stem,
                raw_image=image,
                group=group,
                real_road=rules.real_road,
                instances=out,
            )
        )

    report.kept = len(kept)
    report.labels_deleted = dict(deleted)
    report.instances_kept = dict(kept_counts)
    report.groups = len(group_sizes)
    report.largest_group = max(group_sizes.values(), default=0)
    return kept, report


def format_label(instances: list[tuple[int, np.ndarray]]) -> str:
    """YOLO-seg label text: `class x1 y1 x2 y2 ...` per polygon, normalised coords."""
    lines = []
    for class_id, poly in instances:
        coords = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
        lines.append(f"{class_id} {coords}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_pool(images: list[CuratedImage], out_dir: Path) -> None:
    """Copy images (bytes unchanged), write labels and manifest.csv."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["name", "file", "source", "original", "group", "real_road",
             "n_road", "n_pothole", "road_frac", "pothole_frac"]
        )  # fmt: skip
        for img in images:
            file = f"{img.name}{img.raw_image.suffix.lower()}"
            shutil.copy2(img.raw_image, out_dir / "images" / file)
            (out_dir / "labels" / f"{img.name}.txt").write_text(format_label(img.instances))
            n = Counter(c for c, _ in img.instances)
            writer.writerow(
                [img.name, file, img.source, img.original, img.group, int(img.real_road),
                 n[0], n[1], f"{img.area_frac(0):.5f}", f"{img.area_frac(1):.5f}"]
            )  # fmt: skip


def imbalance(images: list[CuratedImage]) -> dict[str, float]:
    """Mean share of image area per class (real-road images only for road)."""
    real = [i for i in images if i.real_road]
    return {
        "road_area_frac_mean_real_road_images": float(np.mean([i.area_frac(0) for i in real]))
        if real
        else 0.0,
        "pothole_area_frac_mean": float(np.mean([i.area_frac(1) for i in images]))
        if images
        else 0.0,
        "images_without_pothole": sum(1 for i in images if i.area_frac(1) == 0),
    }


def merge(cfg: MergeConfig) -> dict:
    """Curate every source, write the pool and return the report."""
    pool: list[CuratedImage] = []
    report: dict = {"sources": {}}
    for rules in cfg.sources:
        kept, src_report = curate_source(rules, cfg)
        pool.extend(kept)
        report["sources"][rules.slug] = asdict(src_report)
    names = [i.name for i in pool]
    if len(set(names)) != len(names):
        raise ValueError("Output names collide; check source prefixes")
    write_pool(pool, cfg.out_dir)
    report["total_kept"] = len(pool)
    report["total_real_road"] = sum(i.real_road for i in pool)
    report["imbalance"] = imbalance(pool)
    (cfg.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "merge.yaml")
    args = parser.parse_args()
    report = merge(load_config(args.config))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
