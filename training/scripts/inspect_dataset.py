"""Inspect one downloaded YOLO-seg dataset (Roboflow "YOLOv8/YOLO11 segmentation" export).

Reports what the Week 1 quality check needs before any merge code is written:
- images per split, and ORIGINAL images (Roboflow augmentation makes several
  `name.rf.<hash>.jpg` copies of one source image; the plan counts originals)
- class names and annotations per class
- box-only labels (no polygon — useless for segmentation)
- images with no labels, images per class
- pothole size: instance area as a fraction of the image (matches `area_frac`)
- contact sheets of sampled images with masks drawn, for the 50-image eye check

Usage:
    python -m training.scripts.inspect_dataset data/raw/<name>
    python -m training.scripts.inspect_dataset data/raw/<name> --samples 50 --seed 0

Writes outputs/inspect/<name>/summary.json and sheet_XX.jpg. Prints a Markdown block
to paste into docs/datasets.md.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "valid", "val", "test")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)

# Instance size buckets, as a fraction of image area.
SIZE_BUCKETS: tuple[tuple[str, float], ...] = (
    ("tiny <0.1%", 0.001),
    ("small 0.1-1%", 0.01),
    ("medium 1-5%", 0.05),
    ("large >=5%", float("inf")),
)
SHEET_GRID = (5, 2)  # columns, rows per contact sheet
THUMB_WIDTH = 320
PALETTE = [(40, 40, 230), (80, 200, 80), (230, 160, 40), (200, 80, 200), (40, 200, 230)]
BOX_COLOUR = (0, 255, 255)  # yellow: box-only labels in contact sheets


@dataclass
class Instance:
    class_id: int
    polygon: np.ndarray  # (N, 2) normalised xy; empty for box-only labels
    box_only: bool
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # cx, cy, w, h if box_only


@dataclass
class Summary:
    name: str
    class_names: list[str]
    images_per_split: dict[str, int] = field(default_factory=dict)
    originals_per_split: dict[str, int] = field(default_factory=dict)
    originals_in_several_splits: int = 0
    annotations_per_class: dict[str, int] = field(default_factory=dict)
    images_per_class: dict[str, int] = field(default_factory=dict)
    box_only_labels: int = 0
    images_without_labels: int = 0
    size_buckets_per_class: dict[str, dict[str, int]] = field(default_factory=dict)
    median_area_frac_per_class: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- parsing


def source_stem(image_path: Path) -> str:
    """Original image name with Roboflow's `.rf.<hash>` augmentation suffix removed."""
    return RF_SUFFIX.sub("", image_path.stem)


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of an (N, 2) polygon; in normalised coords this is the image fraction."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def parse_label_file(path: Path) -> list[Instance]:
    """Read one YOLO label file. 5 values per line = box only; 7+ = polygon."""
    instances: list[Instance] = []
    if not path.exists():
        return instances
    for line in path.read_text().splitlines():
        values = line.split()
        if not values:
            continue
        class_id = int(float(values[0]))
        coords = np.array(values[1:], dtype=float)
        if len(coords) == 4:
            cx, cy, bw, bh = (float(v) for v in coords)
            instances.append(Instance(class_id, np.empty((0, 2)), True, (cx, cy, bw, bh)))
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            instances.append(Instance(class_id, coords.reshape(-1, 2), box_only=False))
        else:
            raise ValueError(f"{path}: malformed label line: {line!r}")
    return instances


def load_class_names(root: Path) -> list[str]:
    """Class names from the export's data.yaml (list or {id: name} form)."""
    with (root / "data.yaml").open() as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def find_images(root: Path) -> dict[str, list[Path]]:
    """Images per split folder (`<split>/images/*`)."""
    splits: dict[str, list[Path]] = {}
    for split in SPLIT_NAMES:
        image_dir = root / split / "images"
        if image_dir.is_dir():
            splits[split] = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not splits:
        raise SystemExit(f"No <split>/images folders found under {root}")
    return splits


def label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


# ---------------------------------------------------------------- summary


def summarize(root: Path) -> Summary:
    """Count images, originals, classes, box-only labels and instance sizes."""
    names = load_class_names(root)
    summary = Summary(name=root.name, class_names=names)
    splits = find_images(root)

    annotations: Counter[str] = Counter()
    images_with: Counter[str] = Counter()
    areas: dict[str, list[float]] = {n: [] for n in names}
    splits_of_stem: dict[str, set[str]] = {}

    for split, images in splits.items():
        summary.images_per_split[split] = len(images)
        stems = {source_stem(p) for p in images}
        summary.originals_per_split[split] = len(stems)
        for stem in stems:
            splits_of_stem.setdefault(stem, set()).add(split)

        for image in images:
            instances = parse_label_file(label_path(image))
            if not instances:
                summary.images_without_labels += 1
            seen: set[str] = set()
            for inst in instances:
                cls = names[inst.class_id] if inst.class_id < len(names) else f"id{inst.class_id}"
                annotations[cls] += 1
                seen.add(cls)
                if inst.box_only:
                    summary.box_only_labels += 1
                else:
                    areas.setdefault(cls, []).append(polygon_area(inst.polygon))
            images_with.update(seen)

    summary.originals_in_several_splits = sum(len(s) > 1 for s in splits_of_stem.values())
    summary.annotations_per_class = dict(annotations)
    summary.images_per_class = dict(images_with)
    for cls, values in areas.items():
        if not values:
            continue
        buckets = Counter(_bucket(a) for a in values)
        summary.size_buckets_per_class[cls] = {name: buckets[name] for name, _ in SIZE_BUCKETS}
        summary.median_area_frac_per_class[cls] = float(np.median(values))
    return summary


def _bucket(area_frac: float) -> str:
    for name, upper in SIZE_BUCKETS:
        if area_frac < upper:
            return name
    return SIZE_BUCKETS[-1][0]


def to_markdown(s: Summary) -> str:
    """Markdown block for docs/datasets.md."""
    total = sum(s.images_per_split.values())
    originals = sum(s.originals_per_split.values())
    lines = [
        f"#### Counts — `{s.name}`",
        "",
        f"- Classes: {', '.join(f'`{n}`' for n in s.class_names)}",
        f"- Images: {total} ({_fmt(s.images_per_split)})",
        f"- Original images (Roboflow `.rf.` copies removed): {originals} "
        f"({_fmt(s.originals_per_split)})",
        f"- Originals appearing in more than one split (leakage): {s.originals_in_several_splits}",
        f"- Images without labels: {s.images_without_labels}",
        f"- Box-only labels (no polygon): {s.box_only_labels}",
        "",
        "| Class | Annotations | Images | Median area | "
        + " | ".join(n for n, _ in SIZE_BUCKETS)
        + " |",
        "|---|---|---|---|" + "---|" * len(SIZE_BUCKETS),
    ]
    for cls in sorted(s.annotations_per_class, key=lambda c: -s.annotations_per_class[c]):
        buckets = s.size_buckets_per_class.get(cls, {})
        median = s.median_area_frac_per_class.get(cls)
        median_txt = f"{median:.2%}" if median is not None else "—"
        lines.append(
            f"| `{cls}` | {s.annotations_per_class[cls]} | {s.images_per_class.get(cls, 0)} | "
            f"{median_txt} | " + " | ".join(str(buckets.get(n, 0)) for n, _ in SIZE_BUCKETS) + " |"
        )
    return "\n".join(lines)


def _fmt(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items())


# ---------------------------------------------------------------- contact sheets


def draw_labels(image: np.ndarray, instances: list[Instance], names: list[str]) -> np.ndarray:
    """Draw filled, outlined polygons with class names on a copy of `image`.

    Box-only labels are drawn as yellow rectangles so they stand out in the eye check.
    """
    out = image.copy()
    h, w = out.shape[:2]
    fill = out.copy()
    for inst in instances:
        if inst.box_only:
            continue
        colour = PALETTE[inst.class_id % len(PALETTE)]
        pts = (inst.polygon * [w, h]).round().astype(np.int32)
        cv2.fillPoly(fill, [pts], colour)
        cv2.polylines(out, [pts], True, colour, max(1, w // 300))
    # Blend only the polygon fill so box outlines keep full strength.
    out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)
    for inst in instances:
        if inst.box_only:
            cx, cy, bw, bh = inst.box
            p0 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
            p1 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
            cv2.rectangle(out, p0, p1, BOX_COLOUR, max(2, w // 200))
    for inst in instances:
        if inst.box_only or len(inst.polygon) == 0:
            continue
        x, y = (inst.polygon.min(axis=0) * [w, h]).astype(int)
        label = names[inst.class_id] if inst.class_id < len(names) else f"id{inst.class_id}"
        cv2.putText(out, label, (x, max(15, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def make_contact_sheets(root: Path, out_dir: Path, samples: int, seed: int) -> list[Path]:
    """Save grids of `samples` randomly chosen original images with their masks drawn."""
    names = load_class_names(root)
    by_stem: dict[str, Path] = {}
    for images in find_images(root).values():
        for image in images:
            by_stem.setdefault(source_stem(image), image)  # one copy per original
    stems = sorted(by_stem)
    chosen = random.Random(seed).sample(stems, min(samples, len(stems)))

    cols, rows = SHEET_GRID
    per_sheet = cols * rows
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[Path] = []
    for start in range(0, len(chosen), per_sheet):
        thumbs = []
        for stem in chosen[start : start + per_sheet]:
            path = by_stem[stem]
            image = cv2.imread(str(path))
            if image is None:
                continue
            drawn = draw_labels(image, parse_label_file(label_path(path)), names)
            th = int(drawn.shape[0] * THUMB_WIDTH / drawn.shape[1])
            thumb = cv2.resize(drawn, (THUMB_WIDTH, th))
            thumb = cv2.copyMakeBorder(thumb, 0, 22, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.putText(
                thumb, stem[:40], (4, th + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
            thumbs.append(thumb)
        if not thumbs:
            continue
        cell_h = max(t.shape[0] for t in thumbs)
        thumbs = [
            cv2.copyMakeBorder(t, 0, cell_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in thumbs
        ]
        blank = np.zeros_like(thumbs[0])
        thumbs += [blank] * (per_sheet - len(thumbs))
        grid = np.vstack([np.hstack(thumbs[r * cols : (r + 1) * cols]) for r in range(rows)])
        path = out_dir / f"sheet_{start // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), grid)
        sheets.append(path)
    return sheets


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect a YOLO-seg dataset export.")
    parser.add_argument("root", type=Path, help="dataset folder containing data.yaml")
    parser.add_argument("--samples", type=int, default=50, help="images for contact sheets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/inspect/<name>")
    args = parser.parse_args(argv)

    out_dir = args.out or REPO_ROOT / "outputs" / "inspect" / args.root.name
    summary = summarize(args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))
    sheets = make_contact_sheets(args.root, out_dir, args.samples, args.seed)

    print(to_markdown(summary))
    print(f"\nWrote {out_dir / 'summary.json'} and {len(sheets)} contact sheets to {out_dir}")


if __name__ == "__main__":
    main()
