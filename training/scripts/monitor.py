"""Live training panel for the Colab notebook (Ultralytics callbacks, no extra packages).

After every epoch it redraws one HTML panel in the notebook output:
- whole-run progress bar, elapsed time, ETA (median of recent epochs), mosaic-off mark,
- mask mAP50-95 per class: best so far, best epoch, and the reference run (baseline),
- early-stop countdown: epochs since the best one vs `patience`,
- speed: seconds per epoch split into train / val, train images per second, GPU memory,
- charts: mask mAP50-95 per class, seg loss train vs val, and the other losses.

Road mAP here comes from the run's val list, where pothole-only images have no real
road labels, so it is shown as "indicative". The real road number is the end-of-run
check on val_real_road.txt in train.py.

Every epoch is also written to <run dir>/epochs.csv (per-class metrics included, which
Ultralytics' results.csv lacks). On --resume the earlier epochs are read back from it.

Usage (notebook):
    monitor = Monitor(show=Monitor.notebook(), reference=load_reference("baseline"))
    run(..., callbacks=monitor.callbacks())
"""

from __future__ import annotations

import csv
import html
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from training.scripts.train import REPO_ROOT, class_metrics

EPOCHS_CSV = "epochs.csv"
CLASSES = ("pothole", "road")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#898781"
GRID, TILE_BG, TRACK = "#e1e0d9", "#f3f2ee", "#e8e7e1"
GOOD, WARN, BAD = "#0a7d0a", "#9a6700", "#b42318"


# ---------------------------------------------------------------- records


def _float(x: Any) -> float | None:
    try:
        return float(x.item() if hasattr(x, "item") else x)
    except (TypeError, ValueError):
        return None


def _key(name: str) -> str:
    return name.replace("/", "_").replace("(", "_").replace(")", "").replace("-", "_")


def epoch_record(trainer: Any, train_s: float, val_s: float) -> dict[str, Any]:
    """One flat row of numbers for the epoch that just finished."""
    rec: dict[str, Any] = {"epoch": trainer.epoch + 1, "train_s": round(train_s, 1),
                           "val_s": round(val_s, 1)}  # fmt: skip
    if getattr(trainer, "tloss", None) is not None:
        for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            rec[_key(k)] = _float(v)
    for k, v in (trainer.metrics or {}).items():
        if k.startswith("val/") or k.startswith("metrics/mAP"):
            rec[_key(k)] = _float(v)
    metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
    if metrics is not None and hasattr(metrics, "seg"):
        names = dict(trainer.data["names"])
        for cls in CLASSES:
            m = class_metrics(metrics.seg, names, cls)
            rec[f"{cls}_map50"] = m["map50"] if m else None
            rec[f"{cls}_map50_95"] = m["map50_95"] if m else None
    lr = getattr(trainer, "lr", None) or {}
    rec["lr"] = _float(next(iter(lr.values()), None))
    rec["gpu_gb"] = gpu_memory_gb()
    return rec


def gpu_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_reserved() / 1e9, 1)
    except ImportError:
        pass
    return None


def write_epochs(path: Path, records: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for r in records:
        fields += [k for k in r if k not in fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def read_epochs(path: Path, up_to: int) -> list[dict[str, Any]]:
    """Rows of epochs.csv with epoch <= up_to, numbers parsed (for --resume)."""
    if not path.is_file():
        return []
    with path.open() as f:
        rows = [
            {k: _float(v) if v != "" else None for k, v in r.items()} for r in csv.DictReader(f)
        ]
    for r in rows:
        r["epoch"] = int(r["epoch"])
    return [r for r in rows if r["epoch"] <= up_to]


def load_reference(run: str, runs_dir: Path = REPO_ROOT / "docs" / "runs") -> dict | None:
    """Final pothole/road numbers of a collected run, shown next to the live ones."""
    path = runs_dir / run / "summary.json"
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text())["metrics"]
    return {"name": run, **{c: (metrics.get(c) or {}).get("map50_95") for c in CLASSES}}


# ---------------------------------------------------------------- state


def _fmt_time(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60} h {m % 60:02d} m" if m >= 60 else f"{m} m"


def summarise(records: list[dict], info: dict) -> dict[str, Any]:
    """Numbers the panel shows, from the epoch records and the run settings."""
    done = records[-1]["epoch"] if records else 0
    total = info["epochs"]
    durations = [r["train_s"] + r["val_s"] for r in records]
    recent = durations[-5:]
    per_epoch = statistics.median(recent) if recent else None
    best = None
    scored = [r for r in records if r.get("pothole_map50_95") is not None]
    if scored:
        best = max(scored, key=lambda r: (r["pothole_map50_95"], -r["epoch"]))
    last = records[-1] if records else {}
    train_s = last.get("train_s")
    return {
        "done": done,
        "total": total,
        "fraction": min(1.0, done / total) if total else 0.0,
        "elapsed": sum(durations),
        "eta": per_epoch * (total - done) if per_epoch is not None else None,
        "per_epoch": per_epoch,
        "train_s": train_s,
        "val_s": last.get("val_s"),
        "img_per_s": info["train_images"] / train_s
        if train_s and info.get("train_images")
        else None,
        "gpu_gb": last.get("gpu_gb"),
        "lr": last.get("lr"),
        "best": best,
        "since_best": done - best["epoch"] if best else None,
        "mosaic_off": total - info["close_mosaic"] + 1 if info.get("close_mosaic") else None,
    }


# ---------------------------------------------------------------- drawing

# Colab shows each output in its own frame, so this stylesheet only touches the panel.
STYLE = f"""<style>
.rps {{font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #fff; color: {INK}; border: 1px solid {GRID}; border-radius: 12px;
  padding: 16px 20px; max-width: 960px; font-variant-numeric: tabular-nums}}
.rps .head {{display: flex; justify-content: space-between; align-items: baseline;
  flex-wrap: wrap; gap: 8px}}
.rps .title {{font-size: 15px; font-weight: 500}}
.rps .meta {{color: {INK_2}; font-weight: 400; font-size: 13px}}
.rps .badge {{font-size: 12px; padding: 3px 10px; border-radius: 8px}}
.rps .row {{display: flex; justify-content: space-between; font-size: 13px; color: {INK_2};
  margin: 14px 0 6px}}
.rps .track {{position: relative; height: 8px; background: {TRACK}; border-radius: 4px}}
.rps .fill {{height: 100%; background: {BLUE}; border-radius: 4px}}
.rps .tick {{position: absolute; top: -3px; bottom: -3px; width: 2px; background: {INK_3}}}
.rps .note {{font-size: 11px; color: {INK_3}; margin-top: 4px}}
.rps .right {{text-align: right}}
.rps .grid {{display: grid; gap: 12px; margin-top: 14px}}
.rps .g4 {{grid-template-columns: repeat(auto-fit, minmax(140px, 1fr))}}
.rps .g2 {{grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;
  margin-top: 18px}}
.rps .gs {{grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); margin-top: 10px}}
.rps .tile {{background: {TILE_BG}; border-radius: 8px; padding: 10px 12px}}
.rps .label {{font-size: 12px; color: {INK_2}}}
.rps .value {{font-size: 22px; font-weight: 500}}
.rps .unit {{font-size: 13px; color: {INK_2}}}
.rps .sub {{font-size: 12px}}
.rps .chart-title {{font-size: 13px; font-weight: 500}}
.rps .legend {{display: flex; gap: 14px; font-size: 12px; color: {INK_2}; margin: 2px 0 4px}}
.rps .key {{display: inline-flex; align-items: center; gap: 4px}}
.rps .sw {{width: 10px; height: 10px; border-radius: 2px; display: inline-block}}
.rps .dash {{width: 12px; border-top: 2px dashed; display: inline-block}}
.rps .muted {{color: {INK_3}}}
</style>"""
BADGES = {"training": (BLUE, "#e6f1fb"), "finished": (GOOD, "#eaf3de"),
          "stopped early": (WARN, "#faeeda")}  # fmt: skip


def _nice_range(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    pad = (hi - lo) * 0.08
    return (max(0.0, lo - pad) if lo >= 0 else lo - pad), hi + pad


def line_chart(
    series: list[tuple[str, list[float | None], str, bool]],
    total_epochs: int,
    width: int = 420,
    height: int = 170,
    mark_epoch: int | None = None,
    vline: int | None = None,
    end_labels: bool = True,
    robust: bool = False,
) -> str:
    """Inline SVG line chart over epochs 1..total. `series` = (label, values, colour, dashed).

    `robust` scales the y axis to epochs 3+ once there are enough of them, so the
    first-epoch loss spike does not flatten the rest (higher points are clipped).
    """
    left, right, top, bottom = 40, 46, 8, 22
    pw, ph = width - left - right, height - top - bottom
    svg = f'<svg viewBox="0 0 {width} {height}" width="100%" style="display:block">'
    text = f'fill="{INK_3}" font-size="11"'
    values = [v for _, vals, _, _ in series for v in vals if v is not None]
    if not values:
        return (f'{svg}<text x="{width / 2}" y="{height / 2}" {text} text-anchor="middle">'
                "Waiting for the first epoch</text></svg>")  # fmt: skip
    if robust:
        later = [v for _, vals, _, _ in series for v in vals[2:] if v is not None]
        values = later if len(later) >= 3 else values
    lo, hi = _nice_range(values)

    def x(epoch: float) -> float:
        return left + pw * (epoch - 1) / max(1, total_epochs - 1)

    def y(v: float) -> float:
        return top + ph * (1 - (min(max(v, lo), hi) - lo) / (hi - lo))

    parts = [svg]
    for i in range(4):
        v = lo + (hi - lo) * i / 3
        parts.append(f'<line x1="{left}" x2="{left + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                     f'stroke="{GRID}"/><text x="{left - 6}" y="{y(v) + 4:.1f}" {text} '
                     f'text-anchor="end">{v:.2f}</text>')  # fmt: skip
    for e in sorted({1, max(1, total_epochs // 2), total_epochs}):
        parts.append(
            f'<text x="{x(e):.1f}" y="{height - 6}" {text} text-anchor="middle">{e}</text>'
        )
    if vline and 1 <= vline <= total_epochs:
        parts.append(f'<line x1="{x(vline):.1f}" x2="{x(vline):.1f}" y1="{top}" y2="{top + ph}" '
                     f'stroke="{INK_3}" stroke-dasharray="2 3"/>')  # fmt: skip
    for k, (label, vals, colour, dashed) in enumerate(series):
        pts = [(i + 1, v) for i, v in enumerate(vals) if v is not None]
        if not pts:
            continue
        d = " ".join(f"{'L' if j else 'M'}{x(e):.1f},{y(v):.1f}" for j, (e, v) in enumerate(pts))
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"{dash}>'
                     f"<title>{html.escape(label)}</title></path>")  # fmt: skip
        e, v = pts[-1]
        parts.append(f'<circle cx="{x(e):.1f}" cy="{y(v):.1f}" r="3.5" fill="{colour}" '
                     'stroke="#fff" stroke-width="2"/>')  # fmt: skip
        if end_labels:
            parts.append(f'<text x="{x(e) + 6:.1f}" y="{y(v) + 4:.1f}" fill="{INK_2}" '
                         f'font-size="11">{v:.3f}</text>')  # fmt: skip
        if k == 0 and mark_epoch and vals[mark_epoch - 1] is not None:
            mx, my = x(mark_epoch), y(vals[mark_epoch - 1])
            parts.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="5" fill="none" '
                         f'stroke="{colour}" stroke-width="2">'
                         f"<title>best, epoch {mark_epoch}</title></circle>")  # fmt: skip
    return "".join(parts) + "</svg>"


def _legend(items: list[tuple[str, str, bool]]) -> str:
    keys = "".join(
        f'<span class="key"><span class="{"dash" if dashed else "sw"}" '
        f'style="{"border-color" if dashed else "background"}:{colour}"></span>{label}</span>'
        for label, colour, dashed in items
    )
    return f'<div class="legend">{keys}</div>'


def _tile(label: str, value: str, sub: str, sub_colour: str = INK_3) -> str:
    return (f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="sub" style="color:{sub_colour}">{sub}</div></div>')  # fmt: skip


def _loss_names(records: list[dict]) -> list[str]:
    """Loss names that have a train series, seg first."""
    names = {k[6:] for r in records for k in r if k.startswith("train_") and k.endswith("_loss")}
    return sorted(names, key=lambda n: (n != "seg_loss", n))


def _col(records: list[dict], key: str) -> list[float | None]:
    return [r.get(key) for r in records]


def _progress(s: dict, status: str) -> str:
    times = f"{_fmt_time(s['elapsed'])} elapsed"
    if status == "training" and s["eta"] is not None:
        times += f" · about {_fmt_time(s['eta'])} left"
    total, mosaic = s["total"], s["mosaic_off"]
    tick = note = ""
    if mosaic and 1 < mosaic <= total:
        tick = f'<div class="tick" style="left:{100 * (mosaic - 1) / total:.1f}%"></div>'
        note = f'<div class="note right">mosaic off from epoch {mosaic}</div>'
    return (f'<div class="row"><span>Epoch {s["done"]} of {total}</span><span>{times}</span></div>'
            f'<div class="track"><div class="fill" style="width:{100 * s["fraction"]:.1f}%"></div>'
            f"{tick}</div>{note}")  # fmt: skip


def _tiles(s: dict, last: dict, info: dict, reference: dict | None) -> str:
    best = s["best"]
    if best:
        ref = (reference or {}).get("pothole")
        sub, colour = f"best, epoch {best['epoch']}", INK_3
        if ref is not None:
            diff = best["pothole_map50_95"] - ref
            sub += f" · {html.escape(reference['name'])} {ref:.3f} ({diff:+.3f})"
            colour = GOOD if diff > 0 else INK_3
        pothole = _tile("Pothole mask mAP50-95", f"{best['pothole_map50_95']:.3f}", sub, colour)
    else:
        pothole = _tile("Pothole mask mAP50-95", "—", "after the first epoch")
    road_now = last.get("road_map50_95")
    road_value = f"{road_now:.3f}" if road_now is not None else "—"
    road = _tile("Road mask mAP50-95", road_value, "mixed val, indicative")

    patience = info.get("patience") or 0
    if s["since_best"] is not None and patience:
        left = patience - s["since_best"]
        colour = BAD if left <= 5 else WARN if left <= patience // 2 else INK_3
        value = f'{s["since_best"]} <span class="unit">/ {patience}</span>'
        stop = _tile("Epochs since best", value,
                     f"early stop in {left}" if left > 0 else "early stop now", colour)  # fmt: skip
    else:
        stop = _tile("Epochs since best", "—", f"patience {patience}")

    if s["per_epoch"] is not None:
        bits = [f"train {s['train_s']:.0f} s + val {s['val_s']:.0f} s"]
        bits += [f"{s['img_per_s']:.0f} img/s"] if s["img_per_s"] else []
        bits += [f"{s['gpu_gb']:.1f} GB"] if s["gpu_gb"] is not None else []
        speed = _tile("Seconds per epoch", f"{s['per_epoch']:.0f}", " · ".join(bits))
    else:
        speed = _tile("Seconds per epoch", "—", "after the first epoch")
    return f'<div class="grid g4">{pothole}{road}{stop}{speed}</div>'


def _charts(records: list[dict], s: dict) -> str:
    total, mosaic = s["total"], s["mosaic_off"]
    best = s["best"]["epoch"] if s["best"] else None
    pair = _legend([("train", BLUE, False), ("val", ORANGE, True)])
    big = [(
        "Mask mAP50-95 on val",
        _legend([("pothole", BLUE, False), ("road (indicative)", ORANGE, True)]),
        line_chart([("pothole", _col(records, "pothole_map50_95"), BLUE, False),
                    ("road (indicative)", _col(records, "road_map50_95"), ORANGE, True)],
                   total, mark_epoch=best, vline=mosaic),
    )]  # fmt: skip
    losses = _loss_names(records)
    for name in losses[:1]:
        series = [("train", _col(records, f"train_{name}"), BLUE, False),
                  ("val", _col(records, f"val_{name}"), ORANGE, True)]  # fmt: skip
        big.append((name.replace("_", " ").capitalize(), pair,
                    line_chart(series, total, vline=mosaic, robust=True)))  # fmt: skip
    out = (
        '<div class="grid g2">'
        + "".join(
            f'<div><div class="chart-title">{t}</div>{legend}{chart}</div>'
            for t, legend, chart in big
        )
        + "</div>"
    )

    small = []
    for name in losses[1:]:
        series = [("train", _col(records, f"train_{name}"), BLUE, False)]
        val = _col(records, f"val_{name}")
        if any(val):  # some losses are train-only (val is all zero)
            series.append(("val", val, ORANGE, True))
        now = " · ".join(f"{lbl} {v[-1]:.3f}" for lbl, v, _, _ in series if v[-1] is not None)
        chart = line_chart(series, total, width=230, height=120, vline=mosaic,
                           end_labels=False, robust=True)  # fmt: skip
        small.append(f'<div><div class="label">{name.replace("_", " ")} '
                     f'<span class="muted">· {now}</span></div>{chart}</div>')  # fmt: skip
    if small:
        out += f'<div class="grid gs">{"".join(small)}</div>'
    return out


def render(
    records: list[dict], info: dict, reference: dict | None = None, status: str = "training"
) -> str:
    """The whole panel as one HTML string."""
    s = summarise(records, info)
    last = records[-1] if records else {}
    title = html.escape(info.get("name", "run"))
    fields = ("model", "data", "imgsz", "commit")
    meta = " · ".join(html.escape(str(info[k])) for k in fields if info.get(k))
    colour, bg = BADGES.get(status, (INK_2, TILE_BG))
    lr = f"lr {s['lr']:.2e} · " if s["lr"] else ""
    return (
        f'{STYLE}<div class="rps"><div class="head">'
        f'<div class="title">{title} <span class="meta">{meta}</span></div>'
        f'<span class="badge" style="color:{colour};background:{bg}">{status}</span></div>'
        f"{_progress(s, status)}{_tiles(s, last, info, reference)}{_charts(records, s)}"
        f'<div class="note" style="margin-top:10px">{lr}Updated after every epoch. Road here is '
        "on the mixed val list; the real road score is checked on val_real_road after training."
        "</div></div>"
    )


# ---------------------------------------------------------------- callbacks


class Monitor:
    """Ultralytics callbacks that keep epoch records, epochs.csv and the panel current."""

    def __init__(self, show: Callable[[str], None] | None = None, reference: dict | None = None,
                 commit: str | None = None) -> None:  # fmt: skip
        self.show = show
        self.reference = reference
        self.commit = commit
        self.records: list[dict[str, Any]] = []
        self.info: dict[str, Any] = {}
        self.csv_path: Path | None = None
        self._epoch_start = self._train_end = 0.0

    @staticmethod
    def notebook() -> Callable[[str], None]:
        """A `show` function that redraws one output area in Jupyter/Colab."""
        from IPython.display import HTML, display

        handle = display(HTML("<i>Starting training…</i>"), display_id=True)
        return lambda text: handle.update(HTML(text))

    def callbacks(self) -> dict[str, Callable[[Any], None]]:
        return {
            "on_train_start": self.on_train_start,
            "on_train_epoch_start": self.on_train_epoch_start,
            "on_train_epoch_end": self.on_train_epoch_end,
            "on_fit_epoch_end": self.on_fit_epoch_end,
            "on_train_end": self.on_train_end,
        }

    def on_train_start(self, trainer: Any) -> None:
        a = trainer.args
        dataset = getattr(getattr(trainer, "train_loader", None), "dataset", None)
        self.info = {
            "name": Path(str(trainer.save_dir)).name,
            "model": Path(str(a.model)).stem,
            "data": Path(str(a.data)).stem.removeprefix("data_"),
            "imgsz": a.imgsz,
            "commit": self.commit,
            "epochs": int(trainer.epochs),
            "patience": int(a.patience or 0),
            "close_mosaic": int(a.close_mosaic or 0),
            "train_images": len(dataset) if dataset is not None else None,
        }
        self.csv_path = Path(str(trainer.save_dir)) / EPOCHS_CSV
        self.records = read_epochs(self.csv_path, int(getattr(trainer, "start_epoch", 0)))
        self._draw("training")

    def on_train_epoch_start(self, trainer: Any) -> None:
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer: Any) -> None:
        self._train_end = time.time()

    def on_fit_epoch_end(self, trainer: Any) -> None:
        now = time.time()
        rec = epoch_record(trainer, self._train_end - self._epoch_start, now - self._train_end)
        self.records = [r for r in self.records if r["epoch"] < rec["epoch"]] + [rec]
        if self.csv_path is not None:
            write_epochs(self.csv_path, self.records)
        self._draw("training")

    def on_train_end(self, trainer: Any) -> None:
        done = self.records[-1]["epoch"] if self.records else 0
        self._draw("finished" if done >= self.info.get("epochs", 0) else "stopped early")

    def _draw(self, status: str) -> None:
        if self.show is not None and self.info:
            self.show(render(self.records, self.info, self.reference, status))
