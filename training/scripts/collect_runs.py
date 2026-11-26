"""Collect finished Colab runs from the synced Google Drive folder (Mac only).

Google Drive for desktop shows Drive as a local folder. For every run under
<drive>/runs/ that has a summary.json (written by train.py when a run ends):
- copies the small report files to docs/runs/<run>/ (never the weights),
- appends the run's row to docs/results.md, unless a row with the same run
  name and commit is already there (so running this twice adds nothing).
Runs still training (weights but no summary.json) are listed as in progress.

The Drive folder is found automatically (~/Library/CloudStorage/GoogleDrive-*/
My Drive/road-pothole-seg); set ROAD_POTHOLE_DRIVE or pass --drive to override.

Usage:
    python -m training.scripts.collect_runs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from training.scripts.train import REPO_ROOT, results_row

REPORT_FILES = ("summary.json", "results.csv", "results.png", "args.yaml", "epochs.csv")
DRIVE_GLOB = "Library/CloudStorage/GoogleDrive-*/My Drive/road-pothole-seg"
ENV_VAR = "ROAD_POTHOLE_DRIVE"


@dataclass
class CollectReport:
    added_rows: list[str] = field(default_factory=list)
    copied: list[str] = field(default_factory=list)
    already_logged: list[str] = field(default_factory=list)
    in_progress: list[str] = field(default_factory=list)


def find_drive_root(explicit: Path | None = None, home: Path | None = None) -> Path:
    """The project folder on the synced Drive: --drive, then $ROAD_POTHOLE_DRIVE, then search."""
    if explicit is not None:
        candidates = [explicit]
    elif os.environ.get(ENV_VAR):
        candidates = [Path(os.environ[ENV_VAR])]
    else:
        candidates = sorted((home or Path.home()).glob(DRIVE_GLOB))
    candidates = [c for c in candidates if c.is_dir()]
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected one Drive project folder, found {len(candidates)}. "
            f"Is Google Drive for desktop running? Set {ENV_VAR} to choose one."
        )
    return candidates[0]


def _row_key(row: str) -> tuple[str, str]:
    """(run name, commit) of a results.md row."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    return cells[1], cells[10]


def logged_keys(results_md: Path) -> set[tuple[str, str]]:
    keys = set()
    for line in results_md.read_text().splitlines():
        if line.startswith("| 20"):  # data rows start with the date
            keys.add(_row_key(line))
    return keys


def collect(drive_root: Path, repo_root: Path = REPO_ROOT) -> CollectReport:
    """Copy report files of finished runs and log new ones in docs/results.md."""
    report = CollectReport()
    runs_dir = drive_root / "runs"
    if not runs_dir.is_dir():
        return report
    results_md = repo_root / "docs" / "results.md"
    known = logged_keys(results_md)
    new_rows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            if (run_dir / "weights").is_dir():
                report.in_progress.append(run_dir.name)
            continue
        dest = repo_root / "docs" / "runs" / run_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in REPORT_FILES:
            if (run_dir / name).is_file():
                shutil.copy2(run_dir / name, dest / name)
        report.copied.append(run_dir.name)

        row = results_row(json.loads(summary_path.read_text()))
        if _row_key(row) in known:
            report.already_logged.append(run_dir.name)
        else:
            new_rows.append(row)
            known.add(_row_key(row))
    if new_rows:
        text = results_md.read_text()
        results_md.write_text(text.rstrip("\n") + "\n" + "\n".join(new_rows) + "\n")
    report.added_rows = new_rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--drive", type=Path, help="Drive project folder (overrides search)")
    args = parser.parse_args()
    report = collect(find_drive_root(args.drive))
    print(f"Copied reports: {report.copied or 'none'}")
    print(f"Already in results.md: {report.already_logged or 'none'}")
    print(f"Still training: {report.in_progress or 'none'}")
    print(f"New rows in docs/results.md: {len(report.added_rows)}")
    for row in report.added_rows:
        print(row)


if __name__ == "__main__":
    main()
