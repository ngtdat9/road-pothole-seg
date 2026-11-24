import json
from pathlib import Path

import pytest

from training.scripts.collect_runs import ENV_VAR, collect, find_drive_root

HEADER = "# Results\n\n| Date | Run | ... |\n|---|---|---|\n"


def _summary(name: str, commit: str = "abc1234") -> dict:
    return {
        "date": "2026-11-26",
        "profile": name,
        "name": name,
        "commit": commit,
        "config": {
            "model": "yolo26n-seg.pt",
            "data": {"train": "train.txt", "val": "val.txt"},
            "epochs": 5,
            "imgsz": 640,
        },
        "metrics": {
            "pothole": {"map50": 0.4, "map50_95": 0.2},
            "road": {"map50": 0.9, "map50_95": 0.7},
        },
    }


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path]:
    drive = tmp_path / "drive" / "road-pothole-seg"
    runs = drive / "runs"
    finished = runs / "sanity"
    (finished / "weights").mkdir(parents=True)
    (finished / "weights" / "best.pt").write_bytes(b"weights")
    (finished / "summary.json").write_text(json.dumps(_summary("sanity")))
    (finished / "results.csv").write_text("epoch,loss\n1,0.5\n")
    (finished / "results.png").write_bytes(b"png")
    (runs / "baseline" / "weights").mkdir(parents=True)  # still training
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "results.md").write_text(HEADER)
    return drive, repo


def test_collect_copies_reports_and_adds_row(setup: tuple[Path, Path]) -> None:
    drive, repo = setup
    report = collect(drive, repo)
    assert report.copied == ["sanity"]
    assert report.in_progress == ["baseline"]
    dest = repo / "docs" / "runs" / "sanity"
    assert (dest / "summary.json").exists() and (dest / "results.png").exists()
    assert not (dest / "best.pt").exists()
    text = (repo / "docs" / "results.md").read_text()
    assert text.startswith(HEADER)
    assert "| 2026-11-26 | sanity | yolo26n-seg |" in text
    assert "0.400 | 0.200 | 0.900 | 0.700 | abc1234 |" in text


def test_collect_twice_adds_no_duplicate(setup: tuple[Path, Path]) -> None:
    drive, repo = setup
    collect(drive, repo)
    # The user fills in the conclusion; a second collect must not add the row again.
    md = repo / "docs" / "results.md"
    md.write_text(md.read_text().replace("abc1234 | |", "abc1234 | Pipeline works |"))
    report = collect(drive, repo)
    assert report.added_rows == []
    assert report.already_logged == ["sanity"]
    assert md.read_text().count("| sanity |") == 1


def test_collect_adds_row_for_retrained_run(setup: tuple[Path, Path]) -> None:
    drive, repo = setup
    collect(drive, repo)
    summary = drive / "runs" / "sanity" / "summary.json"
    summary.write_text(json.dumps(_summary("sanity", commit="def5678")))
    assert len(collect(drive, repo).added_rows) == 1


def test_collect_without_runs_folder(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "results.md").write_text(HEADER)
    assert collect(tmp_path / "empty", tmp_path).copied == []


def test_find_drive_root_search_env_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    found = tmp_path / "Library/CloudStorage/GoogleDrive-x/My Drive/road-pothole-seg"
    found.mkdir(parents=True)
    assert find_drive_root(home=tmp_path) == found
    with pytest.raises(SystemExit):
        find_drive_root(home=tmp_path / "nowhere")
    other = tmp_path / "custom"
    other.mkdir()
    monkeypatch.setenv(ENV_VAR, str(other))
    assert find_drive_root(home=tmp_path) == other
    assert find_drive_root(explicit=found) == found
