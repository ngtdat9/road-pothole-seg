"""configs/pipeline.yaml is the source of truth; package defaults must equal it.

Add a check here each time a module with configurable defaults is added
(Segmenter, PotholeTracker).
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from road_pothole_seg import SeverityBins

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def test_config_has_expected_sections(config: dict[str, Any]) -> None:
    assert {"segmenter", "severity", "tracker"} <= config.keys()


def test_severity_bins_match_config(config: dict[str, Any]) -> None:
    defaults = SeverityBins()
    assert tuple(config["severity"]["frac"]) == defaults.frac
    assert tuple(config["severity"]["m2"]) == defaults.m2


def test_config_severity_bins_are_valid(config: dict[str, Any]) -> None:
    SeverityBins(frac=tuple(config["severity"]["frac"]), m2=tuple(config["severity"]["m2"]))
