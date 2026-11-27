"""configs/pipeline.yaml is the source of truth; package defaults must equal it.

Segmenter and PotholeTracker sections must list exactly their optional arguments,
so adding an argument without adding it to the YAML (or vice versa) fails here.
"""

import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

from road_pothole_seg import PotholeTracker, Segmenter, SeverityBins, analyse_potholes

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def test_config_has_expected_sections(config: dict[str, Any]) -> None:
    assert {"segmenter", "severity", "postprocess", "tracker"} <= config.keys()


def test_severity_bins_match_config(config: dict[str, Any]) -> None:
    defaults = SeverityBins()
    assert tuple(config["severity"]["frac"]) == defaults.frac
    assert tuple(config["severity"]["m2"]) == defaults.m2


def _defaults(func: Any) -> dict[str, Any]:
    return {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_segmenter_defaults_match_config(config: dict[str, Any]) -> None:
    assert _defaults(Segmenter.__init__) == config["segmenter"]


def test_postprocess_defaults_match_config(config: dict[str, Any]) -> None:
    tunable = {k: v for k, v in _defaults(analyse_potholes).items() if v is not None}
    assert tunable == config["postprocess"]  # metres_per_px and severity_bins are per-call


def test_tracker_defaults_match_config(config: dict[str, Any]) -> None:
    assert _defaults(PotholeTracker.__init__) == config["tracker"]


def test_config_severity_bins_are_valid(config: dict[str, Any]) -> None:
    SeverityBins(frac=tuple(config["severity"]["frac"]), m2=tuple(config["severity"]["m2"]))
