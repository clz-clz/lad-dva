"""Explicitly opt-in provider integration checks; skipped in normal pytest."""
import os

import pytest

import official_preflight
from run_multiseed import _official_settings_from_env


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LAD_RG_LIVE_TESTS") != "1",
    reason="set RUN_LAD_RG_LIVE_TESTS=1 to authorize provider calls",
)


def _configured_live_configs():
    raw = os.environ.get("LAD_RG_LIVE_CONFIGS", "lad_rg_full")
    return [item.strip() for item in raw.split(",") if item.strip()]


def test_configured_provider_passes_live_launch_preflight():
    configs = _configured_live_configs()
    settings, _ = _official_settings_from_env(configs, os.environ)

    report = official_preflight.run_live_preflight(
        settings=settings, config_names=configs,
    )

    assert report["ok"], report
