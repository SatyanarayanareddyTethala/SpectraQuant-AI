from __future__ import annotations

import time

import pytest

from spectraquant.core.perf import enforce_stage_budget


def test_perf_budget_enforced() -> None:
    config = {"perf": {"max_seconds": 0.0, "max_mb": 0.0}, "research_mode": False}
    with pytest.raises(RuntimeError):
        with enforce_stage_budget("predict", config):
            time.sleep(0.01)


def test_perf_budget_skipped_in_research() -> None:
    config = {"perf": {"max_seconds": 0.0, "max_mb": 0.0}, "research_mode": True}
    with enforce_stage_budget("predict", config):
        time.sleep(0.01)
