"""Tests for horizon-aware annualization in compute_expected_return."""
from __future__ import annotations

import numpy as np
import pytest

from spectraquant.core.predictions import (
    ANNUAL_RETURN_MAX,
    TRADING_DAYS,
    compute_expected_return,
)


REALISTIC_METRICS = {
    "mean_return": 0.001,
    "volatility": 0.012,
    "momentum_daily": 0.0005,
    "rsi": 50.0,
}
FACTOR_SCORE = 0.2
HORIZONS = [1, 5, 20]

# Daily cap derived from annual bound (same formula as in predictions.py)
_DAILY_CAP = float(np.expm1(np.log1p(ANNUAL_RETURN_MAX) / TRADING_DAYS))


def test_annual_return_not_constant_at_max():
    """expected_return_annual must not be pinned to ANNUAL_RETURN_MAX for all horizons."""
    annuals = [
        compute_expected_return(REALISTIC_METRICS, FACTOR_SCORE, h)[3]
        for h in HORIZONS
    ]
    assert not all(a == ANNUAL_RETURN_MAX for a in annuals), (
        f"All annual returns are pinned at {ANNUAL_RETURN_MAX}: {annuals}"
    )


def test_annual_return_has_dispersion():
    """expected_return_annual should not be saturated at the clip bound."""
    annuals = np.array([
        compute_expected_return(REALISTIC_METRICS, FACTOR_SCORE, h)[3]
        for h in HORIZONS
    ])
    # All values should be well below the clip bound
    assert float(np.max(np.abs(annuals))) < float(ANNUAL_RETURN_MAX), (
        f"Annual returns unexpectedly saturated: {annuals}"
    )


def test_clipped_share_low():
    """Fewer than half of annual returns should hit the clip bound with realistic inputs."""
    import pandas as pd

    # Vary mean_return across a realistic range
    mean_returns = np.linspace(-0.002, 0.002, 20)
    annuals = []
    for mr in mean_returns:
        metrics = {**REALISTIC_METRICS, "mean_return": float(mr)}
        annuals.append(compute_expected_return(metrics, FACTOR_SCORE, 20)[3])
    annual_series = pd.Series(annuals)
    clipped_share = (annual_series.abs() >= float(ANNUAL_RETURN_MAX)).mean()
    assert clipped_share <= 0.5, (
        f"Too many annual returns clipped ({clipped_share:.2%}) – saturation still present."
    )


def test_predicted_return_1d_is_capped():
    """predicted_return_1d should be within the daily cap bounds."""
    for h in HORIZONS:
        _, daily, _, _ = compute_expected_return(REALISTIC_METRICS, FACTOR_SCORE, h)
        assert -_DAILY_CAP <= daily <= _DAILY_CAP, (
            f"Daily return {daily} outside cap bounds for horizon {h}"
        )


def test_no_saturation_with_stronger_metrics():
    """Stronger metrics that previously would rail should no longer saturate expected_return_annual."""
    import pandas as pd

    # "Realistic but stronger" metrics – would have railed with the old ±0.01 hard clip
    strong_metrics = {
        "mean_return": 0.008,
        "volatility": 0.018,
        "momentum_daily": 0.006,
        "rsi": 62.0,
    }
    # 30 samples gives enough spread to detect saturation without slowing the suite
    mean_returns = np.linspace(-0.005, 0.010, 30)
    annuals = []
    for mr in mean_returns:
        metrics = {**strong_metrics, "mean_return": float(mr)}
        annuals.append(compute_expected_return(metrics, 1.5, 20)[3])

    annual_series = pd.Series(annuals)
    clipped_share = (annual_series.abs() >= float(ANNUAL_RETURN_MAX)).mean()

    assert not all(a == ANNUAL_RETURN_MAX for a in annuals), (
        "expected_return_annual is still constant at ANNUAL_RETURN_MAX with strong metrics"
    )
    assert clipped_share <= 0.5, (
        f"clipped_share={clipped_share:.2%} > 0.5 – saturation still present with strong metrics"
    )
