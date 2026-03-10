"""Tests for predictions schema creation – validates return columns are non-degenerate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectraquant.core.predictions import (
    ANNUAL_RETURN_MAX,
    ANNUAL_RETURN_TARGET,
    TRADING_DAYS,
    build_prediction_frame,
)
from spectraquant.core.schema import _assert_return_columns_not_degenerate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TICKERS = ["ALPHA", "BETA", "GAMMA"]

_METRICS = {
    "ALPHA": {
        "mean_return": 0.012,
        "volatility": 0.018,
        "momentum_daily": 0.008,
        "rsi": 65.0,
    },
    "BETA": {
        "mean_return": -0.004,
        "volatility": 0.025,
        "momentum_daily": -0.006,
        "rsi": 38.0,
    },
    "GAMMA": {
        "mean_return": 0.001,
        "volatility": 0.010,
        "momentum_daily": 0.0,
        "rsi": 50.0,
    },
}

_FACTOR_SCORES = {"ALPHA": 0.8, "BETA": -0.5, "GAMMA": 0.0}
_DATES = {t: pd.Timestamp("2024-06-01") for t in _TICKERS}


def _build(horizon: str, horizon_days: float) -> pd.DataFrame:
    return build_prediction_frame(
        tickers=_TICKERS,
        metrics_by_ticker=_METRICS,
        factor_scores=_FACTOR_SCORES,
        horizon=horizon,
        horizon_days=horizon_days,
        model_version="test-v1",
        factor_set_version="test-factors",
        regime="neutral",
        prediction_dates=_DATES,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_expected_return_annual_varies_across_tickers():
    """expected_return_annual must not be constant when tickers have different metrics."""
    df = _build("20d", 20.0)
    annual = df["expected_return_annual"]
    assert annual.nunique() > 1, (
        f"expected_return_annual is constant at {annual.iloc[0]:.4f} for all tickers. "
        "This indicates a saturation / aliasing bug."
    )


def test_expected_return_annual_not_constant_0_4():
    """expected_return_annual must not be stuck at 0.4 (ANNUAL_RETURN_MAX fallback)."""
    df = _build("20d", 20.0)
    all_at_max = (df["expected_return_annual"] == ANNUAL_RETURN_MAX).all()
    assert not all_at_max, (
        "All expected_return_annual values equal ANNUAL_RETURN_MAX – "
        "tanh saturation or wrong constant fallback."
    )


def test_expected_return_differs_from_expected_return_horizon():
    """expected_return (daily) must differ from expected_return_horizon for horizons > 1d."""
    df = _build("20d", 20.0)
    # For h>1, daily return != horizon return
    identical = (df["expected_return"] == df["expected_return_horizon"]).all()
    assert not identical, (
        "expected_return equals expected_return_horizon for all rows – "
        "they should represent different time windows (daily vs horizon)."
    )


def test_predicted_return_is_horizon_return():
    """predicted_return should equal expected_return_horizon (both are horizon-based)."""
    df = _build("20d", 20.0)
    assert (df["predicted_return"] == df["expected_return_horizon"]).all()


def test_expected_return_annual_annualization_consistency():
    """Spot-check that annualization formula is correct: (1+r_h)^(252/h) - 1 clipped."""
    df = _build("20d", 20.0)
    for _, row in df.iterrows():
        r_h = row["expected_return_horizon"]
        # Formula: compound annualization, clipped at ±ANNUAL_RETURN_MAX
        expected_annual_check = (1 + r_h) ** (TRADING_DAYS / 20.0) - 1
        expected_annual_check = float(np.clip(expected_annual_check, -ANNUAL_RETURN_MAX, ANNUAL_RETURN_MAX))
        assert abs(row["expected_return_annual"] - expected_annual_check) < 1e-9, (
            f"Annualization inconsistency for {row['ticker']}: "
            f"got {row['expected_return_annual']:.6f}, expected {expected_annual_check:.6f}"
        )


def test_three_tickers_all_return_columns_differ():
    """With 3 tickers having different metrics, most return columns should vary."""
    df = _build("5d", 5.0)
    for col in ("expected_return_annual", "expected_return_horizon", "predicted_return"):
        assert df[col].nunique() > 1, (
            f"Column '{col}' is constant across tickers – metrics are different so this "
            "suggests wrong assignment or constant default."
        )


def test_assert_return_columns_not_degenerate_raises_when_annual_constant_returns_vary():
    """_assert_return_columns_not_degenerate raises if annual is constant but returns vary."""
    df = pd.DataFrame({
        "expected_return_annual": [0.4, 0.4, 0.4],
        "expected_return_horizon": [0.05, 0.08, 0.12],
        "predicted_return": [0.05, 0.08, 0.12],
    })
    with pytest.raises(ValueError, match="expected_return_annual is constant"):
        _assert_return_columns_not_degenerate(df)


def test_assert_return_columns_not_degenerate_ok_when_all_zero():
    """No raise when all columns are zero (model predicts zero for all tickers)."""
    df = pd.DataFrame({
        "expected_return_annual": [0.0, 0.0, 0.0],
        "expected_return_horizon": [0.0, 0.0, 0.0],
        "predicted_return": [0.0, 0.0, 0.0],
    })
    _assert_return_columns_not_degenerate(df)  # should not raise


def test_assert_return_columns_not_degenerate_ok_when_annual_varies():
    """No raise when expected_return_annual varies normally."""
    df = pd.DataFrame({
        "expected_return_annual": [0.10, 0.15, 0.22],
        "expected_return_horizon": [0.02, 0.03, 0.05],
        "predicted_return": [0.02, 0.03, 0.05],
    })
    _assert_return_columns_not_degenerate(df)  # should not raise


def test_1d_horizon_columns():
    """For a 1-day horizon, expected_return (daily) ≈ expected_return_horizon (within float precision)."""
    df = _build("1d", 1.0)
    # For 1-day horizon, daily return ≈ horizon return (they differ only by float rounding ~1e-17)
    assert np.allclose(df["expected_return"], df["expected_return_horizon"], rtol=1e-12, atol=1e-14), (
        "For a 1-day horizon, daily and horizon returns should be numerically identical."
    )


def test_new_explainability_columns_present():
    """build_prediction_frame must include new explainability columns."""
    df = _build("5d", 5.0)
    new_cols = ["reason", "event_type", "analysis_model", "expected_move_pct",
                "target_price", "stop_price", "confidence", "risk_score", "news_refs"]
    for col in new_cols:
        assert col in df.columns, f"Column '{col}' missing from prediction frame"


def test_ensure_prediction_columns_adds_new_columns():
    """_ensure_prediction_columns adds new columns when they are absent."""
    from spectraquant.cli.main import _ensure_prediction_columns
    df = pd.DataFrame({
        "ticker": ["A.NS", "B.NS"],
        "expected_return_annual": [0.1, 0.2],
        "expected_return_horizon": [0.02, 0.04],
        "predicted_return": [0.02, 0.04],
        "predicted_return_1d": [0.005, 0.008],
        "expected_return": [0.005, 0.008],
    })
    result = _ensure_prediction_columns(df)
    new_cols = ["reason", "event_type", "analysis_model", "expected_move_pct",
                "confidence", "risk_score", "news_refs", "stop_price"]
    for col in new_cols:
        assert col in result.columns, f"Column '{col}' not added by _ensure_prediction_columns"
    assert result["confidence"].iloc[0] == 0.5
    assert result["stop_price"].iloc[0] == 0.0
    assert isinstance(result["news_refs"].iloc[0], list)
