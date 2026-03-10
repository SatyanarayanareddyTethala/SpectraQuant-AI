from __future__ import annotations

# replaced hardcoded ticker
import pandas as pd
import pytest

from spectraquant.qa.quality_gates import QualityGateError, run_quality_gates_predictions


def _constant_predictions() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-10", "2024-01-10"], utc=True)
    return pd.DataFrame(
        {
            "ticker": ["TICKER1", "TICKER2"],
            "date": dates,
            "horizon": ["1d", "1d"],
            "score": [50.0, 50.0],
            "probability": [0.5, 0.5],
            "expected_return_annual": [0.0, 0.0],
            "expected_return_horizon": [0.0, 0.0],
            "expected_return": [0.0, 0.0],
            "predicted_return": [0.0, 0.0],
            "target_price": [100.0, 100.0],
            "predicted_return_1d": [0.0, 0.0],
            "target_price_1d": [100.0, 100.0],
            "model_version": [1, 1],
            "factor_set_version": ["test", "test"],
            "regime": ["neutral", "neutral"],
            "schema_version": [4, 4],
            "last_close": [100.0, 100.0],
        }
    )


def _horizon_predictions(horizon: str, returns: list[float]) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-10"] * len(returns), utc=True)
    tickers = [f"TICKER{i}" for i in range(1, len(returns) + 1)]
    scores = [50.0 + i for i in range(len(returns))]
    return pd.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "horizon": [horizon] * len(returns),
            "score": scores,
            "expected_return_horizon": returns,
        }
    )


def test_quality_gates_enforced_without_test_mode() -> None:
    predictions = _constant_predictions()
    with pytest.raises(QualityGateError):
        run_quality_gates_predictions(predictions, {"test_mode": False})


def test_force_pass_tests_bypasses_failures() -> None:
    predictions = _constant_predictions()
    run_quality_gates_predictions(predictions, {"qa": {"force_pass_tests": True}})


def test_daily_predictions_near_constant_fail() -> None:
    predictions = _horizon_predictions("1d", [1e-8, 2e-8, 3e-8])
    with pytest.raises(QualityGateError) as excinfo:
        run_quality_gates_predictions(predictions, {})
    assert any(issue.code == "degenerate_expected_return" for issue in excinfo.value.issues)


def test_intraday_predictions_near_constant_pass() -> None:
    predictions = _horizon_predictions("5m", [1e-8, 2e-8, 3e-8])
    run_quality_gates_predictions(predictions, {})


def test_daily_predictions_near_constant_warn_in_test_mode() -> None:
    predictions = _horizon_predictions("1d", [1e-8, 2e-8, 3e-8])
    run_quality_gates_predictions(predictions, {"test_mode": {"enabled": True}})
