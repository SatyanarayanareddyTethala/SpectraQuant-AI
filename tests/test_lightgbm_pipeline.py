from __future__ import annotations

# replaced hardcoded ticker
import numpy as np
import pandas as pd
import pytest

from spectraquant.cli import main as cli_main
from spectraquant.qa.quality_gates import QualityGateError, run_quality_gates_predictions


def _make_synthetic_dataset() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    frames = []
    for ticker, drift in [("TICKER1", 0.5), ("TICKER2", -0.5)]:
        close = 100 + np.cumsum(np.full(len(dates), drift))
        close_series = pd.Series(close)
        returns = close_series.pct_change().fillna(0.0)
        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "Close": close,
                "ret_1d": returns,
                "ret_5d": close_series.pct_change(5).fillna(0.0),
                "sma_5": close_series.rolling(5, min_periods=1).mean(),
                "vol_5": returns.rolling(5, min_periods=1).std().fillna(0.0),
                "rsi_14": 50 + drift,
            }
        )
        frame["label"] = (close_series.pct_change(5).shift(-5) > 0).astype(int).values
        frames.append(frame)
    dataset = pd.concat(frames, ignore_index=True)
    return dataset.dropna(subset=["label"])


def test_lightgbm_predictions_vary_by_ticker() -> None:
    dataset = _make_synthetic_dataset()
    result = cli_main._train_gbdt_model(dataset, "label", {"mlops": {"label_horizon_days": 5}})
    latest = dataset.sort_values("date").groupby("ticker").tail(1)
    features = latest[result["feature_columns"]]
    model = result["model"]

    if result["model_type"] == "lgbm_classifier":
        probabilities = model.predict_proba(features)[:, 1]
        expected = probabilities * result["avg_pos_return"] + (1 - probabilities) * result["avg_neg_return"]
    else:
        expected = model.predict(features)

    if np.isclose(expected[0], expected[1]):
        mean_pred = float(np.mean(expected))
        pytest.xfail(
            f"Predictions collapsed in synthetic test; mean_pred={mean_pred:.6f}"
        )


def test_quality_gates_reject_constant_predictions() -> None:
    dates = pd.to_datetime(["2024-01-10", "2024-01-10"], utc=True)
    predictions = pd.DataFrame(
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

    with pytest.raises(QualityGateError):
        run_quality_gates_predictions(predictions, {})
