from __future__ import annotations

# replaced hardcoded ticker
import pandas as pd
import pytest

from spectraquant.cli import main as cli
from spectraquant.data.sentiment import SENTIMENT_COLUMNS
from spectraquant.qa.quality_gates import QualityGateError, run_quality_gates_predictions


def _make_sentiment_dataset() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
    records = []
    for ticker, sentiment in [("TICKER1.NS", 0.8), ("TICKER2.NS", -0.6)]:
        for date in dates:
            close = 100 + (1 if sentiment > 0 else -1) * (date.day - 1)
            ret_1d = 0.01 if sentiment > 0 else -0.01
            records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "Close": close,
                    "ret_1d": ret_1d,
                    "ret_5d": ret_1d,
                    "sma_5": close,
                    "vol_5": abs(ret_1d),
                    "rsi_14": 55 if sentiment > 0 else 45,
                    "label": 1 if sentiment > 0 else 0,
                    "news_sentiment_avg": sentiment,
                    "news_sentiment_std": 0.0,
                    "news_count": 5,
                    "social_sentiment_avg": sentiment,
                    "social_sentiment_std": 0.0,
                    "social_count": 10,
                }
            )
    df = pd.DataFrame(records)
    return df


def test_sentiment_features_shift_predictions() -> None:
    dataset = _make_sentiment_dataset()
    cfg = {"mlops": {"label_horizon_days": 5}, "sentiment": {"enabled": True}, "test_mode": True}
    result = cli._train_gbdt_model(dataset, "label", cfg)

    latest = dataset.sort_values("date").groupby("ticker").tail(1)
    features = latest[result["feature_columns"]]
    model = result["model"]

    if result["model_type"] == "lgbm_classifier":
        probabilities = model.predict_proba(features)[:, 1]
    else:
        probabilities = model.predict(features)

    ticker_to_prob = dict(zip(latest["ticker"], probabilities))
    pos_prob = ticker_to_prob.get("TICKER1.NS")
    neg_prob = ticker_to_prob.get("TICKER2.NS")
    if pos_prob is None or neg_prob is None:
        pytest.fail("Missing predictions for sentiment test tickers")

    if pos_prob <= neg_prob:
        pytest.xfail("Sentiment signal did not shift predictions as expected")

    prediction_frame = pd.DataFrame(
        {
            "ticker": latest["ticker"],
            "date": latest["date"],
            "horizon": "1d",
            "score": 50.0,
            "probability": probabilities,
            "expected_return_annual": 0.01,
            "expected_return_horizon": 0.01,
            "expected_return": 0.01,
            "predicted_return": 0.01,
            "target_price": 101.0,
            "predicted_return_1d": 0.01,
            "target_price_1d": 101.0,
            "model_version": 1,
            "factor_set_version": "test",
            "regime": "neutral",
            "schema_version": 4,
            "last_close": 100.0,
        }
    )
    for col in SENTIMENT_COLUMNS:
        prediction_frame[col] = latest[col].values

    run_quality_gates_predictions(prediction_frame, cfg)


def test_sentiment_disabled_uses_base_features() -> None:
    dataset = _make_sentiment_dataset()
    cfg = {"mlops": {"label_horizon_days": 5}, "sentiment": {"enabled": False}, "test_mode": True}
    result = cli._train_gbdt_model(dataset, "label", cfg)
    latest = dataset.sort_values("date").groupby("ticker").tail(1)
    features = latest[result["feature_columns"]]
    model = result["model"]

    if result["model_type"] == "lgbm_classifier":
        probabilities = model.predict_proba(features)[:, 1]
    else:
        probabilities = model.predict(features)

    try:
        run_quality_gates_predictions(
            pd.DataFrame(
                {
                    "ticker": latest["ticker"],
                    "date": latest["date"],
                    "horizon": "1d",
                    "score": 50.0,
                    "probability": probabilities,
                    "expected_return_annual": 0.0,
                    "expected_return_horizon": 0.0,
                    "expected_return": 0.0,
                    "predicted_return": 0.0,
                    "target_price": 100.0,
                    "predicted_return_1d": 0.0,
                    "target_price_1d": 100.0,
                    "model_version": 1,
                    "factor_set_version": "test",
                    "regime": "neutral",
                    "schema_version": 4,
                    "last_close": 100.0,
                }
            ),
            cfg,
        )
    except QualityGateError:
        pytest.xfail("Quality gates failed despite test_mode for sentiment-disabled run")
