import pandas as pd
# replaced hardcoded ticker

from spectraquant.core.predictions import build_prediction_frame


def test_prediction_frame_varies_by_ticker() -> None:
    metrics_by_ticker = {
        "TICKER1": {
            "mean_return": 0.01,
            "volatility": 0.02,
            "momentum_daily": 0.015,
            "rsi": 62.0,
        },
        "TICKER2": {
            "mean_return": -0.005,
            "volatility": 0.03,
            "momentum_daily": -0.01,
            "rsi": 40.0,
        },
    }
    factor_scores = {"TICKER1": 0.6, "TICKER2": -0.4}
    dates = {"TICKER1": pd.Timestamp("2024-01-02"), "TICKER2": pd.Timestamp("2024-01-02")}

    df = build_prediction_frame(
        tickers=["TICKER1", "TICKER2"],
        metrics_by_ticker=metrics_by_ticker,
        factor_scores=factor_scores,
        horizon="5d",
        horizon_days=5.0,
        model_version="test-model",
        factor_set_version="test-factors",
        regime="neutral",
        prediction_dates=dates,
    )

    aaa_return = df.loc[df["ticker"] == "TICKER1", "expected_return"].iloc[0]
    bbb_return = df.loc[df["ticker"] == "TICKER2", "expected_return"].iloc[0]
    assert aaa_return != bbb_return
