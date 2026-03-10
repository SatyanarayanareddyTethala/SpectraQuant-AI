from __future__ import annotations

import numpy as np
import pandas as pd

from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine, compute_features


def _ohlcv_df(n: int = 72) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(np.linspace(100.0, 120.0, n), index=idx)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000.0, 3000.0, n),
        },
        index=idx,
    )


def test_news_features_asof_alignment_no_lookahead() -> None:
    prices = _ohlcv_df(8)
    prices.index = pd.date_range("2024-01-01 00:00:00", periods=8, freq="h")

    news = pd.DataFrame(
        {
            "sentiment": [0.2, 0.8, -0.5],
        },
        index=pd.to_datetime(
            [
                "2024-01-01 01:30:00",
                "2024-01-01 02:30:00",
                "2024-01-01 04:30:00",
            ]
        ),
    )

    out = compute_features(prices, symbol="ETH", news_df=news)

    assert np.isnan(out.loc[pd.Timestamp("2024-01-01 01:00:00"), "news_sentiment_1h"])
    assert out.loc[pd.Timestamp("2024-01-01 02:00:00"), "news_sentiment_1h"] == 0.2
    assert out.loc[pd.Timestamp("2024-01-01 03:00:00"), "news_sentiment_1h"] == 0.8
    assert np.isnan(out.loc[pd.Timestamp("2024-01-01 04:00:00"), "news_sentiment_1h"])
    assert out.loc[pd.Timestamp("2024-01-01 05:00:00"), "news_sentiment_1h"] == -0.5


def test_news_features_symbol_filtering() -> None:
    prices = _ohlcv_df(6)
    prices.index = pd.date_range("2024-01-01 00:00:00", periods=6, freq="h")

    news = pd.DataFrame(
        {
            "symbol": ["ETH", "BTC", "ETH", "BTC"],
            "sentiment": [0.1, 0.9, -0.2, -0.8],
        },
        index=pd.to_datetime(
            [
                "2024-01-01 01:10:00",
                "2024-01-01 01:20:00",
                "2024-01-01 02:10:00",
                "2024-01-01 02:20:00",
            ]
        ),
    )

    out_eth = compute_features(prices, symbol="ETH", news_df=news)
    out_btc = compute_features(prices, symbol="BTC", news_df=news)

    assert out_eth.loc[pd.Timestamp("2024-01-01 03:00:00"), "news_sentiment_1h"] == -0.2
    assert out_btc.loc[pd.Timestamp("2024-01-01 03:00:00"), "news_sentiment_1h"] == -0.8


def test_news_alignment_prevents_future_leakage() -> None:
    prices = _ohlcv_df(4)
    prices.index = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
            "2024-01-01 03:00:00",
        ]
    )
    news = pd.DataFrame(
        {"sentiment": [1.0]},
        index=pd.to_datetime(["2024-01-01 01:01:00"]),
    )

    out = compute_features(prices, symbol="BTC", news_df=news)

    assert np.isnan(out.loc[pd.Timestamp("2024-01-01 01:00:00"), "news_sentiment_1h"])
    assert out.loc[pd.Timestamp("2024-01-01 02:00:00"), "news_sentiment_1h"] == 1.0


def test_news_volume_and_shock_features_present() -> None:
    prices = _ohlcv_df(72)
    news_idx = pd.date_range("2024-01-01", periods=48, freq="h")
    news = pd.DataFrame({"sentiment": np.sin(np.linspace(0, 2, 48))}, index=news_idx)

    out = compute_features(prices, symbol="BTC", news_df=news)

    assert "news_sentiment_24h" in out.columns
    assert "news_sentiment_3d" in out.columns
    assert "news_volume_24h" in out.columns
    assert "news_shock_zscore" in out.columns
    assert out["news_volume_24h"].max() >= 1.0


def test_context_factors_and_nan_policy_deterministic() -> None:
    prices = _ohlcv_df(220)
    ctx = pd.DataFrame(
        {
            "btc_close": np.linspace(30000.0, 36000.0, 220),
            "market_breadth": np.linspace(0.3, 0.7, 220),
        },
        index=prices.index,
    )

    out = compute_features(
        prices,
        symbol="SOL",
        context_df=ctx,
        nan_policy="zero",
    )

    for col in [
        "price_vs_ma_50",
        "price_vs_ma_200",
        "rolling_volatility",
        "volume_zscore",
        "btc_regime",
        "market_breadth",
        "alt_btc_relative_strength",
    ]:
        assert col in out.columns

    assert out.isna().sum().sum() == 0


def test_manifest_metadata_contains_feature_version() -> None:
    engine = CryptoFeatureEngine(feature_version="2.1.0", nan_policy="keep")
    meta = engine.manifest_metadata()
    assert meta.feature_version == "2.1.0"
    assert meta.nan_policy == "keep"
    assert meta.feature_set == "crypto_ohlcv_news_context"
