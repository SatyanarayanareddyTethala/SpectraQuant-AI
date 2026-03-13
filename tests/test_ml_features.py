"""Tests for spectraquant.ml.features – ML feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectraquant.ml.features import ML_FEATURE_COLS, add_features


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with a UTC DatetimeIndex."""
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices - 0.5,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=dates,
    )


def test_add_features_output_columns():
    """All ML_FEATURE_COLS must be present in the output."""
    df = _make_ohlcv(100)
    out = add_features(df)
    for col in ML_FEATURE_COLS:
        assert col in out.columns, f"Missing column: {col}"


def test_add_features_preserves_ohlcv():
    """Original OHLCV columns must be retained."""
    df = _make_ohlcv(60)
    out = add_features(df)
    for col in ("open", "high", "low", "close", "volume"):
        assert col in out.columns


def test_add_features_sentiment_default_zero():
    """When sentiment_score is absent, it must be filled with 0.0."""
    df = _make_ohlcv(60)
    assert "sentiment_score" not in df.columns
    out = add_features(df)
    assert "sentiment_score" in out.columns
    assert (out["sentiment_score"] == 0.0).all()


def test_add_features_sentiment_forwarded():
    """When sentiment_score is present, it must be preserved."""
    df = _make_ohlcv(60)
    df["sentiment_score"] = 0.7
    out = add_features(df)
    assert (out["sentiment_score"] == 0.7).all()


def test_add_features_no_inf():
    """Output must not contain any +/-inf values."""
    df = _make_ohlcv(120)
    out = add_features(df)
    for col in ML_FEATURE_COLS:
        assert not np.isinf(out[col]).any(), f"Column {col!r} contains inf"


def test_add_features_missing_columns_raises():
    """Missing required columns must raise ValueError."""
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        add_features(df)


def test_add_features_index_preserved():
    """DatetimeIndex must be preserved unchanged."""
    df = _make_ohlcv(80)
    out = add_features(df)
    assert out.index.equals(df.index)


def test_add_features_row_count_preserved():
    """add_features must NOT drop any rows (dropna is the caller's job)."""
    df = _make_ohlcv(80)
    out = add_features(df)
    assert len(out) == len(df)
