"""Tests for spectraquant.ml.targets – supervised target generation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectraquant.ml.targets import add_target


def _make_close(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"close": prices}, index=dates)


def test_add_target_columns_exist():
    """future_return and target columns must be present."""
    df = _make_close(60)
    out = add_target(df, horizon=1)
    assert "future_return" in out.columns
    assert "target" in out.columns


def test_add_target_horizon_1_nan_tail():
    """Last horizon=1 row must have NaN target."""
    df = _make_close(60)
    out = add_target(df, horizon=1)
    assert pd.isna(out["future_return"].iloc[-1])
    assert pd.isna(out["target"].iloc[-1])


def test_add_target_horizon_5_nan_tail():
    """Last 5 rows must have NaN target when horizon=5."""
    df = _make_close(60)
    out = add_target(df, horizon=5)
    assert out["future_return"].iloc[-5:].isna().all()
    assert out["target"].iloc[-5:].isna().all()


def test_add_target_binary_values():
    """Non-NaN target values must be exactly 0 or 1."""
    df = _make_close(60)
    out = add_target(df, horizon=1)
    clean = out["target"].dropna()
    assert set(clean.unique()).issubset({0, 1})


def test_add_target_future_return_correctness():
    """future_return[i] == close[i+1] / close[i] - 1 for horizon=1."""
    df = _make_close(20)
    out = add_target(df, horizon=1)
    for i in range(len(df) - 1):
        expected = df["close"].iloc[i + 1] / df["close"].iloc[i] - 1
        actual = float(out["future_return"].iloc[i])
        assert abs(actual - expected) < 1e-10, f"Mismatch at row {i}"


def test_add_target_target_aligned_with_future_return():
    """target[i] == 1 iff future_return[i] > 0."""
    df = _make_close(60)
    out = add_target(df, horizon=1)
    clean = out.dropna(subset=["future_return", "target"])
    assert ((clean["future_return"] > 0).astype(int) == clean["target"].astype(int)).all()


def test_add_target_invalid_horizon_raises():
    df = _make_close(20)
    with pytest.raises(ValueError, match="horizon"):
        add_target(df, horizon=0)
    with pytest.raises(ValueError, match="horizon"):
        add_target(df, horizon=-1)


def test_add_target_missing_close_raises():
    df = pd.DataFrame({"open": [1.0, 2.0]})
    with pytest.raises(ValueError, match="close"):
        add_target(df)


def test_add_target_preserves_original_columns():
    """add_target must not drop any existing columns."""
    df = _make_close(20)
    df["extra"] = 99
    out = add_target(df)
    assert "close" in out.columns
    assert "extra" in out.columns
