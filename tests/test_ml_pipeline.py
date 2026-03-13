"""Smoke + integration tests for spectraquant.ml.pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectraquant.ml.pipeline import run_ml_pipeline, MLPipelineResult


def _make_ohlcv(n: int = 600) -> pd.DataFrame:
    """Return realistic OHLCV DataFrame with UTC DatetimeIndex."""
    rng = np.random.default_rng(99)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
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


_MINIMAL_CONFIG = {
    "ml": {
        "train_size": 252,
        "test_size": 21,
        "step_size": 21,
        "threshold": 0.55,
        "use_xgboost": False,
        "use_sarimax": False,
    }
}


def test_pipeline_returns_result_type():
    df = _make_ohlcv(600)
    result = run_ml_pipeline(df, config=_MINIMAL_CONFIG)
    assert isinstance(result, MLPipelineResult)


def test_pipeline_signals_shape():
    df = _make_ohlcv(600)
    result = run_ml_pipeline(df, config=_MINIMAL_CONFIG)
    assert "signal_score" in result.signals.columns
    assert "ensemble_prob" in result.signals.columns
    assert len(result.signals) > 0


def test_pipeline_signal_values():
    df = _make_ohlcv(600)
    result = run_ml_pipeline(df, config=_MINIMAL_CONFIG)
    assert set(result.signals["signal_score"].unique()).issubset({-1, 0, 1})


def test_pipeline_ensemble_prob_bounds():
    df = _make_ohlcv(600)
    result = run_ml_pipeline(df, config=_MINIMAL_CONFIG)
    assert result.signals["ensemble_prob"].between(0.0, 1.0).all()


def test_pipeline_fold_results_exist():
    df = _make_ohlcv(600)
    result = run_ml_pipeline(df, config=_MINIMAL_CONFIG)
    assert len(result.rf_fold_results) > 0


def test_pipeline_no_temporal_leakage():
    """Every walk-forward fold must have train_end < test_start."""
    df = _make_ohlcv(600)
    result = run_ml_pipeline(df, config=_MINIMAL_CONFIG)
    for fold in result.rf_fold_results:
        assert fold.train_end < fold.test_start, (
            f"Fold {fold.fold}: train_end={fold.train_end} >= test_start={fold.test_start}"
        )


def test_pipeline_feature_importance_rf():
    df = _make_ohlcv(600)
    result = run_ml_pipeline(df, config=_MINIMAL_CONFIG)
    assert not result.rf_importance.empty
    assert "feature" in result.rf_importance.columns
    assert "importance" in result.rf_importance.columns


def test_pipeline_datetime_index_required():
    df = _make_ohlcv(300)
    df_reset = df.reset_index()  # drops DatetimeIndex
    with pytest.raises((ValueError, TypeError)):
        run_ml_pipeline(df_reset, config=_MINIMAL_CONFIG)


def test_pipeline_pre_2000_index_raises():
    """Timestamps before 2000 indicate a likely epoch-millisecond bug."""
    rng = np.random.default_rng(5)
    prices = 100 + np.cumsum(rng.normal(0, 1, 300))
    old_dates = pd.date_range("1990-01-01", periods=300, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "open": prices - 0.5,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": 1000.0,
        },
        index=old_dates,
    )
    with pytest.raises(ValueError, match="2000"):
        run_ml_pipeline(df, config=_MINIMAL_CONFIG)


def test_pipeline_missing_sentiment_handled():
    """Pipeline must complete without a sentiment_score column."""
    df = _make_ohlcv(600)
    assert "sentiment_score" not in df.columns
    result = run_ml_pipeline(df, config=_MINIMAL_CONFIG)
    assert isinstance(result, MLPipelineResult)


def test_pipeline_metadata_populated():
    df = _make_ohlcv(600)
    result = run_ml_pipeline(df, config=_MINIMAL_CONFIG)
    assert "run_at" in result.metadata
    assert "horizon" in result.metadata
    assert "clean_rows" in result.metadata


def test_pipeline_insufficient_data_raises():
    """Too few rows must raise ValueError, not silently fail."""
    df = _make_ohlcv(50)
    with pytest.raises(ValueError):
        run_ml_pipeline(df, config=_MINIMAL_CONFIG)


def test_pipeline_signals_index_matches_clean_data():
    """Signals index must be a DatetimeIndex with UTC timezone."""
    df = _make_ohlcv(600)
    result = run_ml_pipeline(df, config=_MINIMAL_CONFIG)
    assert isinstance(result.signals.index, pd.DatetimeIndex)
    assert result.signals.index.tz is not None
