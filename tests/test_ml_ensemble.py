"""Tests for spectraquant.ml.ensemble – ensemble probability and signal."""
from __future__ import annotations

import numpy as np
import pytest

from spectraquant.ml.ensemble import ensemble_probability, ensemble_to_signal


def test_ensemble_probability_basic():
    rf = np.array([0.6, 0.4, 0.7])
    xgb = np.array([0.5, 0.5, 0.8])
    result = ensemble_probability(rf, xgb)
    assert result.shape == (3,)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_ensemble_probability_without_ts():
    """Without ts_signal weights must renormalise to sum to 1."""
    rf = np.array([1.0, 0.0])
    xgb = np.array([1.0, 0.0])
    result = ensemble_probability(rf, xgb, ts_signal=None, w_rf=0.4, w_xgb=0.4, w_ts=0.2)
    # With renormalisation: w_rf=0.5, w_xgb=0.5
    np.testing.assert_allclose(result, [1.0, 0.0])


def test_ensemble_probability_with_ts():
    rf = np.array([0.6, 0.4])
    xgb = np.array([0.6, 0.4])
    ts = np.array([0.6, 0.4])
    result = ensemble_probability(rf, xgb, ts_signal=ts, w_rf=0.4, w_xgb=0.4, w_ts=0.2)
    expected = 0.4 * 0.6 + 0.4 * 0.6 + 0.2 * 0.6
    np.testing.assert_allclose(result[0], expected, rtol=1e-6)


def test_ensemble_probability_clipped():
    """Output must be clipped to [0, 1] even with extreme inputs."""
    rf = np.array([2.0, -1.0])
    xgb = np.array([2.0, -1.0])
    result = ensemble_probability(rf, xgb)
    assert result[0] <= 1.0
    assert result[1] >= 0.0


def test_ensemble_probability_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        ensemble_probability(np.array([0.5, 0.5]), np.array([0.5]))


def test_ensemble_probability_invalid_weight_raises():
    rf = np.array([0.5])
    xgb = np.array([0.5])
    with pytest.raises(ValueError, match="w_rf"):
        ensemble_probability(rf, xgb, w_rf=1.5)


def test_ensemble_probability_ts_scalar_broadcast():
    """A scalar ts_signal must broadcast correctly."""
    rf = np.array([0.6, 0.4, 0.7])
    xgb = np.array([0.5, 0.5, 0.6])
    ts = np.array([0.5])
    result = ensemble_probability(rf, xgb, ts_signal=ts, w_rf=0.4, w_xgb=0.4, w_ts=0.2)
    assert result.shape == (3,)


def test_ensemble_to_signal_buy():
    probs = np.array([0.8, 0.9])
    signals = ensemble_to_signal(probs, threshold=0.55)
    assert (signals == 1).all()


def test_ensemble_to_signal_sell():
    probs = np.array([0.2, 0.1])
    signals = ensemble_to_signal(probs, threshold=0.55)
    assert (signals == -1).all()


def test_ensemble_to_signal_hold():
    probs = np.array([0.5, 0.53])
    signals = ensemble_to_signal(probs, threshold=0.55)
    assert (signals == 0).all()


def test_ensemble_to_signal_values_restricted():
    probs = np.random.default_rng(0).random(100)
    signals = ensemble_to_signal(probs, threshold=0.6)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_ensemble_to_signal_invalid_threshold_raises():
    probs = np.array([0.5])
    with pytest.raises(ValueError, match="threshold"):
        ensemble_to_signal(probs, threshold=0.5)
    with pytest.raises(ValueError, match="threshold"):
        ensemble_to_signal(probs, threshold=1.0)


def test_ensemble_to_signal_index_attached():
    """When an index is provided it must appear on the returned Series."""
    import pandas as pd

    probs = np.array([0.7, 0.3, 0.5])
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    signals = ensemble_to_signal(probs, index=idx)
    assert signals.index.equals(idx)


def test_ensemble_probability_deterministic():
    rf = np.array([0.6, 0.4, 0.55])
    xgb = np.array([0.7, 0.3, 0.50])
    r1 = ensemble_probability(rf, xgb)
    r2 = ensemble_probability(rf, xgb)
    np.testing.assert_array_equal(r1, r2)
