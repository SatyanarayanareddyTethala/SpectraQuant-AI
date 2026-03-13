"""Tests for spectraquant.ml.importance – feature importance extraction."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from spectraquant.ml.importance import get_feature_importance
from spectraquant.ml.models import get_random_forest


def _trained_rf():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] > 0).astype(int)
    model = get_random_forest()
    model.fit(X, y)
    return model


_FEATURE_COLS = ["alpha", "beta", "gamma", "delta"]


def test_get_feature_importance_returns_dataframe():
    model = _trained_rf()
    df = get_feature_importance(model, _FEATURE_COLS)
    assert hasattr(df, "columns")
    assert "feature" in df.columns
    assert "importance" in df.columns


def test_get_feature_importance_sorted_descending():
    model = _trained_rf()
    df = get_feature_importance(model, _FEATURE_COLS)
    assert list(df["importance"]) == sorted(df["importance"], reverse=True)


def test_get_feature_importance_all_features_present():
    model = _trained_rf()
    df = get_feature_importance(model, _FEATURE_COLS)
    assert set(df["feature"]) == set(_FEATURE_COLS)


def test_get_feature_importance_values_non_negative():
    model = _trained_rf()
    df = get_feature_importance(model, _FEATURE_COLS)
    assert (df["importance"] >= 0).all()


def test_get_feature_importance_sums_to_one():
    model = _trained_rf()
    df = get_feature_importance(model, _FEATURE_COLS)
    assert abs(df["importance"].sum() - 1.0) < 1e-6


def test_get_feature_importance_no_feature_importances_raises():
    """Models without feature_importances_ must raise ValueError."""
    model = LogisticRegression()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    with pytest.raises(ValueError, match="feature_importances_"):
        get_feature_importance(model, _FEATURE_COLS)


def test_get_feature_importance_reset_index():
    """Index must be 0-based after reset."""
    model = _trained_rf()
    df = get_feature_importance(model, _FEATURE_COLS)
    assert list(df.index) == list(range(len(_FEATURE_COLS)))
