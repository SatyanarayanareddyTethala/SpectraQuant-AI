"""Tests for spectraquant.ml.models – model factories and metrics."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from spectraquant.ml.models import HAS_XGB, classification_metrics, get_random_forest, get_xgboost


def _binary_data(n: int = 200):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + rng.standard_normal(n) > 0).astype(int)
    return X, y


def test_get_random_forest_returns_rf():
    model = get_random_forest()
    assert isinstance(model, RandomForestClassifier)


def test_get_random_forest_parameters():
    model = get_random_forest()
    assert model.n_estimators == 300
    assert model.max_depth == 6
    assert model.min_samples_leaf == 10
    assert model.random_state == 42


def test_get_random_forest_trains_and_predicts():
    model = get_random_forest()
    X, y = _binary_data()
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})
    assert len(preds) == len(y)


def test_get_random_forest_predict_proba():
    model = get_random_forest()
    X, y = _binary_data()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_get_xgboost_or_raise():
    if not HAS_XGB:
        with pytest.raises(ImportError, match="xgboost"):
            get_xgboost()
    else:
        model = get_xgboost()
        assert model is not None


def test_get_xgboost_trains_when_available():
    if not HAS_XGB:
        pytest.skip("xgboost not installed")
    model = get_xgboost()
    X, y = _binary_data()
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


def test_classification_metrics_perfect():
    y = np.array([0, 1, 0, 1, 1])
    metrics = classification_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_classification_metrics_keys():
    y = np.array([0, 1, 0, 1])
    p = np.array([1, 1, 0, 0])
    metrics = classification_metrics(y, p)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    for v in metrics.values():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_get_random_forest_reproducible():
    """Two calls with same data must produce identical predictions."""
    m1 = get_random_forest()
    m2 = get_random_forest()
    X, y = _binary_data(300)
    m1.fit(X, y)
    m2.fit(X, y)
    assert np.array_equal(m1.predict(X), m2.predict(X))
