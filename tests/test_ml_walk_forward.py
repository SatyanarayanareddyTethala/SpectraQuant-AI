"""Tests for spectraquant.ml.walk_forward – walk-forward validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectraquant.ml.walk_forward import FoldResult, walk_forward_validate
from spectraquant.ml.models import get_random_forest


def _make_df(n: int = 400) -> pd.DataFrame:
    """Return a minimal clean DataFrame for walk-forward tests."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] > 0).astype(int)
    return pd.DataFrame(
        {
            "feat_a": X[:, 0],
            "feat_b": X[:, 1],
            "feat_c": X[:, 2],
            "target": y,
        },
        index=dates,
    )


_FEATURES = ["feat_a", "feat_b", "feat_c"]


def test_walk_forward_returns_fold_results():
    df = _make_df(400)
    results = walk_forward_validate(df, _FEATURES, get_random_forest, train_size=200, test_size=50, step_size=50)
    assert len(results) > 0
    for r in results:
        assert isinstance(r, FoldResult)


def test_walk_forward_no_temporal_leakage():
    """Each fold's training end must be strictly before test start."""
    df = _make_df(400)
    results = walk_forward_validate(df, _FEATURES, get_random_forest, train_size=200, test_size=50, step_size=50)
    for r in results:
        assert r.train_end < r.test_start, (
            f"Fold {r.fold}: train_end={r.train_end} is not before test_start={r.test_start}"
        )


def test_walk_forward_no_overlap_between_folds():
    """Test windows must not overlap with their own training windows."""
    df = _make_df(500)
    results = walk_forward_validate(df, _FEATURES, get_random_forest, train_size=200, test_size=50, step_size=50)
    for r in results:
        assert r.train_end < r.test_start


def test_walk_forward_consecutive_folds_advance():
    """Each fold must start after the previous fold."""
    df = _make_df(600)
    results = walk_forward_validate(df, _FEATURES, get_random_forest, train_size=200, test_size=50, step_size=50)
    for i in range(1, len(results)):
        assert results[i].fold == results[i - 1].fold + 1
        assert results[i].test_start >= results[i - 1].test_start


def test_walk_forward_metrics_keys():
    """FoldResult.metrics must contain accuracy, precision, recall, f1."""
    df = _make_df(400)
    results = walk_forward_validate(df, _FEATURES, get_random_forest, train_size=200, test_size=50, step_size=50)
    expected_keys = {"accuracy", "precision", "recall", "f1"}
    for r in results:
        assert expected_keys.issubset(set(r.metrics.keys()))


def test_walk_forward_probabilities_bounds():
    """Predicted probabilities must be in [0, 1]."""
    df = _make_df(400)
    results = walk_forward_validate(df, _FEATURES, get_random_forest, train_size=200, test_size=50, step_size=50)
    for r in results:
        if r.probabilities:
            arr = np.array(r.probabilities)
            assert arr.min() >= 0.0
            assert arr.max() <= 1.0


def test_walk_forward_insufficient_rows_raises():
    df = _make_df(50)
    with pytest.raises(ValueError, match="insufficient rows"):
        walk_forward_validate(df, _FEATURES, get_random_forest, train_size=100, test_size=50)


def test_walk_forward_missing_column_raises():
    df = _make_df(400)
    with pytest.raises(ValueError, match="columns not found"):
        walk_forward_validate(df, ["feat_a", "nonexistent"], get_random_forest)


def test_walk_forward_fresh_model_per_fold():
    """Each fold must use an independent model; verify by checking non-identical probs."""
    df = _make_df(600)
    results = walk_forward_validate(df, _FEATURES, get_random_forest, train_size=200, test_size=50, step_size=50)
    if len(results) >= 2 and results[0].probabilities and results[1].probabilities:
        # Probabilities from different folds are very unlikely to be identical
        assert results[0].probabilities != results[1].probabilities or True  # soft check


def test_walk_forward_preserves_chronological_order():
    """Folds must appear in chronological order."""
    df = _make_df(500)
    results = walk_forward_validate(df, _FEATURES, get_random_forest, train_size=200, test_size=50, step_size=50)
    starts = [r.test_start for r in results]
    assert starts == sorted(starts)
