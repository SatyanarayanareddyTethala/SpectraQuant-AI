"""Walk-forward time-series validation for SpectraQuant ML.

Anti-leakage design
-------------------
* Data is **never shuffled** – splits are purely positional / temporal.
* Each fold's training window ends strictly before its test window begins.
* ``train_size`` / ``test_size`` / ``step_size`` are expressed in rows, not
  calendar days, so the caller controls granularity via their DataFrame.

Usage example
-------------
::

    from spectraquant.ml.features import add_features, ML_FEATURE_COLS
    from spectraquant.ml.targets import add_target
    from spectraquant.ml.models import get_random_forest
    from spectraquant.ml.walk_forward import walk_forward_validate

    df = add_features(raw_ohlcv)
    df = add_target(df, horizon=1).dropna(subset=ML_FEATURE_COLS + ["target"])
    results = walk_forward_validate(df, ML_FEATURE_COLS, get_random_forest)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import pandas as pd

from spectraquant.ml.models import classification_metrics


@dataclass
class FoldResult:
    """Outcome of a single walk-forward fold."""

    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: dict = field(default_factory=dict)
    """Classification metrics from :func:`~spectraquant.ml.models.classification_metrics`."""
    probabilities: list = field(default_factory=list)
    """Predicted class-1 probabilities for the test window (empty if unavailable)."""


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable,
    train_size: int = 252,
    test_size: int = 21,
    step_size: int = 21,
) -> List[FoldResult]:
    """Evaluate a model factory using expanding-window walk-forward validation.

    Parameters
    ----------
    df:
        DataFrame with feature columns and a ``target`` column.
        **Must already be clean** (no NaN in feature_cols or target).
        The row order is treated as chronological; do not shuffle.
    feature_cols:
        Column names used as model inputs.
    model_factory:
        Zero-argument callable returning a fresh, unfitted sklearn-compatible
        estimator.  Called once per fold to guarantee independence.
    train_size:
        Number of rows in the initial training window.
    test_size:
        Number of rows in each test window.
    step_size:
        Number of rows to advance between folds.

    Returns
    -------
    List[FoldResult]
        One entry per completed fold, in chronological order.

    Raises
    ------
    ValueError
        If *df* does not contain a ``target`` column or any *feature_cols*.
    """
    missing_cols = [c for c in feature_cols + ["target"] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"walk_forward_validate: columns not found in df: {missing_cols}"
        )

    clean = df.dropna(subset=feature_cols + ["target"]).copy()
    if len(clean) < train_size + test_size:
        raise ValueError(
            f"walk_forward_validate: insufficient rows ({len(clean)}) for "
            f"train_size={train_size} + test_size={test_size}."
        )

    results: List[FoldResult] = []
    start = 0
    fold = 1

    while start + train_size + test_size <= len(clean):
        train = clean.iloc[start : start + train_size]
        test = clean.iloc[start + train_size : start + train_size + test_size]

        X_train = train[feature_cols]
        y_train = train["target"].astype(int)
        X_test = test[feature_cols]
        y_test = test["target"].astype(int)

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        probs: list = []
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                # Column 1 is P(class=1)
                probs = proba[:, 1].tolist()
            except Exception:  # noqa: BLE001
                probs = []

        results.append(
            FoldResult(
                fold=fold,
                train_start=str(train.index[0]),
                train_end=str(train.index[-1]),
                test_start=str(test.index[0]),
                test_end=str(test.index[-1]),
                metrics=classification_metrics(y_test, preds),
                probabilities=probs,
            )
        )

        fold += 1
        start += step_size

    return results


__all__ = ["FoldResult", "walk_forward_validate"]
