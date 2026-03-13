"""Model factories for SpectraQuant ML.

Provides consistent, reproducible model constructors for Random Forest and
XGBoost classifiers, plus a unified metrics helper used across walk-forward
folds.

XGBoost is optional: if the package is not installed the ``get_xgboost``
factory raises ``ImportError`` with a clear message rather than failing
silently at a later point.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from xgboost import XGBClassifier  # type: ignore[import]

    HAS_XGB: bool = True
except Exception:  # noqa: BLE001
    HAS_XGB = False


def get_random_forest() -> RandomForestClassifier:
    """Return a reproducible RandomForestClassifier instance.

    Hyper-parameters are chosen to balance expressiveness against
    over-fitting on typical daily-bar equity data.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


def get_xgboost() -> Any:
    """Return a reproducible XGBClassifier instance.

    Raises
    ------
    ImportError
        If the ``xgboost`` package is not installed in the current environment.
    """
    if not HAS_XGB:
        raise ImportError(
            "xgboost is not installed.  Install it with:\n"
            "    pip install xgboost\n"
            "or add it to the project's optional 'ml' extras."
        )
    return XGBClassifier(  # type: ignore[call-arg]
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )


def classification_metrics(
    y_true: "np.ndarray | pd.Series",
    y_pred: "np.ndarray | pd.Series",
) -> dict[str, float]:
    """Compute standard binary-classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0 / 1).
    y_pred:
        Predicted labels (0 / 1).

    Returns
    -------
    dict with keys ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


__all__ = [
    "HAS_XGB",
    "get_random_forest",
    "get_xgboost",
    "classification_metrics",
]
