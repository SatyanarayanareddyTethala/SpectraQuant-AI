"""Feature importance extraction and persistence for SpectraQuant ML.

Supports any tree-based sklearn-compatible estimator that exposes
``feature_importances_``.  Results are persisted to the repo's existing
``reports/`` output hierarchy so they appear alongside other evaluation
artefacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


_IMPORTANCE_DIR = Path("reports/ml/importance")


def get_feature_importance(
    model: Any,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Extract and return feature importances as a ranked DataFrame.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible estimator with ``feature_importances_``.
    feature_cols:
        Ordered list of feature column names used during training.

    Returns
    -------
    pd.DataFrame
        Columns ``feature`` and ``importance``, sorted descending.

    Raises
    ------
    ValueError
        If the model does not expose ``feature_importances_``.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"get_feature_importance: model {type(model).__name__!r} does not "
            "expose feature_importances_.  Only tree-based estimators are "
            "supported (RandomForest, XGBoost, LightGBM, etc.)."
        )
    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance(
    importance_df: pd.DataFrame,
    label: str = "",
) -> Path:
    """Persist feature importance to the reports directory.

    Parameters
    ----------
    importance_df:
        Output of :func:`get_feature_importance`.
    label:
        Optional tag appended to the filename (e.g. ``"rf"`` or ``"xgb"``).

    Returns
    -------
    Path
        Absolute path of the written CSV file.
    """
    _IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    path = _IMPORTANCE_DIR / f"feature_importance{suffix}_{ts}.csv"
    importance_df.to_csv(path, index=False)
    return path


__all__ = ["get_feature_importance", "save_feature_importance"]
