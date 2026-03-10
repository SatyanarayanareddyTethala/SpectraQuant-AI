"""Walk-forward evaluation utilities."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score, mean_absolute_error


def _generate_splits(dates: pd.Series, n_splits: int = 3) -> list[tuple[pd.Index, pd.Index]]:
    unique_dates = pd.Series(pd.to_datetime(dates, utc=True, errors="coerce")).dropna().sort_values().unique()
    if unique_dates.size < 10:
        return []

    total = len(unique_dates)
    test_size = max(int(total * 0.2), 1)
    train_end = max(int(total * 0.6), 1)
    splits = []
    for _ in range(n_splits):
        test_end = min(train_end + test_size, total)
        train_dates = unique_dates[:train_end]
        test_dates = unique_dates[train_end:test_end]
        if len(test_dates) == 0:
            break
        splits.append((pd.Index(train_dates), pd.Index(test_dates)))
        train_end = test_end
        if train_end >= total:
            break
    return splits


def _trading_metrics(actual_returns: pd.Series, predicted_direction: pd.Series) -> dict[str, float]:
    aligned = actual_returns.align(predicted_direction, join="inner")
    actual = aligned[0]
    direction = aligned[1]
    if actual.empty:
        return {"sharpe": 0.0, "hit_rate": 0.0}
    strat_returns = actual * direction
    vol = strat_returns.std() * np.sqrt(252)
    mean_return = strat_returns.mean() * 252
    sharpe = mean_return / vol if vol not in (0, np.nan) else 0.0
    hit_rate = float((actual > 0).eq(direction > 0).mean())
    return {"sharpe": float(sharpe), "hit_rate": hit_rate}


def walk_forward_evaluate(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    cls_target: str,
    reg_target: str,
    horizon: int,
) -> dict[str, Any]:
    """Evaluate simple baselines in a walk-forward fashion."""

    if df.empty:
        return {}

    splits = _generate_splits(df["date"])
    if not splits:
        return {}

    metrics = {"classification": [], "regression": []}

    for train_dates, test_dates in splits:
        train_mask = df["date"].isin(train_dates)
        test_mask = df["date"].isin(test_dates)

        train = df.loc[train_mask]
        test = df.loc[test_mask]
        if train.empty or test.empty:
            continue

        X_train = train[feature_cols]
        X_test = test[feature_cols]
        y_train_cls = train[cls_target].astype(int)
        y_test_cls = test[cls_target].astype(int)
        y_train_reg = train[reg_target]
        y_test_reg = test[reg_target]

        cls_model = LogisticRegression(max_iter=1000)
        cls_model.fit(X_train, y_train_cls)
        cls_pred = cls_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test_cls, cls_pred) if len(y_test_cls.unique()) > 1 else np.nan
        logloss = log_loss(y_test_cls, cls_pred, labels=[0, 1])
        direction = pd.Series(np.where(cls_pred >= 0.5, 1, -1), index=y_test_reg.index)
        cls_trade = _trading_metrics(y_test_reg, direction)

        reg_model = Ridge(alpha=1.0)
        reg_model.fit(X_train, y_train_reg)
        reg_pred = reg_model.predict(X_test)
        mae = mean_absolute_error(y_test_reg, reg_pred)
        reg_direction = pd.Series(np.where(reg_pred >= 0, 1, -1), index=y_test_reg.index)
        reg_trade = _trading_metrics(y_test_reg, reg_direction)
        directional_accuracy = float((np.sign(reg_pred) == np.sign(y_test_reg)).mean())

        metrics["classification"].append(
            {
                "auc": float(auc) if auc == auc else None,
                "logloss": float(logloss),
                "sharpe": cls_trade["sharpe"],
                "hit_rate": cls_trade["hit_rate"],
            }
        )
        metrics["regression"].append(
            {
                "mae": float(mae),
                "directional_accuracy": directional_accuracy,
                "sharpe": reg_trade["sharpe"],
                "hit_rate": reg_trade["hit_rate"],
            }
        )

    return {"horizon": horizon, **metrics}


__all__ = ["walk_forward_evaluate"]
