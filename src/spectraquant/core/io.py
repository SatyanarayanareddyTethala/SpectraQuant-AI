"""Validated IO helpers for pipeline artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from spectraquant.core.schema import (
    order_columns,
    schema_version_for,
    validate_portfolio_results,
    validate_predictions,
    validate_signals,
)
from spectraquant.core.time import ensure_datetime_column, normalize_time_index


def write_predictions(df: pd.DataFrame, path: Path) -> Path:
    df = df.copy()
    df["schema_version"] = schema_version_for("predictions")
    validated = validate_predictions(df)
    validated = order_columns(validated, "predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(path, index=False)
    return path


def write_signals(df: pd.DataFrame, path: Path) -> Path:
    df = df.copy()
    df["schema_version"] = schema_version_for("signals")
    validated = validate_signals(df)
    validated = normalize_time_index(validated.set_index("date", drop=False), context="signals write")
    validated = validated.reset_index(drop=True)
    validated = order_columns(validated, "signals")
    path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(path, index=False)
    return path


def write_portfolio(
    returns: pd.Series,
    weights: pd.DataFrame,
    metrics: Dict,
    returns_path: Path,
    weights_path: Path,
    metrics_path: Path,
) -> None:
    validate_portfolio_results({"returns": returns, "metrics": metrics})
    returns_path.parent.mkdir(parents=True, exist_ok=True)
    returns_df = returns.to_frame("return").reset_index().rename(columns={"index": "date"})
    returns_df = ensure_datetime_column(returns_df, "date")
    returns_df["schema_version"] = schema_version_for("portfolio_returns")
    weights_df = weights.reset_index().rename(columns={"index": "date"})
    weights_df = ensure_datetime_column(weights_df, "date")
    weights_df["schema_version"] = schema_version_for("portfolio_weights")
    returns_df = order_columns(returns_df, "portfolio_returns")
    weights_df = order_columns(weights_df, "portfolio_weights")
    returns_df.to_csv(returns_path, index=False)
    weights_df.to_csv(weights_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
