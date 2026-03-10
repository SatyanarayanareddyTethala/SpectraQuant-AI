"""Evaluation utilities for predictions and portfolio artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from spectraquant.core.schema import validate_predictions, validate_signals
from spectraquant.core.time import ensure_datetime_column


def evaluate_predictions(pred_df: pd.DataFrame, dataset: pd.DataFrame) -> Dict[str, float]:
    pred_df = validate_predictions(pred_df)
    dataset = ensure_datetime_column(dataset, "date")
    if "label" not in dataset.columns:
        raise ValueError("Dataset missing label column for evaluation")
    merged = pred_df.merge(dataset[["ticker", "date", "label"]], on=["ticker", "date"], how="inner")
    if merged.empty:
        raise ValueError("No overlap between predictions and dataset labels")
    scores = merged["score"] / 100.0
    labels = merged["label"].astype(int)
    predictions = (scores >= scores.median()).astype(int)
    accuracy = float((predictions == labels).mean())
    brier = float(np.mean((scores - labels) ** 2))
    buy_bucket = merged[scores >= scores.quantile(0.8)]
    hit_rate = float((buy_bucket["label"] == 1).mean()) if not buy_bucket.empty else 0.0
    return {"accuracy": accuracy, "brier_score": brier, "hit_rate_buy_bucket": hit_rate}


def evaluate_signals(signals_df: pd.DataFrame) -> Dict[str, float]:
    signals_df = validate_signals(signals_df)
    counts = signals_df["signal"].value_counts(normalize=True).to_dict()
    return {f"signal_share_{k.lower()}": float(v) for k, v in counts.items()}


def evaluate_portfolio(returns_df: pd.DataFrame, weights_df: pd.DataFrame) -> Dict[str, float]:
    returns_df = ensure_datetime_column(returns_df, "date")
    weights_df = ensure_datetime_column(weights_df, "date")
    numeric_weights = weights_df.drop(columns=["date"], errors="ignore").select_dtypes(include="number")
    turnover = float(numeric_weights.diff().abs().sum(axis=1).mean()) if not numeric_weights.empty else 0.0
    concentration = float(numeric_weights.max(axis=1).mean()) if not numeric_weights.empty else 0.0
    exposure_count = float((numeric_weights > 0).sum(axis=1).mean()) if not numeric_weights.empty else 0.0
    return {
        "turnover": turnover,
        "concentration": concentration,
        "exposure_count": exposure_count,
    }


def evaluate_tx_cost_sensitivity(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    cost_bps_grid: Iterable[float],
    output_path: Path,
) -> tuple[pd.DataFrame, float | None]:
    returns_df = ensure_datetime_column(returns_df, "date")
    weights_df = ensure_datetime_column(weights_df, "date")
    returns = returns_df.set_index("date")["return"].astype(float)
    weights = weights_df.set_index("date").drop(columns=["schema_version"], errors="ignore")
    numeric_weights = weights.select_dtypes(include="number").fillna(0.0)
    turnover = numeric_weights.diff().abs().sum(axis=1).fillna(0.0)

    results = []
    break_even_bps = None
    for cost_bps in cost_bps_grid:
        cost_rate = float(cost_bps) / 10000.0
        adjusted_returns = returns - (turnover * cost_rate)
        cumulative = (1 + adjusted_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0
        vol = adjusted_returns.std() * np.sqrt(252)
        mean_return = adjusted_returns.mean() * 252
        sharpe = mean_return / vol if vol not in (0, np.nan) else 0.0
        hit_rate = float((adjusted_returns > 0).mean()) if not adjusted_returns.empty else 0.0
        cumulative_return = float(cumulative.iloc[-1] - 1) if not cumulative.empty else 0.0
        results.append(
            {
                "transaction_cost_bps": float(cost_bps),
                "cumulative_return": cumulative_return,
                "sharpe_ratio": float(sharpe) if sharpe == sharpe else 0.0,
                "max_drawdown": float(max_drawdown) if max_drawdown == max_drawdown else 0.0,
                "hit_rate": hit_rate,
            }
        )
        if cumulative_return >= 0 and break_even_bps is None:
            break_even_bps = float(cost_bps)

    df = pd.DataFrame(results)
    if break_even_bps is not None:
        df["break_even_bps"] = break_even_bps
    else:
        df["break_even_bps"] = np.nan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df, break_even_bps


def evaluate_feature_drift(dataset: pd.DataFrame, output_path: Path) -> Path:
    dataset = ensure_datetime_column(dataset, "date")
    numeric = dataset.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
    if numeric.empty:
        raise ValueError("No numeric features available for drift evaluation")
    recent = numeric.tail(max(10, len(numeric) // 10))
    drift = ((recent.mean() - numeric.mean()) / (numeric.std() + 1e-9)).to_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(drift).to_json(output_path, indent=2)
    return output_path
