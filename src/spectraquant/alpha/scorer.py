"""Alpha scoring utilities for SpectraQuant."""
from __future__ import annotations

import logging
from typing import Dict, Mapping

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "momentum": 0.35,
    "trend": 0.25,
    "volatility": 0.20,
    "value": 0.20,
}


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean()
    std = df.std(ddof=0)
    z = (df - mean) / std.replace({0: pd.NA})
    return z


def compute_alpha_score(alpha_df: pd.DataFrame, config: Dict) -> pd.Series:
    """Compute an aggregate alpha score from factor data."""

    if alpha_df.empty:
        raise ValueError("alpha_df is empty; cannot compute alpha score")

    weights: Mapping[str, float] = (
        config.get("alpha", {}).get("weights", {}) if config is not None else {}
    )
    weights = {**DEFAULT_WEIGHTS, **weights}

    z_scores = _zscore(alpha_df)

    group_columns = {
        "momentum": [c for c in z_scores.columns if "momentum" in c or "normalized_return" in c],
        "trend": [c for c in z_scores.columns if "trend" in c or "ema" in c],
        "volatility": [c for c in z_scores.columns if "volatility" in c],
        "value": [c for c in z_scores.columns if "value_" in c],
    }

    combined = pd.Series(0.0, index=alpha_df.index)
    total_weight = 0.0

    for group, cols in group_columns.items():
        if not cols:
            logger.info("No columns found for alpha group '%s'; skipping.", group)
            continue

        group_score = z_scores[cols].mean(axis=1)
        weight = float(weights.get(group, 0))
        if weight == 0:
            logger.info("Weight for group '%s' is zero; skipping.", group)
            continue

        combined = combined.add(group_score * weight, fill_value=0)
        total_weight += weight

    if total_weight == 0:
        raise ValueError("Total weight is zero; cannot compute alpha score")

    return combined / total_weight


def compute_factor_contributions(alpha_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Compute normalized factor group contributions that sum to 1 per row."""

    if alpha_df.empty:
        raise ValueError("alpha_df is empty; cannot compute factor contributions")

    weights: Mapping[str, float] = (
        config.get("alpha", {}).get("weights", {}) if config is not None else {}
    )
    weights = {**DEFAULT_WEIGHTS, **weights}

    z_scores = _zscore(alpha_df)

    group_columns = {
        "momentum": [c for c in z_scores.columns if "momentum" in c or "normalized_return" in c],
        "trend": [c for c in z_scores.columns if "trend" in c or "ema" in c],
        "volatility": [c for c in z_scores.columns if "volatility" in c],
        "value": [c for c in z_scores.columns if "value_" in c],
    }

    contributions = pd.DataFrame(index=alpha_df.index)
    for group, cols in group_columns.items():
        if not cols:
            continue
        weight = float(weights.get(group, 0))
        if weight == 0:
            continue
        contributions[group] = z_scores[cols].mean(axis=1) * weight

    if contributions.empty:
        raise ValueError("No factor contributions could be computed")

    contributions = contributions.abs()
    total = contributions.sum(axis=1)
    # Handle division by zero without pd.NA to avoid FutureWarning
    normalized = contributions.div(total.where(total != 0, 1.0), axis=0)
    zero_total = total == 0
    if zero_total.any():
        equal_weight = 1 / len(contributions.columns)
        normalized.loc[zero_total, :] = equal_weight
    return normalized


__all__ = ["compute_alpha_score", "compute_factor_contributions"]
