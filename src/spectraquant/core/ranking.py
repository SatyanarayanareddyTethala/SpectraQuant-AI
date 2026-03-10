"""Ranking utilities for snapshot scoring."""
from __future__ import annotations

import pandas as pd


def normalize_scores(scores: pd.Series) -> pd.Series:
    """Normalize scores to 0-100 scale."""
    normalized = pd.to_numeric(scores, errors="coerce")
    normalized = normalized.apply(lambda v: v * 100 if pd.notna(v) and v <= 1 else v)
    normalized = normalized.fillna(50.0).clip(lower=0, upper=100)
    return normalized


def add_rank(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Add rank column based on descending scores."""
    ranked = df.sort_values(score_col, ascending=False).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def normalize_scores_multi(scores_by_horizon: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Normalize scores per horizon to 0-100 scale."""
    return {horizon: normalize_scores(series) for horizon, series in scores_by_horizon.items()}
