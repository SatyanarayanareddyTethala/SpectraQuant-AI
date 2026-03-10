"""Snapshot portfolio utilities."""
from __future__ import annotations

import pandas as pd
import numpy as np

from spectraquant.data.normalize import normalize_price_frame


def select_top_k(signals: pd.DataFrame, k: int) -> pd.DataFrame:
    """Select top-K ranked snapshot entries."""
    if signals.empty:
        raise ValueError("Signals snapshot is empty")
    if "rank" not in signals.columns:
        raise ValueError("Signals must include rank column")
    return signals.sort_values("rank").head(k).copy()


def apply_risk_constraints(
    weights: pd.Series,
    price_data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: dict,
) -> pd.Series:
    portfolio_cfg = config.get("portfolio", {})
    max_weight = portfolio_cfg.get("max_asset_weight")
    if max_weight is not None:
        weights = weights.clip(upper=float(max_weight))
        weights = weights / weights.sum()

    volatility_target = portfolio_cfg.get("volatility_target")
    if volatility_target:
        target = float(volatility_target)
        vols = {}
        for ticker, df in price_data.items():
            if df is None or df.empty:
                continue
            df = normalize_price_frame(df)
            series = df.sort_index()["close"].pct_change().dropna()
            if series.empty:
                continue
            vols[ticker] = series.std()
        if vols:
            port_vol = sum(weights.get(t, 0) * vols.get(t, 0) for t in weights.index)
            if port_vol > 0:
                scale = target / port_vol
                weights = weights * scale
                weights = weights / weights.sum()

    min_weight = portfolio_cfg.get("min_weight_threshold")
    if min_weight is not None:
        min_weight_val = float(min_weight)
        weights = weights[weights >= min_weight_val]
        if not weights.empty:
            weights = weights / weights.sum()

    return weights


def validate_weight_matrix(weights: pd.DataFrame, config: dict) -> None:
    portfolio_cfg = config.get("portfolio", {})
    max_weight = portfolio_cfg.get("max_asset_weight")
    min_weight = portfolio_cfg.get("min_weight_threshold")
    top_k = int(portfolio_cfg.get("top_k", weights.shape[1] if weights is not None else 0))

    if weights.empty:
        raise ValueError("Portfolio weights are empty")

    sums = weights.sum(axis=1)
    invested = sums > 0
    if invested.any() and not np.allclose(sums[invested], 1.0, atol=1e-6):
        raise ValueError("Portfolio weights must sum to 1 when invested")

    if max_weight is not None and (weights > float(max_weight)).any().any():
        raise ValueError("Portfolio weights exceed max_asset_weight")

    if min_weight is not None and (weights[(weights > 0) & (weights < float(min_weight))].any().any()):
        raise ValueError("Portfolio weights below min_weight_threshold")

    non_zero = (weights > 0).sum(axis=1)
    if (non_zero > top_k).any():
        raise ValueError("Portfolio holds more than top_k positions")
