"""Compute standardized on-chain features (z-scores, rolling stats).

All outputs are keyed by a UTC ``asof_utc`` DatetimeIndex and a ``symbol``
column so they can be joined directly with price-based feature frames.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROLLING_WINDOW = 20


def compute_onchain_features(
    raw_df: pd.DataFrame,
    *,
    window: int = _ROLLING_WINDOW,
) -> pd.DataFrame:
    """Derive z-score and rolling features from raw on-chain data.

    Parameters
    ----------
    raw_df:
        Raw collector output with a UTC DatetimeIndex named ``asof_utc``
        and a ``symbol`` column.  All other numeric columns are treated
        as input metrics.
    window:
        Look-back window for rolling mean / std (default 20).

    Returns
    -------
    pd.DataFrame
        Feature frame indexed by ``asof_utc`` with ``symbol`` column and
        derived columns: ``{col}_zscore``, ``{col}_rmean``, ``{col}_rstd``
        for every numeric input column.
    """
    if raw_df.empty:
        logger.warning("Empty raw DataFrame — returning empty features.")
        return pd.DataFrame()

    # Ensure UTC DatetimeIndex
    if raw_df.index.name != "asof_utc":
        if "asof_utc" in raw_df.columns:
            raw_df = raw_df.set_index("asof_utc")
        else:
            logger.warning("No 'asof_utc' index or column found.")
            return pd.DataFrame()

    if not isinstance(raw_df.index, pd.DatetimeIndex):
        raw_df.index = pd.to_datetime(raw_df.index, utc=True)
    elif raw_df.index.tz is None:
        raw_df.index = raw_df.index.tz_localize("UTC")

    has_symbol = "symbol" in raw_df.columns
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        logger.warning("No numeric columns in raw DataFrame.")
        return raw_df[["symbol"]].copy() if has_symbol else pd.DataFrame()

    groups = raw_df.groupby("symbol") if has_symbol else [(None, raw_df)]
    parts: list[pd.DataFrame] = []

    for sym, grp in groups:
        feat = pd.DataFrame(index=grp.index)
        if has_symbol:
            feat["symbol"] = sym

        for col in numeric_cols:
            series = grp[col].astype(float)
            rmean = series.rolling(window=window, min_periods=1).mean()
            rstd = series.rolling(window=window, min_periods=1).std().fillna(0)

            feat[f"{col}_rmean"] = rmean
            feat[f"{col}_rstd"] = rstd

            # Z-score: zero variance → z-score of 0 (no deviation from mean)
            safe_std = rstd.replace(0, np.nan)
            feat[f"{col}_zscore"] = ((series - rmean) / safe_std).fillna(0)

        parts.append(feat)

    result = pd.concat(parts)
    result.index.name = "asof_utc"
    logger.info(
        "Computed %d on-chain features for %d rows.",
        len([c for c in result.columns if c != "symbol"]),
        len(result),
    )
    return result
