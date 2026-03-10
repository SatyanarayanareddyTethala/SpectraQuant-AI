"""Alpha factor computation utilities for SpectraQuant."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping, MutableMapping

import numpy as np
import pandas as pd

from spectraquant.data.normalize import normalize_price_columns, normalize_price_frame

logger = logging.getLogger(__name__)


def _select_price_series(price_df: pd.DataFrame) -> pd.Series:
    """Pick a representative price series from the input dataframe."""

    for col in ("close", "adj_close", "price"):
        if col in price_df.columns:
            return price_df[col]

    for col in price_df.columns:
        if isinstance(col, str) and "close" in col:
            return price_df[col]

    numeric_cols = price_df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        return price_df[numeric_cols[0]]

    raise ValueError("Price dataframe does not contain a numeric price column.")


def _rolling_momentum(price: pd.Series, window: int) -> pd.Series:
    return price.pct_change(periods=window)


def _normalized_returns(returns: pd.Series, window: int = 20) -> pd.Series:
    rolling_std = returns.rolling(window=window, min_periods=window // 2).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = returns / rolling_std
    return normalized.replace([np.inf, -np.inf], np.nan)


def _downside_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    downside = returns.copy()
    downside[downside > 0] = 0
    return downside.rolling(window=window, min_periods=window // 2).std()


def _sma_trend(price: pd.Series, window: int) -> pd.Series:
    sma = price.rolling(window=window, min_periods=window // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        trend = price / sma - 1
    return trend.replace([np.inf, -np.inf], np.nan)


def _ema_slope(price: pd.Series, span: int = 20, lag: int = 5) -> pd.Series:
    ema = price.ewm(span=span, adjust=False, min_periods=lag).mean()
    return ema.diff(periods=lag) / lag


def _broadcast_fundamental(
    index: pd.Index, fundamentals: Mapping[str, float] | None, keys: Iterable[str], label: str
) -> pd.Series | None:
    if fundamentals is None:
        logger.info("No fundamentals provided; skipping %s factor.", label)
        return None

    value = None
    for key in keys:
        if key in fundamentals and fundamentals[key] is not None:
            value = fundamentals[key]
            break

    if value is None:
        logger.info("Fundamental value for %s not available; skipping.", label)
        return None

    return pd.Series(value, index=index)


def compute_alpha_factors(
    price_df: pd.DataFrame,
    fundamentals: Dict[str, float] | None,
    config: Dict,
) -> pd.DataFrame:
    """Compute multi-factor alpha features for a single asset."""

    price_df = normalize_price_columns(price_df)
    price_df = normalize_price_frame(price_df)
    price_series = _select_price_series(price_df)
    returns = price_series.pct_change()

    factors: MutableMapping[str, pd.Series] = {}

    for window in (20, 60, 120):
        factors[f"momentum_{window}d"] = _rolling_momentum(price_series, window)
    factors["normalized_return"] = _normalized_returns(returns)

    for window in (20, 60):
        factors[f"volatility_{window}d"] = returns.rolling(
            window=window, min_periods=window // 2
        ).std()
    factors["downside_volatility"] = _downside_volatility(returns)

    for window in (20, 50, 200):
        factors[f"trend_sma_{window}"] = _sma_trend(price_series, window)
    factors["ema_slope"] = _ema_slope(price_series)

    value_keys: Mapping[str, Iterable[str]] = {
        "value_pe": ("pe_ratio", "trailingPE"),
        "value_pb": ("pb_ratio", "priceToBook"),
        "value_roe": ("roe", "returnOnEquity"),
        "value_debt_to_equity": ("debt_to_equity", "debtToEquity"),
    }

    for label, keys in value_keys.items():
        series = _broadcast_fundamental(price_series.index, fundamentals, keys, label)
        if series is not None:
            factors[label] = series

    factor_df = pd.DataFrame(factors)
    factor_df = factor_df.add_prefix("alpha_")
    factor_df = factor_df.ffill()
    if factor_df.isna().all().all():
        factor_df = factor_df.fillna(0.0)
    else:
        factor_df = factor_df.dropna(how="all").fillna(0.0)
    return factor_df


__all__ = ["compute_alpha_factors"]
