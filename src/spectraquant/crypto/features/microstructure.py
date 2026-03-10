"""Crypto microstructure feature engineering utilities."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}

# Rolling window sizes expressed in number of bars.
# Intended for minute-level data: 5m, 1h, 4h, 1d.
_VOL_WINDOWS = {"5m": 5, "1h": 60, "4h": 240, "1d": 1440}


def _to_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype(float)


# ------------------------------------------------------------------
# Spread features
# ------------------------------------------------------------------


def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute bid-ask spread, relative spread, and spread volatility.

    Requires *bid* and *ask* columns.  If they are absent the function
    returns an empty DataFrame with the original index so that downstream
    concatenation still works.
    """
    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]

    if not {"bid", "ask"}.issubset(frame.columns):
        logger.info("bid/ask columns missing – skipping spread features")
        return pd.DataFrame(index=frame.index)

    bid = _to_numeric(frame["bid"])
    ask = _to_numeric(frame["ask"])

    spread = ask - bid
    mid = (ask + bid) / 2
    mid_safe = mid.replace(0, np.nan)

    relative_spread = spread / mid_safe
    spread_vol = spread.rolling(20).std()

    return pd.DataFrame(
        {
            "spread": spread,
            "relative_spread": relative_spread,
            "spread_volatility": spread_vol,
        },
        index=frame.index,
    )


# ------------------------------------------------------------------
# Order imbalance features
# ------------------------------------------------------------------


def compute_order_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute buy/sell volume ratio, order-flow imbalance, and VWAP deviation.

    Requires *buy_volume* and *sell_volume* for the first two features.
    Requires *close* and *volume* for VWAP deviation.
    Missing columns are skipped gracefully.
    """
    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]

    features: dict[str, pd.Series] = {}

    has_trade_sides = {"buy_volume", "sell_volume"}.issubset(frame.columns)
    if has_trade_sides:
        buy_vol = _to_numeric(frame["buy_volume"])
        sell_vol = _to_numeric(frame["sell_volume"])
        total = buy_vol + sell_vol
        total_safe = total.replace(0, np.nan)

        features["buy_sell_ratio"] = buy_vol / sell_vol.replace(0, np.nan)
        features["order_flow_imbalance"] = (buy_vol - sell_vol) / total_safe
    else:
        logger.info("buy_volume/sell_volume missing – skipping order-flow features")

    if {"close", "volume"}.issubset(frame.columns):
        close = _to_numeric(frame["close"])
        volume = _to_numeric(frame["volume"])
        # Typical-price VWAP proxy over a 20-bar rolling window
        cum_pv = (close * volume).rolling(20).sum()
        cum_v = volume.rolling(20).sum()
        vwap = cum_pv / cum_v.replace(0, np.nan)
        features["vwap_deviation"] = (close - vwap) / vwap.replace(0, np.nan)

    if not features:
        return pd.DataFrame(index=frame.index)

    return pd.DataFrame(features, index=frame.index)


# ------------------------------------------------------------------
# Volatility features
# ------------------------------------------------------------------


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a suite of volatility estimators.

    * Parkinson volatility
    * Garman-Klass volatility
    * Realized volatility at multiple windows (5m, 1h, 4h, 1d)
    * Volatility-of-volatility (vol-of-vol)

    Requires *open*, *high*, *low*, *close* columns.
    """
    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]

    missing = OHLCV_COLUMNS - set(frame.columns)
    if missing:
        logger.warning("Missing columns %s – skipping volatility features", sorted(missing))
        return pd.DataFrame(index=frame.index)

    o = _to_numeric(frame["open"])
    h = _to_numeric(frame["high"])
    lo = _to_numeric(frame["low"])
    c = _to_numeric(frame["close"])

    log_hl = np.log(h / lo.replace(0, np.nan))
    log_co = np.log(c / o.replace(0, np.nan))

    # Parkinson volatility (20-bar rolling)
    parkinson = np.sqrt(
        (1 / (4 * np.log(2))) * (log_hl ** 2).rolling(20).mean()
    )

    # Garman-Klass volatility (20-bar rolling)
    log_ho = np.log(h / o.replace(0, np.nan))
    log_lo = np.log(lo / o.replace(0, np.nan))
    log_c_o = log_co
    gk_term = 0.5 * (log_ho - log_lo) ** 2 - (2 * np.log(2) - 1) * log_c_o ** 2
    garman_klass = np.sqrt(gk_term.rolling(20).mean().clip(lower=0))

    # Realized volatility at multiple windows
    log_returns = np.log(c / c.shift(1))
    features: dict[str, pd.Series] = {
        "parkinson_vol": parkinson,
        "garman_klass_vol": garman_klass,
    }

    for label, window in _VOL_WINDOWS.items():
        features[f"realized_vol_{label}"] = log_returns.rolling(window).std()

    # Volatility-of-volatility (rolling std of the 1h realized vol)
    rv_1h = features["realized_vol_1h"]
    features["vol_of_vol"] = rv_1h.rolling(60).std()

    return pd.DataFrame(features, index=frame.index)


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all microstructure features and return a single DataFrame."""
    logger.info("Computing microstructure features (%d rows)", len(df))

    parts = [
        compute_spread_features(df),
        compute_order_imbalance(df),
        compute_volatility_features(df),
    ]
    result = pd.concat(parts, axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


__all__ = [
    "compute_spread_features",
    "compute_order_imbalance",
    "compute_volatility_features",
    "compute_microstructure_features",
]
