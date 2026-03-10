"""Regime Engine — classify current market state for NSE/India.

Computes regime labels from recent price/volume/breadth data and returns a
structured dict usable by downstream modules.

Labels
------
TRENDING      Sustained directional move with low-medium volatility.
CHOPPY        Range-bound, mean-reverting, elevated noise.
RISK_ON       Broad market advancing; high breadth, low VIX proxy.
RISK_OFF      Broad market declining; low breadth, high VIX proxy.
EVENT_DRIVEN  Price action dominated by a discrete event (earnings, policy).
PANIC         Extreme volatility, breadth collapse, high correlation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ordered label list (stable for encoding purposes)
REGIME_LABELS = (
    "TRENDING",
    "CHOPPY",
    "RISK_ON",
    "RISK_OFF",
    "EVENT_DRIVEN",
    "PANIC",
)

# Days to look back when computing SMA slope
_SLOPE_LOOKBACK_DAYS = 5

# Thresholds (tunable via config)
_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "panic_vol": 0.04,          # daily vol > 4 % → PANIC candidate
    "high_vol": 0.025,          # daily vol > 2.5 % → high-vol regime
    "trend_slope_pct": 0.001,   # SMA slope > 0.1 % per day → trending
    "risk_on_breadth": 0.55,    # advance/decline ratio > 0.55 → RISK_ON
    "risk_off_breadth": 0.45,   # advance/decline ratio < 0.45 → RISK_OFF
    "event_vol_spike": 1.8,     # vol_today / vol_20 > 1.8 → event-driven
}


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_regime_features(
    df: pd.DataFrame,
    window: int = 20,
) -> Dict[str, float]:
    """Extract regime-relevant features from a daily OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least a ``close`` column.  Optional columns:
        ``volume``, ``breadth`` (adv/dec ratio), ``vix_proxy``.
        Index should be a DatetimeIndex.
    window : int
        Lookback window for rolling statistics (default 20 trading days).

    Returns
    -------
    dict
        Feature dictionary with keys:
        ``daily_vol``, ``vol_ratio``, ``slope_pct``, ``close_vs_sma``,
        ``breadth``, ``vix_proxy``, ``trend_strength``.
    """
    if df.empty or "close" not in df.columns:
        raise ValueError("DataFrame must be non-empty and contain 'close'")

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(close) < 2:
        raise ValueError("Insufficient data rows for regime computation")

    # Use last `window` rows
    close_w = close.iloc[-window:] if len(close) > window else close

    # Daily returns and rolling volatility
    returns = close_w.pct_change().dropna()
    daily_vol = float(returns.std()) if len(returns) > 1 else 0.0

    # Vol ratio: today's absolute return vs rolling vol
    today_ret = abs(float(returns.iloc[-1])) if len(returns) >= 1 else 0.0
    vol_ratio = today_ret / max(daily_vol, 1e-9)

    # SMA slope as % change per day
    sma = close_w.rolling(min(window, len(close_w)), min_periods=5).mean()
    valid_sma = sma.dropna()
    if len(valid_sma) >= _SLOPE_LOOKBACK_DAYS:
        slope_pct = float(
            (valid_sma.iloc[-1] - valid_sma.iloc[-_SLOPE_LOOKBACK_DAYS])
            / (valid_sma.iloc[-_SLOPE_LOOKBACK_DAYS] + 1e-9)
            / _SLOPE_LOOKBACK_DAYS
        )
    else:
        slope_pct = 0.0

    # Close vs SMA (last value)
    close_vs_sma = float(
        (close_w.iloc[-1] - sma.iloc[-1]) / (sma.iloc[-1] + 1e-9)
        if not pd.isna(sma.iloc[-1])
        else 0.0
    )

    # Trend strength: fraction of days close > previous close
    trend_strength = float((returns > 0).mean()) if len(returns) > 0 else 0.5

    # Optional: breadth (advance/decline ratio)
    if "breadth" in df.columns:
        breadth = float(pd.to_numeric(df["breadth"], errors="coerce").iloc[-1])
        if np.isnan(breadth):
            breadth = 0.5
    else:
        breadth = trend_strength  # proxy

    # Optional: VIX proxy
    if "vix_proxy" in df.columns:
        vix_proxy = float(pd.to_numeric(df["vix_proxy"], errors="coerce").iloc[-1])
        if np.isnan(vix_proxy):
            vix_proxy = daily_vol * 100
    else:
        vix_proxy = daily_vol * 100  # annualise as percentage

    return {
        "daily_vol": daily_vol,
        "vol_ratio": vol_ratio,
        "slope_pct": slope_pct,
        "close_vs_sma": close_vs_sma,
        "breadth": breadth,
        "vix_proxy": vix_proxy,
        "trend_strength": trend_strength,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_regime(
    features: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[str, float]:
    """Map feature dict to a regime label + confidence.

    Parameters
    ----------
    features : dict
        Output of :func:`compute_regime_features`.
    thresholds : dict, optional
        Override default classification thresholds.

    Returns
    -------
    tuple[str, float]
        ``(label, confidence)`` where confidence is in [0, 1].
    """
    t = dict(_DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    vol = features.get("daily_vol", 0.0)
    vol_ratio = features.get("vol_ratio", 1.0)
    slope = features.get("slope_pct", 0.0)
    close_vs_sma = features.get("close_vs_sma", 0.0)
    breadth = features.get("breadth", 0.5)
    vix = features.get("vix_proxy", 0.0)
    trend_str = features.get("trend_strength", 0.5)

    # ---- PANIC -------------------------------------------------------
    if vol >= t["panic_vol"] and breadth < 0.3:
        conf = min(1.0, vol / (t["panic_vol"] + 1e-9) * 0.5)
        return "PANIC", round(conf, 3)

    # ---- RISK_OFF / RISK_ON (breadth-dominant) -----------------------
    if breadth < t["risk_off_breadth"] and vol >= t["high_vol"]:
        conf = 1.0 - breadth
        return "RISK_OFF", round(min(1.0, conf), 3)

    if breadth > t["risk_on_breadth"] and slope > 0:
        conf = breadth
        return "RISK_ON", round(min(1.0, conf), 3)

    # ---- EVENT_DRIVEN ------------------------------------------------
    if vol_ratio > t["event_vol_spike"]:
        conf = min(1.0, vol_ratio / t["event_vol_spike"] * 0.6)
        return "EVENT_DRIVEN", round(conf, 3)

    # ---- TRENDING -------------------------------------------------------
    if slope > t["trend_slope_pct"] and close_vs_sma > 0 and trend_str > 0.55:
        conf = min(1.0, trend_str + max(0.0, close_vs_sma) * 2)
        return "TRENDING", round(min(1.0, conf), 3)

    if slope < -t["trend_slope_pct"] and close_vs_sma < 0 and trend_str < 0.45:
        conf = min(1.0, (1.0 - trend_str) + abs(close_vs_sma) * 2)
        return "TRENDING", round(min(1.0, conf), 3)  # downward trend

    # ---- CHOPPY (default) -------------------------------------------
    conf = 1.0 - abs(slope) / (t["trend_slope_pct"] + 1e-9) * 0.3
    return "CHOPPY", round(max(0.1, min(1.0, conf)), 3)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def get_current_regime(
    config: Optional[Dict[str, Any]] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compute and return the current regime as a structured dict.

    Parameters
    ----------
    config : dict, optional
        May contain ``regime_window_days`` and ``thresholds`` keys.
    df : pd.DataFrame, optional
        Price dataframe.  When *None* a synthetic test series is used
        (useful for unit tests without live data).

    Returns
    -------
    dict
        Keys: ``label``, ``confidence``, ``features``, ``as_of``.
    """
    cfg = config or {}
    window = int(cfg.get("regime_window_days", 20))
    thresholds = cfg.get("thresholds", None)

    if df is None:
        # Synthetic fallback — keeps tests deterministic without live data
        rng = np.random.default_rng(42)
        n = max(window + 5, 30)
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        dates = pd.date_range(
            end=datetime.now(tz=timezone.utc).date(), periods=n, freq="B"
        )
        df = pd.DataFrame({"close": prices}, index=dates)

    features = compute_regime_features(df, window=window)
    label, confidence = classify_regime(features, thresholds=thresholds)

    result = {
        "label": label,
        "confidence": confidence,
        "features": features,
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
    }
    logger.info("Current regime: %s (conf=%.3f)", label, confidence)
    return result
