"""Simple market regime detection."""
from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_LABELS = (
    "LOW_VOL_TREND",
    "LOW_VOL_CHOP",
    "HIGH_VOL_TREND",
    "HIGH_VOL_CHOP",
)


def _rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    def _percentile(x: pd.Series) -> float:
        if x.empty:
            return np.nan
        return float((x.rank(pct=True).iloc[-1]))

    return series.rolling(window, min_periods=20).apply(_percentile, raw=False)


def compute_regime(df: pd.DataFrame) -> pd.Series:
    """Compute a simple volatility/trend regime label."""

    if df.empty:
        raise ValueError("Input dataframe is empty")

    frame = df.copy()
    if "close" not in frame.columns:
        raise ValueError("Input dataframe must include close column")

    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    if "vol_20" in frame.columns:
        vol_20 = pd.to_numeric(frame["vol_20"], errors="coerce").astype(float)
    else:
        vol_20 = close.pct_change().rolling(20).std()

    sma_50 = frame["sma_50"] if "sma_50" in frame.columns else close.rolling(50).mean()
    sma_50 = pd.to_numeric(sma_50, errors="coerce").astype(float)
    slope = sma_50.diff(5)

    vol_pct = _rolling_percentile(vol_20)
    vol_state = np.where(vol_pct <= 0.5, "LOW", "HIGH")
    trend_state = (close > sma_50) & (slope > 0)

    regime = pd.Series(index=frame.index, dtype="object")
    regime.loc[(vol_state == "LOW") & trend_state] = "LOW_VOL_TREND"
    regime.loc[(vol_state == "LOW") & (~trend_state)] = "LOW_VOL_CHOP"
    regime.loc[(vol_state == "HIGH") & trend_state] = "HIGH_VOL_TREND"
    regime.loc[(vol_state == "HIGH") & (~trend_state)] = "HIGH_VOL_CHOP"

    regime = regime.fillna("LOW_VOL_CHOP")
    return regime


__all__ = ["compute_regime", "REGIME_LABELS"]
