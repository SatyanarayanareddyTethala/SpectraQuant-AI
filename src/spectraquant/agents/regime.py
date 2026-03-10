"""Market regime detection for crypto assets."""
from __future__ import annotations

import enum
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CryptoRegime(enum.Enum):
    """Broad crypto market regime classification."""

    BULL = "BULL"
    BEAR = "BEAR"
    RANGE = "RANGE"
    HIGH_VOL = "HIGH_VOL"


def detect_regime(
    df: pd.DataFrame,
    vol_window: int = 20,
    trend_window: int = 50,
) -> CryptoRegime:
    """Classify the current market regime from *close* prices.

    Logic
    -----
    * Compute realised volatility (rolling std of returns) and SMA.
    * If vol above its rolling median → ``HIGH_VOL``.
    * Else if close > SMA → ``BULL``.
    * Else if close < SMA → ``BEAR``.
    * Otherwise → ``RANGE``.

    Falls back to :func:`spectraquant.regime.simple_regime.compute_regime`
    when available and returns a compatible mapping.
    """
    if "close" not in df.columns or df.empty:
        logger.warning("detect_regime: 'close' column missing or empty df")
        return CryptoRegime.RANGE

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    if close.dropna().empty:
        return CryptoRegime.RANGE

    vol = close.pct_change().rolling(vol_window).std()
    vol_median = vol.rolling(min(len(vol), 252), min_periods=vol_window).median()

    sma = close.rolling(trend_window, min_periods=1).mean()

    latest_close = close.iloc[-1]
    latest_vol = vol.iloc[-1]
    latest_vol_median = vol_median.iloc[-1]
    latest_sma = sma.iloc[-1]

    if np.isnan(latest_vol) or np.isnan(latest_vol_median):
        return CryptoRegime.RANGE

    if latest_vol > latest_vol_median:
        return CryptoRegime.HIGH_VOL

    if latest_close > latest_sma:
        return CryptoRegime.BULL

    if latest_close < latest_sma:
        return CryptoRegime.BEAR

    return CryptoRegime.RANGE


__all__ = ["CryptoRegime", "detect_regime"]
