"""ML feature engineering for SpectraQuant.

Extends the existing OHLCV feature set with additional technical indicators
and optional sentiment integration required for supervised classification.

The output columns are designed to complement (not duplicate) the 22-column
OHLCV feature set in ``spectraquant.features.ohlcv_features``.  Callers who
already have those base features can safely pass the full OHLCV dataframe;
this module adds the ML-specific columns on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ML-specific feature columns produced by add_features()
ML_FEATURE_COLS: list[str] = [
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "volatility_5",
    "volatility_10",
    "volume_change",
    "sma_10_ratio",
    "sma_20_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "sentiment_score",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ML classification features to an OHLCV dataframe.

    Parameters
    ----------
    df:
        DataFrame with at minimum ``close`` and ``volume`` columns.
        Optional ``sentiment_score`` column is forwarded unchanged;
        if absent it is filled with 0.0.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with all :data:`ML_FEATURE_COLS` columns appended.
        Rows where the new columns contain NaN / Inf are **retained** so that
        callers can dropna() according to their own policy.  The raw OHLCV
        columns are preserved.

    Raises
    ------
    ValueError
        If ``close`` or ``volume`` columns are missing.
    """
    required = {"close", "volume"}
    missing = required - set(str(c).lower() for c in df.columns)
    if missing:
        raise ValueError(f"add_features: missing required columns: {sorted(missing)}")

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    volume = pd.to_numeric(out["volume"], errors="coerce").astype(float)

    # --- Returns -------------------------------------------------------
    out["return_1"] = close.pct_change(1)
    out["return_3"] = close.pct_change(3)
    out["return_5"] = close.pct_change(5)
    out["return_10"] = close.pct_change(10)

    # --- Volatility ----------------------------------------------------
    daily_ret = close.pct_change()
    out["volatility_5"] = daily_ret.rolling(5).std()
    out["volatility_10"] = daily_ret.rolling(10).std()

    # --- Volume --------------------------------------------------------
    out["volume_change"] = volume.pct_change()

    # --- Moving-average ratios -----------------------------------------
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    out["sma_10_ratio"] = close / sma_10
    out["sma_20_ratio"] = close / sma_20

    # --- RSI-14 --------------------------------------------------------
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD ----------------------------------------------------------
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # --- Sentiment (optional) ------------------------------------------
    if "sentiment_score" not in out.columns:
        out["sentiment_score"] = 0.0

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


__all__ = ["ML_FEATURE_COLS", "add_features"]
