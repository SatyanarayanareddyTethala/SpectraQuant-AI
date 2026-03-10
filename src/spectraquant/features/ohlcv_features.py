"""OHLCV feature engineering utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def _to_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype(float)


def compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute OHLCV features for a single ticker dataframe."""

    if df.empty:
        raise ValueError("Input OHLCV dataframe is empty")

    frame = df.copy()
    frame.columns = [str(col).lower() for col in frame.columns]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    open_price = _to_numeric(frame["open"])
    high = _to_numeric(frame["high"])
    low = _to_numeric(frame["low"])
    close = _to_numeric(frame["close"])
    volume = _to_numeric(frame["volume"])

    returns = close.pct_change()
    ret_5d = close.pct_change(5)
    ret_20d = close.pct_change(20)
    logret_1d = np.log(close / close.shift(1))
    roc_10 = close.pct_change(10)
    momentum_20d = close - close.shift(20)

    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    close_sma20 = close / sma_20 - 1
    sma10_sma50 = sma_10 / sma_50 - 1

    vol_20 = returns.rolling(20).std()
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    hl_range = (high - low) / close

    std_20 = close.rolling(20).std()
    z_20 = (close - sma_20) / std_20
    bollinger_width = (4 * std_20) / sma_20

    body = (close - open_price) / close
    upper_wick = (high - np.maximum(open_price, close)) / close
    lower_wick = (np.minimum(open_price, close) - low) / close
    gap = (open_price - close.shift(1)) / close.shift(1)

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    vol_z_20 = (volume - vol_mean) / vol_std
    volume_change_1d = volume.pct_change()

    feature_df = pd.DataFrame(
        {
            "ret_1d": returns,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "logret_1d": logret_1d,
            "roc_10": roc_10,
            "momentum_20d": momentum_20d,
            "sma_10": sma_10,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "close_sma20": close_sma20,
            "sma10_sma50": sma10_sma50,
            "vol_20": vol_20,
            "atr_14": atr_14,
            "hl_range": hl_range,
            "z_20": z_20,
            "bollinger_width": bollinger_width,
            "body": body,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "gap": gap,
            "vol_z_20": vol_z_20,
            "volume_change_1d": volume_change_1d,
        },
        index=frame.index,
    )

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.dropna()
    return feature_df


__all__ = ["compute_ohlcv_features"]
