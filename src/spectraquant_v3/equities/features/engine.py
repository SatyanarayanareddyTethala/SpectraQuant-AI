"""Equity feature engine for SpectraQuant-AI-V3.

Computes technical indicators from OHLCV DataFrames for equity symbols.
All features are computed in-memory and returned as new columns appended
to the input DataFrame.

Features computed:
- ``ret_1d``        – 1-day log return
- ``ret_Nd``        – N-day cumulative log return (momentum)
- ``rsi``           – Wilder's RSI
- ``volume_ratio``  – today's volume / rolling mean volume
- ``atr_norm``      – ATR normalised by close price
- ``vol_realised``  – rolling annualised realised volatility

This module must never import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spectraquant_v3.core.errors import DataSchemaError, EmptyPriceDataError


def _validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> None:
    label = f" for '{symbol}'" if symbol else ""
    if df.empty:
        raise EmptyPriceDataError(
            f"Feature engine{label}: DataFrame is empty."
        )
    required = {"open", "high", "low", "close", "volume"}
    cols = {c.lower() for c in df.columns}
    missing = required - cols
    if missing:
        raise DataSchemaError(
            f"Feature engine{label}: missing columns {sorted(missing)}."
        )


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's exponential moving average."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_features(
    df: pd.DataFrame,
    symbol: str = "",
    momentum_window: int = 20,
    rsi_period: int = 14,
    vol_window: int = 20,
    volume_ma_window: int = 20,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Compute equity technical features and return an enriched DataFrame.

    Args:
        df:                Input OHLCV DataFrame (DatetimeIndex recommended).
        symbol:            Symbol name used in error messages.
        momentum_window:   Lookback period for momentum (N-day return).
        rsi_period:        RSI period (Wilder's method).
        vol_window:        Rolling window for realised volatility.
        volume_ma_window:  Rolling window for volume moving average.
        atr_window:        ATR smoothing window.

    Returns:
        Copy of *df* with additional feature columns.

    Raises:
        DataSchemaError:    If required OHLCV columns are missing.
        EmptyPriceDataError: If *df* is empty.
    """
    _validate_ohlcv(df, symbol)

    out = df.copy()
    out.columns = [c.lower() for c in out.columns]

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    log_ret = np.log(close / close.shift(1))
    out["ret_1d"] = log_ret

    out[f"ret_{momentum_window}d"] = np.log(
        close / close.shift(momentum_window)
    )

    out["rsi"] = _rsi(close, period=rsi_period)

    vol_ma = volume.rolling(window=volume_ma_window, min_periods=1).mean()
    out["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_window, min_periods=1).mean()
    out["atr_norm"] = atr / close.replace(0, np.nan)

    out["vol_realised"] = log_ret.rolling(window=vol_window, min_periods=2).std() * (
        252 ** 0.5
    )

    return out


class EquityFeatureEngine:
    """Stateless feature engine for the equity pipeline.

    Args:
        momentum_window:  Lookback for momentum return.
        rsi_period:       RSI smoothing period.
        vol_window:       Realised volatility window.
        volume_ma_window: Volume moving-average window.
        atr_window:       ATR window.
    """

    def __init__(
        self,
        momentum_window: int = 20,
        rsi_period: int = 14,
        vol_window: int = 20,
        volume_ma_window: int = 20,
        atr_window: int = 14,
    ) -> None:
        self.momentum_window = momentum_window
        self.rsi_period = rsi_period
        self.vol_window = vol_window
        self.volume_ma_window = volume_ma_window
        self.atr_window = atr_window

    @classmethod
    def from_config(cls, cfg: dict) -> "EquityFeatureEngine":
        """Build from merged equity config."""
        signals_cfg = cfg.get("equities", {}).get("signals", {})
        return cls(
            momentum_window=int(signals_cfg.get("momentum_lookback", 20)),
            rsi_period=int(signals_cfg.get("rsi_period", 14)),
        )

    def transform(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """Compute and return features for *df*."""
        return compute_features(
            df,
            symbol=symbol,
            momentum_window=self.momentum_window,
            rsi_period=self.rsi_period,
            vol_window=self.vol_window,
            volume_ma_window=self.volume_ma_window,
            atr_window=self.atr_window,
        )

    def transform_many(
        self, price_map: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Compute features for multiple symbols."""
        result: dict[str, pd.DataFrame] = {}
        for sym, df in price_map.items():
            try:
                result[sym] = self.transform(df, symbol=sym)
            except (DataSchemaError, EmptyPriceDataError):
                continue
        return result
