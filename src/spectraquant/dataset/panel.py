"""Panel-based dataset assembly utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


PANEL_REQUIRED_COLUMNS = ("date", "ticker", "Close", "ret_1d", "ret_5d", "sma_5", "vol_5", "rsi_14", "label")


def _compute_panel_rsi(close_panel: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close_panel.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    return rsi.ffill().fillna(50.0)


def build_price_feature_panel(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a vectorized (date, ticker) panel from normalized close series."""

    closes: dict[str, pd.Series] = {}
    for ticker, df in price_data.items():
        if "close" in df.columns:
            series = pd.to_numeric(df["close"], errors="coerce")
        elif "Close" in df.columns:
            series = pd.to_numeric(df["Close"], errors="coerce")
        else:
            continue
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get("date"), utc=True, errors="coerce")
        series.index = pd.to_datetime(idx, utc=True, errors="coerce")
        closes[ticker] = series.sort_index()

    if not closes:
        return pd.DataFrame(columns=list(PANEL_REQUIRED_COLUMNS))

    close_panel = pd.DataFrame(closes).sort_index()
    ret_1d = close_panel.pct_change()
    ret_5d = close_panel.pct_change(5)
    sma_5 = close_panel.rolling(5, min_periods=3).mean()
    vol_5 = ret_1d.rolling(5, min_periods=3).std()
    rsi_14 = _compute_panel_rsi(close_panel)
    fwd_5d = close_panel.pct_change(5).shift(-5)
    label = pd.DataFrame(np.where(fwd_5d.notna(), (fwd_5d > 0).astype(float), np.nan), index=fwd_5d.index, columns=fwd_5d.columns)

    panel = pd.concat(
        {
            "Close": close_panel,
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "sma_5": sma_5,
            "vol_5": vol_5,
            "rsi_14": rsi_14,
            "label": label,
        },
        axis=1,
    )
    # pandas>=2.1 supports future_stack; keep compatibility with older versions.
    try:
        panel = panel.stack(level=1, future_stack=True).reset_index()
    except TypeError:
        panel = panel.stack(level=1, dropna=False).reset_index()
    panel.columns = list(PANEL_REQUIRED_COLUMNS)
    panel["date"] = pd.to_datetime(panel["date"], utc=True, errors="coerce")
    return panel.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)
