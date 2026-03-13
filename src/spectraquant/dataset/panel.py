"""Panel-based dataset assembly utilities."""
from __future__ import annotations

import pandas as pd


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
        return pd.DataFrame()

    close_panel = pd.DataFrame(closes).sort_index()
    ret_1d = close_panel.pct_change()
    ret_5d = close_panel.pct_change(5)
    sma_5 = close_panel.rolling(5, min_periods=3).mean()
    vol_5 = ret_1d.rolling(5, min_periods=3).std()
    label = (close_panel.pct_change(5).shift(-5) > 0).astype("float")

    panel = pd.concat(
        {
            "Close": close_panel,
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "sma_5": sma_5,
            "vol_5": vol_5,
            "label": label,
        },
        axis=1,
    )
    panel = panel.stack(level=1, future_stack=True).reset_index()
    panel.columns = ["date", "ticker", "Close", "ret_1d", "ret_5d", "sma_5", "vol_5", "label"]
    panel["date"] = pd.to_datetime(panel["date"], utc=True, errors="coerce")
    return panel.dropna(subset=["date"]).sort_values(["date", "ticker"])  # type: ignore[return-value]
