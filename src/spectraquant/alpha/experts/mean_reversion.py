from __future__ import annotations
import pandas as pd
from ._base import _latest_date, _frame_row

def compute_expert_scores(prices, fundamentals, news_features, config):
    date = _latest_date(prices)
    horizons = config.get("predictions", {}).get("daily_horizons", ["1d"])
    rows = []
    for t, df in prices.items():
        close = pd.to_numeric(df.get("close", df.get("Close")), errors="coerce").dropna()
        if len(close) < 20:
            continue
        mean = close.tail(20).mean(); std = close.tail(20).std() or 1.0
        z = (close.iloc[-1] - mean) / std
        for h in horizons:
            rows.append(_frame_row(t, date, h, float(-z * 0.1), "mean_reversion"))
    return pd.DataFrame(rows)
