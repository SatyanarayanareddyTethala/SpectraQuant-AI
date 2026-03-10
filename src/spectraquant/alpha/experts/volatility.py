from __future__ import annotations
import pandas as pd
from ._base import _latest_date, _frame_row

def compute_expert_scores(prices, fundamentals, news_features, config):
    date = _latest_date(prices)
    horizons = config.get("predictions", {}).get("daily_horizons", ["1d"])
    rows = []
    for t, df in prices.items():
        close = pd.to_numeric(df.get("close", df.get("Close")), errors="coerce").dropna()
        if len(close) < 25:
            continue
        vol = close.pct_change().tail(20).std()
        score = float(0.2 - (vol if pd.notna(vol) else 0.2))
        for h in horizons:
            rows.append(_frame_row(t, date, h, score, "volatility"))
    return pd.DataFrame(rows)
