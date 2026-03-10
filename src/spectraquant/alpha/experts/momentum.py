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
        ret20 = close.iloc[-1] / close.iloc[-20] - 1
        vol = close.pct_change().tail(20).std() or 1.0
        score = float(ret20 / max(vol, 1e-6) * 0.05)
        for h in horizons:
            rows.append(_frame_row(t, date, h, score, "momentum"))
    return pd.DataFrame(rows)
