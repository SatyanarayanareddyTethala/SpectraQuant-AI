from __future__ import annotations
import pandas as pd
from ._base import _latest_date, _frame_row

def compute_expert_scores(prices, fundamentals, news_features, config):
    date = _latest_date(prices)
    horizons = config.get("predictions", {}).get("daily_horizons", ["1d"])
    rows = []
    for t in prices:
        f = fundamentals.get(t, {}) if isinstance(fundamentals, dict) else {}
        pe = float(f.get("pe_ratio", 20) or 20)
        pb = float(f.get("pb_ratio", 3) or 3)
        roe = float(f.get("roe", 0.1) or 0.1)
        score = float((roe * 2.0) - (pe / 50.0) - (pb / 10.0))
        for h in horizons:
            rows.append(_frame_row(t, date, h, score, "value"))
    return pd.DataFrame(rows)
