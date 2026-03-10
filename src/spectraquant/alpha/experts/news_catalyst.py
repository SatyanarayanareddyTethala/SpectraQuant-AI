from __future__ import annotations
import pandas as pd
from ._base import _latest_date, _frame_row

def compute_expert_scores(prices, fundamentals, news_features, config):
    date = _latest_date(prices)
    horizons = config.get("predictions", {}).get("daily_horizons", ["1d"])
    nf = news_features if isinstance(news_features, pd.DataFrame) else pd.DataFrame(columns=["ticker","score","mentions"])
    nf = nf.set_index("ticker") if not nf.empty and "ticker" in nf.columns else pd.DataFrame()
    rows = []
    for t in prices:
        nscore = float(nf.loc[t, "score"]) if not nf.empty and t in nf.index and "score" in nf.columns else 0.0
        mentions = float(nf.loc[t, "mentions"]) if not nf.empty and t in nf.index and "mentions" in nf.columns else 0.0
        score = float(max(min(nscore * 0.2 + mentions * 0.01, 1), -1))
        for h in horizons:
            rows.append(_frame_row(t, date, h, score, "news_catalyst"))
    return pd.DataFrame(rows)
