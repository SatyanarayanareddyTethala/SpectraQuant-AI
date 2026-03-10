from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LATEST_JSON = Path("data/news_cache/news_universe_latest.json")
LATEST_CSV  = Path("data/news_cache/news_candidates_latest.csv")

def write_latest_news_universe(candidates_df, source_csv_path: str | None = None) -> None:
    """
    Persist the latest news-driven universe selection for downstream stages.
    Writes:
      - data/news_cache/news_universe_latest.json
      - data/news_cache/news_candidates_latest.csv (optional convenience)
    """
    LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)

    tickers = (
        candidates_df.get("ticker")
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if "ticker" in candidates_df.columns else []
    )

    payload = {
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "source_candidates_csv": source_csv_path,
    }
    LATEST_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Optional: also keep a stable "latest" CSV for quick inspection
    try:
        candidates_df.to_csv(LATEST_CSV, index=False)
    except Exception:
        pass
