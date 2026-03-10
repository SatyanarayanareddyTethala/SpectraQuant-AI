from __future__ import annotations

import pandas as pd


def _latest_date(prices: dict[str, pd.DataFrame]) -> pd.Timestamp:
    dates = []
    for df in prices.values():
        if "date" in df.columns:
            s = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
            if not s.empty:
                dates.append(s.max())
        elif isinstance(df.index, pd.DatetimeIndex) and len(df.index):
            dates.append(df.index.max())
    return max(dates) if dates else pd.Timestamp.utcnow().tz_localize("UTC")


def _frame_row(ticker: str, date: pd.Timestamp, horizon: str, score: float, expert: str) -> dict:
    signal = "BUY" if score > 0.1 else "SELL" if score < -0.1 else "HOLD"
    conf = min(max(abs(score), 0.0), 1.0)
    return {
        "ticker": ticker,
        "date": date,
        "horizon": horizon,
        "score": float(score),
        "signal": signal,
        "confidence": float(conf),
        "expected_return": float(score * 0.02),
        "risk_estimate": float(max(1e-6, 1.0 - conf)),
        "expert_name": expert,
        "schema_version": 1,
    }
