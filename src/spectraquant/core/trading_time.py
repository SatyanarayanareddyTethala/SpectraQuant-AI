"""Trading time utilities for NSE and LSE."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pandas as pd


_SESSIONS = {
    "NSE": {"start": time(3, 45), "end": time(10, 0)},
    "LSE": {"start": time(8, 0), "end": time(16, 30)},
}


def ensure_utc_tzaware(series_or_df, col: str = "date"):
    if isinstance(series_or_df, pd.Series):
        return pd.to_datetime(series_or_df, utc=True, errors="coerce")
    if isinstance(series_or_df, pd.DataFrame):
        df = series_or_df.copy()
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df.index.name = col
        else:
            raise KeyError(f"Missing {col} column or datetime index")
        return df
    raise TypeError("Expected pandas Series or DataFrame")


def align_to_interval(ts_utc: pd.Timestamp, interval: str) -> pd.Timestamp:
    if interval.endswith("m"):
        minutes = int(interval.replace("m", ""))
        return ts_utc.floor(f"{minutes}min")
    if interval.endswith("d"):
        days = int(interval.replace("d", ""))
        return ts_utc.floor(f"{days}D")
    raise ValueError(f"Unsupported interval: {interval}")


def is_market_open(exchange: str, ts_utc: pd.Timestamp) -> bool:
    session = _SESSIONS.get(exchange)
    if session is None:
        raise ValueError(f"Unknown exchange: {exchange}")
    local_time = ts_utc.time()
    return session["start"] <= local_time <= session["end"]


def expected_latest_time(exchange: str, interval: str, now_utc: pd.Timestamp) -> pd.Timestamp:
    session = _SESSIONS.get(exchange)
    if session is None:
        raise ValueError(f"Unknown exchange: {exchange}")
    session_end = datetime.combine(now_utc.date(), session["end"], tzinfo=timezone.utc)
    if now_utc <= session_end:
        return align_to_interval(now_utc, interval)
    return align_to_interval(pd.Timestamp(session_end), interval)


def latest_valid_bar_time(exchange: str, df: pd.DataFrame, interval: str) -> pd.Timestamp:
    df = ensure_utc_tzaware(df, "date")
    if "date" in df.columns:
        latest = df["date"].max()
    elif isinstance(df.index, pd.DatetimeIndex):
        latest = df.index.max()
    else:
        raise KeyError("Missing date column and datetime index")
    return align_to_interval(latest, interval)


def is_stale(
    exchange: str,
    interval: str,
    latest_ts: pd.Timestamp,
    now_utc: pd.Timestamp,
    tolerance_minutes: int,
) -> bool:
    expected = expected_latest_time(exchange, interval, now_utc)
    tolerance = timedelta(minutes=tolerance_minutes)
    return latest_ts < expected - tolerance
