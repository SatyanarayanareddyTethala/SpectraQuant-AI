#!/usr/bin/env python
"""Fetch intraday 1m data with safe normalization, fallback, and retention."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from spectraquant.config import get_config
from spectraquant.core.time import ensure_datetime_column
from spectraquant.data.normalize import normalize_price_columns
from spectraquant.core.providers.base import get_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
INTRADAY_DIR = ROOT / "data" / "intraday_1m"
INTRADAY_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_intraday(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = normalize_price_columns(df, ticker)
    df = df.rename_axis("date").reset_index() if "date" not in df.columns else df
    df = ensure_datetime_column(df, "date")
    return df


def fetch_intraday(
    ticker: str,
    retention_days: int = 7,
    interval: str = "1m",
    provider_name: str = "yfinance",
) -> Path:
    path = INTRADAY_DIR / f"{ticker}.parquet"
    existing = None
    if path.exists():
        existing = pd.read_parquet(path)
    last_ts = None
    if existing is not None and not existing.empty:
        last_ts = pd.to_datetime(existing["date"]).max()

    fetched = None
    provider_cls = get_provider(provider_name)
    provider = provider_cls() if provider_name != "mock" else provider_cls({})

    for period in ("7d", "1d"):
        try:
            df = provider.fetch_intraday(ticker, period=period, interval=interval)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed intraday fetch for %s (%s): %s", ticker, period, exc)
            continue
        if df is None or df.empty:
            logger.warning("No intraday rows for %s with period %s", ticker, period)
            continue
        fetched = _normalize_intraday(df.reset_index(), ticker)
        break

    if fetched is None or fetched.empty:
        logger.warning("No intraday data fetched for %s", ticker)
        return path

    if last_ts is not None:
        fetched = fetched[fetched["date"] > last_ts]
    combined = existing if existing is not None else pd.DataFrame()
    if combined is not None and not combined.empty:
        combined = pd.concat([combined, fetched], ignore_index=True)
    else:
        combined = fetched

    combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    combined = combined[combined["date"] >= cutoff]

    tmp_path = path.with_suffix(".tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)
    logger.info("Saved %s rows for %s to %s (retention=%sd)", len(combined), ticker, path, retention_days)
    return path


def main():
    cfg = get_config()
    intraday_cfg = cfg.get("intraday", {})
    tickers = intraday_cfg.get("tickers") or cfg.get("data", {}).get("tickers", [])
    if not tickers:
        raise ValueError("Intraday tickers must be explicitly configured")
    retention = int(intraday_cfg.get("retention_days", 7))
    interval = intraday_cfg.get("interval", "1m")
    provider_name = cfg.get("data", {}).get("provider", "yfinance")
    for ticker in tickers:
        fetch_intraday(ticker, retention_days=retention, interval=interval, provider_name=provider_name)


if __name__ == "__main__":
    main()
