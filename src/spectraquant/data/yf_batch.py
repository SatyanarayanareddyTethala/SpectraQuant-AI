"""Batched yfinance helpers with caching and retention."""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from spectraquant.core.providers.base import get_provider
from spectraquant.data.normalize import assert_price_frame, normalize_price_columns, normalize_price_frame
from spectraquant.data.retention import prune_dataframe_to_last_n_years

logger = logging.getLogger(__name__)

PRICES_DIR = Path("data/prices")
RAW_DATA_DIR = Path("data/raw")


def _safe_write_price(ticker: str, df: pd.DataFrame) -> None:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PRICES_DIR / f"{ticker}.csv"
    parquet_path = PRICES_DIR / f"{ticker}.parquet"
    raw_csv_path = RAW_DATA_DIR / f"{ticker}.csv"

    assert_price_frame(df, context=f"write {ticker}")
    df.to_csv(csv_path, index=True)
    df.to_csv(raw_csv_path, index=True)
    try:
        df.to_parquet(parquet_path, index=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save Parquet for %s: %s", ticker, exc)


def _merge_existing(ticker: str, fresh: pd.DataFrame, retention_years: int) -> pd.DataFrame:
    parquet_path = PRICES_DIR / f"{ticker}.parquet"
    if not parquet_path.exists():
        return prune_dataframe_to_last_n_years(fresh, retention_years, date_column="date")

    try:
        existing = pd.read_parquet(parquet_path)
    except Exception:  # noqa: BLE001
        existing = pd.DataFrame()

    existing = normalize_price_columns(existing, ticker) if not existing.empty else existing
    fresh = normalize_price_columns(fresh, ticker)
    existing = normalize_price_frame(existing) if not existing.empty else existing
    fresh = normalize_price_frame(fresh)
    combined = pd.concat([existing, fresh], ignore_index=False)
    combined = normalize_price_columns(combined, ticker)
    combined = normalize_price_frame(combined)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    combined = prune_dataframe_to_last_n_years(combined, retention_years, date_column="date")
    return combined


def _compute_backoff(attempt: int, base: float = 0.5, cap: float = 10.0) -> float:
    jitter = random.uniform(0, 0.75)
    return min(cap, base * (2 ** (attempt - 1)) + jitter)


def fetch_history_batched(
    tickers: Iterable[str],
    period: str = "5y",
    interval: str = "1d",
    batch_size: int = 50,
    sleep_seconds: int = 3,
    max_retries: int = 3,
    retention_years: int = 5,
    provider_name: str = "yfinance",
    config: dict | None = None,
) -> None:
    tickers_list = [t for t in tickers if t]
    if not tickers_list:
        logger.warning("No tickers provided for batch download.")
        return

    provider_cls = get_provider(provider_name)
    if provider_name == "mock":
        provider = provider_cls({})
    else:
        try:
            provider = provider_cls(config=config)
        except TypeError:
            provider = provider_cls()

    total_batches = (len(tickers_list) + batch_size - 1) // batch_size
    total_ok = 0
    total_failed = 0
    total_rows_written = 0

    for batch_idx, start in enumerate(range(0, len(tickers_list), batch_size), start=1):
        batch = tickers_list[start : start + batch_size]
        logger.info(
            "Downloading batch %s/%s (%s tickers, interval=%s, period=%s)",
            batch_idx,
            total_batches,
            len(batch),
            interval,
            period,
        )

        batch_start = time.time()
        batch_ok = 0
        batch_failed = 0

        for ticker in batch:
            attempts = 0
            success = False
            while attempts < max_retries:
                attempts += 1
                try:
                    df = provider.fetch_daily(ticker, period=period, interval=interval)
                    if df is None or df.empty:
                        logger.warning("Empty download for %s (attempt %s)", ticker, attempts)
                        time.sleep(_compute_backoff(attempts))
                        continue
                    df = _merge_existing(ticker, df, retention_years)
                    if df.empty:
                        logger.warning("No retained data for %s after retention filter.", ticker)
                        break
                    assert_price_frame(df, context=f"post-merge {ticker}")
                    _safe_write_price(ticker, df)
                    rows_written = len(df)
                    logger.debug("Saved %s rows for %s after retention.", rows_written, ticker)
                    total_rows_written += rows_written
                    success = True
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Download failed for %s attempt %s/%s: %s", ticker, attempts, max_retries, exc)
                    time.sleep(_compute_backoff(attempts))
            else:
                logger.error("Exceeded max retries for %s; skipping.", ticker)
            if success:
                batch_ok += 1
            else:
                batch_failed += 1

        total_ok += batch_ok
        total_failed += batch_failed
        elapsed = time.time() - batch_start
        logger.info(
            "Batch %s complete: ok=%s failed=%s elapsed=%.2fs",
            batch_idx,
            batch_ok,
            batch_failed,
            elapsed,
        )
        if start + batch_size < len(tickers_list):
            logger.info("Sleeping %s seconds between batches.", sleep_seconds)
            time.sleep(sleep_seconds)

    logger.info(
        "Download complete: %s symbols processed (%s ok, %s failed, %s rows written)",
        len(tickers_list),
        total_ok,
        total_failed,
        total_rows_written,
    )
