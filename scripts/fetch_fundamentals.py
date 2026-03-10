#!/usr/bin/env python
"""Fetch fundamentals for configured tickers and cache to CSV/JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import yfinance as yf

from spectraquant.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FUND_DIR = ROOT / "data" / "fundamentals"
FUND_DIR.mkdir(parents=True, exist_ok=True)


def _extract_numeric(payload: Mapping, keys: Iterable[str]):
    for key in keys:
        if key in payload and payload[key] is not None:
            val = pd.to_numeric(payload[key], errors="coerce")
            if pd.notna(val):
                return float(val)
    return None


def fetch_for_ticker(ticker: str) -> dict | None:
    ticker_obj = yf.Ticker(ticker)
    payloads = []
    for accessor in ("fast_info", "info"):
        try:
            value = getattr(ticker_obj, accessor)
            if callable(value):
                value = value()
            if isinstance(value, Mapping):
                payloads.append(value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch %s for %s: %s", accessor, ticker, exc)
    fields = {
        "pe": ("pe_ratio", "trailingPE", "forwardPE"),
        "pb": ("pb_ratio", "priceToBook"),
        "roe": ("roe", "returnOnEquity"),
        "debt_to_equity": ("debt_to_equity", "debtToEquity"),
    }
    out: dict[str, float] = {}
    for dest, keys in fields.items():
        for payload in payloads:
            val = _extract_numeric(payload, keys)
            if val is not None:
                out[dest] = val
                break
    if out:
        cache = FUND_DIR / f"{ticker}.json"
        with cache.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info("Cached fundamentals for %s", ticker)
        return out
    logger.info("No fundamentals for %s", ticker)
    return None


def resolve_tickers(cfg: dict) -> list[str]:
    universe = cfg.get("universe", {})
    tickers = universe.get("tickers") or cfg.get("data", {}).get("tickers") or []
    if not tickers:
        tickers = []
        for key in ("india", "uk"):
            maybe = universe.get(key)
            if isinstance(maybe, list):
                tickers.extend(maybe)
    return [str(t).upper() for t in tickers]


def main():
    cfg = get_config()
    tickers = resolve_tickers(cfg)
    logger.info("Fetching fundamentals for %d tickers", len(tickers))
    rows = []
    for ticker in tickers:
        payload = fetch_for_ticker(ticker)
        if not payload:
            continue
        payload["ticker"] = ticker
        rows.append(payload)
    if rows:
        df = pd.DataFrame(rows)
        csv_path = FUND_DIR / "fundamentals_latest.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved fundamentals CSV to %s", csv_path)
    else:
        logger.warning("No fundamentals fetched")


if __name__ == "__main__":
    main()
