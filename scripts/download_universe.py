#!/usr/bin/env python
"""Download exchange universe constituents and normalize to a standard schema."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen
import json

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


UNIVERSE_SOURCES = {
    "nifty_50": {
        "exchange": "NSE",
        "url": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
    },
    "nifty_100": {
        "exchange": "NSE",
        "url": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100",
    },
    "nifty_200": {
        "exchange": "NSE",
        "url": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200",
    },
    "nifty_500": {
        "exchange": "NSE",
        "url": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
    },
    "ftse_100": {
        "exchange": "LSE",
        "url": "https://www.ftserussell.com/sites/default/files/ftse_uk_index_series_constituents.csv",
        "filter": "FTSE 100",
    },
    "ftse_250": {
        "exchange": "LSE",
        "url": "https://www.ftserussell.com/sites/default/files/ftse_uk_index_series_constituents.csv",
        "filter": "FTSE 250",
    },
    "ftse_all_share": {
        "exchange": "LSE",
        "url": "https://www.ftserussell.com/sites/default/files/ftse_uk_index_series_constituents.csv",
        "filter": "FTSE All-Share",
    },
}


def _fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as response:  # noqa: S310 - intentional public download
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _fetch_csv(url: str) -> pd.DataFrame:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as response:  # noqa: S310 - intentional public download
        return pd.read_csv(response)


def _normalize_universe(df: pd.DataFrame, exchange: str) -> pd.DataFrame:
    lower_cols = {c.lower(): c for c in df.columns}
    ticker_col = None
    for candidate in ("symbol", "ticker", "code", "epic", "instrument"):
        if candidate in lower_cols:
            ticker_col = lower_cols[candidate]
            break
    if ticker_col is None:
        raise ValueError("Unable to locate ticker column in universe data")
    name_col = lower_cols.get("name") or lower_cols.get("company name") or lower_cols.get("security")
    normalized = pd.DataFrame(
        {
            "ticker": df[ticker_col].astype(str).str.strip(),
            "exchange": exchange,
            "name": df[name_col].astype(str).str.strip() if name_col else "",
        }
    )
    normalized = normalized.dropna(subset=["ticker"])
    normalized = normalized[normalized["ticker"] != ""]
    return normalized


def _save_universe(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    logger.info("Saved %s tickers to %s", len(df), path)
    return path


def _download_sources(keys: Iterable[str], out_dir: Path) -> None:
    for key in keys:
        source = UNIVERSE_SOURCES.get(key)
        if source is None:
            raise ValueError(f"Unknown universe key: {key}")
        exchange = source["exchange"]
        url = source["url"]
        logger.info("Downloading %s from %s", key, url)
        if exchange == "NSE":
            payload = _fetch_json(url)
            data = payload.get("data") or payload.get("records", {}).get("data")
            if not data:
                raise ValueError(f"No data payload returned for {key}")
            raw = pd.DataFrame(data)
        else:
            raw = _fetch_csv(url)
            filter_label = source.get("filter")
            if filter_label:
                candidates = [c for c in raw.columns if "index" in c.lower()]
                if candidates:
                    raw = raw[raw[candidates[0]].astype(str).str.contains(filter_label, na=False)]
        normalized = _normalize_universe(raw, exchange=exchange)
        _save_universe(normalized, out_dir, key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official index universe CSVs.")
    parser.add_argument(
        "--out-dir",
        default="data/universe",
        help="Output directory for normalized universe files",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help=f"Universe keys to download (default: {', '.join(UNIVERSE_SOURCES)})",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    keys = args.only if args.only else list(UNIVERSE_SOURCES.keys())
    _download_sources(keys, out_dir)


if __name__ == "__main__":
    main()
