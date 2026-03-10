"""Lightweight universe utilities for NSE-only canonical universe management."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Optional

import pandas as pd
import requests

NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
CANONICAL_COLUMNS = ["exchange", "symbol", "ticker", "name", "isin", "currency", "source", "asof_date"]
_SYMBOL_RE = re.compile(r"^[A-Z0-9&._-]+$")


def parse_nse_equity_list(raw_df: pd.DataFrame, *, asof_date: str | None = None) -> tuple[pd.DataFrame, int]:
    """Parse NSE EQUITY_L dataframe into canonical universe schema."""

    cols = {str(c).strip().upper(): c for c in raw_df.columns}
    symbol_col = cols.get("SYMBOL")
    if symbol_col is None:
        raise ValueError("EQUITY_L is missing SYMBOL column")

    name_col = cols.get("NAME OF COMPANY") or cols.get("NAME") or cols.get("COMPANY NAME")
    isin_col = cols.get("ISIN NUMBER") or cols.get("ISIN")

    out = pd.DataFrame()
    out["symbol"] = raw_df[symbol_col].astype(str).str.strip().str.upper()
    out = out[out["symbol"] != ""]
    out = out[out["symbol"].str.match(_SYMBOL_RE, na=False)]
    out["exchange"] = "NSE"
    out["ticker"] = out["symbol"] + ".NS"
    out["name"] = raw_df.loc[out.index, name_col].astype(str).str.strip() if name_col else ""
    out["isin"] = raw_df.loc[out.index, isin_col].astype(str).str.strip() if isin_col else ""
    out["currency"] = "INR"
    out["source"] = "NSE:EQUITY_L"
    out["asof_date"] = asof_date or date.today().isoformat()

    before = len(out)
    out = out.drop_duplicates(subset=["symbol"], keep="first").copy()
    dedup_removed = before - len(out)

    out = out[out["ticker"].str.endswith(".NS", na=False)]
    out = out[CANONICAL_COLUMNS].reset_index(drop=True)
    return out, dedup_removed


def update_nse_universe(
    *,
    raw_path: str | Path = "data/universe/raw/EQUITY_L.csv",
    output_path: str | Path = "data/universe/universe_nse.csv",
    url: str = NSE_EQUITY_URL,
) -> tuple[pd.DataFrame, int]:
    """Download latest NSE EQUITY_L and write canonical NSE universe."""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    raw_path = Path(raw_path)
    output_path = Path(output_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path.write_bytes(response.content)
    raw_df = pd.read_csv(raw_path)
    parsed, dedup_removed = parse_nse_equity_list(raw_df)
    parsed.to_csv(output_path, index=False)
    return parsed, dedup_removed


def load_universe(path: str | Path) -> pd.DataFrame:
    """Load canonical universe CSV and validate required schema."""

    df = pd.read_csv(path)
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Universe file missing required columns: {missing}")
    return df.copy()


def get_symbols(df: pd.DataFrame) -> list[str]:
    return [s for s in df["symbol"].astype(str).str.strip().tolist() if s]


def map_symbol_to_ticker(df: pd.DataFrame, symbol: str) -> Optional[str]:
    if not symbol:
        return None
    s = symbol.strip().upper()
    match = df[df["symbol"].astype(str).str.upper() == s]
    if match.empty:
        return None
    ticker = str(match.iloc[0]["ticker"]).strip()
    return ticker or None
