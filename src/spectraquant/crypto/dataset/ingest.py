"""Cryptocurrency dataset ingestion pipeline.

Loads ``cryptocurrency_dataset.csv``, normalises columns, builds:
  - ``asset_master.parquet``  – unique-by-symbol fundamentals/metadata
  - ``market_snapshot.parquet`` – append-only time-series with ``as_of``
  - ``symbol_map.parquet``    – canonical → yfinance/exchange symbol mapping

All timestamps are UTC-aware.  Runs are idempotent.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root; callers may override via config)
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIR = Path("data/crypto")
_ASSET_MASTER_FILE = "asset_master.parquet"
_MARKET_SNAPSHOT_FILE = "market_snapshot.parquet"
_SYMBOL_MAP_FILE = "symbol_map.parquet"

# ---------------------------------------------------------------------------
# Known yfinance symbol overrides (canonical → yfinance ticker)
# Users can extend via config: crypto_dataset.yfinance_overrides
# ---------------------------------------------------------------------------
_YFINANCE_OVERRIDES: dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "AVAX": "AVAX-USD",
    "DOT": "DOT-USD",
    "MATIC": "MATIC-USD",
    "LINK": "LINK-USD",
    "LTC": "LTC-USD",
    "BCH": "BCH-USD",
    "ALGO": "ALGO-USD",
    "ATOM": "ATOM-USD",
    "XLM": "XLM-USD",
    "VET": "VET-USD",
    "FIL": "FIL-USD",
    "TRX": "TRX-USD",
    "ETC": "ETC-USD",
    "NEAR": "NEAR-USD",
}

# ---------------------------------------------------------------------------
# Required and optional columns
# ---------------------------------------------------------------------------
_ASSET_MASTER_COLS = [
    "canonical_symbol",
    "name",
    "category",
    "network_type",
    "launch_date",
    "all_time_high_usd",
    "all_time_low_usd",
    "max_supply",
]

_MARKET_SNAPSHOT_COLS = [
    "canonical_symbol",
    "as_of",
    "price_usd",
    "market_cap_usd",
    "volume_24h_usd",
    "change_24h_pct",
    "circulating_supply",
    "community_rank",
]


# ---------------------------------------------------------------------------
# Column normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_col_name(col: str) -> str:
    """Lower-case, strip, replace spaces/special chars with underscores."""
    col = col.strip().lower()
    col = re.sub(r"[%$,\(\)]", "", col)
    col = re.sub(r"[\s\-/]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


_COL_ALIASES: dict[str, str] = {
    # name/symbol
    "coin_name": "name",
    "token_name": "name",
    "ticker": "canonical_symbol",
    "symbol": "canonical_symbol",
    # price
    "price": "price_usd",
    "current_price_usd": "price_usd",
    "price_usd_": "price_usd",
    # market cap
    "market_cap": "market_cap_usd",
    "market_capitalization_usd": "market_cap_usd",
    # volume
    "24h_trading_volume_usd": "volume_24h_usd",
    "volume_24h": "volume_24h_usd",
    "trading_volume_24h_usd": "volume_24h_usd",
    "24h_volume": "volume_24h_usd",
    # change
    "24h_change_percent": "change_24h_pct",
    "24h_change": "change_24h_pct",
    "percent_change_24h": "change_24h_pct",
    # supply
    "circulating_supply_coins": "circulating_supply",
    # ATH / ATL
    "all_time_high": "all_time_high_usd",
    "ath": "all_time_high_usd",
    "all_time_low": "all_time_low_usd",
    "atl": "all_time_low_usd",
    # dates
    "launch_date_iso": "launch_date",
    "date_launched": "launch_date",
    # community
    "community_score": "community_rank",
    "social_rank": "community_rank",
    # other
    "network": "network_type",
    "coin_category": "category",
    "sector": "category",
    "last_updated_utc": "last_updated",
}


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and apply aliases."""
    df = df.copy()
    df.columns = [_normalise_col_name(str(c)) for c in df.columns]
    df = df.rename(columns=_COL_ALIASES)
    # Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns to float, stripping commas/$ signs."""
    for col in cols:
        if col not in df.columns:
            continue
        raw = df[col].astype(str).str.replace(r"[$,\s%]", "", regex=True)
        df[col] = pd.to_numeric(raw, errors="coerce")
    return df


def _coerce_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns to UTC-aware datetime."""
    for col in cols:
        if col not in df.columns:
            continue
        try:
            parsed = pd.to_datetime(df[col], utc=True, errors="coerce")
            df[col] = parsed
        except Exception:
            logger.warning("Failed to parse date column '%s'", col)
            df[col] = pd.NaT
    return df


# ---------------------------------------------------------------------------
# Symbol canonicalisation
# ---------------------------------------------------------------------------

def _canonicalise_symbol(sym: str) -> str:
    """Return upper-case canonical symbol, stripping exchange suffixes."""
    sym = str(sym).strip().upper()
    # Remove common suffixes: -USD, /USDT, USDT, etc.
    # Only strip if there is a non-empty base remaining to avoid
    # stripping single-token symbols like "BTC" or "USDT" themselves.
    cleaned = re.sub(r"[-/]?(USD[TC]?|BUSD|USDC)$", "", sym)
    if cleaned:
        return cleaned
    return sym


def _build_yfinance_symbol(canonical: str, overrides: dict[str, str]) -> str:
    """Return the yfinance-compatible ticker for a canonical symbol."""
    return overrides.get(canonical, f"{canonical}-USD")


def _build_exchange_symbol(canonical: str) -> str:
    """Return a CCXT-compatible exchange symbol."""
    return f"{canonical}/USDT"


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------

def ingest_crypto_dataset(
    csv_path: str | Path,
    as_of: datetime | None = None,
    append_snapshot: bool = True,
    data_dir: str | Path | None = None,
    yfinance_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load ``csv_path``, normalise, and write parquet artefacts.

    Parameters
    ----------
    csv_path:
        Path to the ``cryptocurrency_dataset.csv`` file.
    as_of:
        Timestamp for the market snapshot.  Defaults to UTC now.
    append_snapshot:
        When *True* (default), the snapshot is appended to any existing
        ``market_snapshot.parquet``; otherwise it overwrites.
    data_dir:
        Output directory for parquet files.  Defaults to ``data/crypto``.
    yfinance_overrides:
        Extra canonical → yfinance symbol mappings.

    Returns
    -------
    dict
        Summary: rows_read, rows_kept, duplicates_removed, nulls_filled.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    if as_of is None:
        as_of = datetime.now(timezone.utc)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    overrides = {**_YFINANCE_OVERRIDES, **(yfinance_overrides or {})}

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    logger.info("Loading dataset from %s", csv_path)
    try:
        raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(csv_path, encoding="latin-1")

    rows_read = len(raw)
    logger.info("Rows read: %d", rows_read)

    # ------------------------------------------------------------------
    # 2. Normalise columns
    # ------------------------------------------------------------------
    df = _map_columns(raw)
    logger.info("Columns after normalisation: %s", list(df.columns))

    # Ensure canonical_symbol exists
    if "canonical_symbol" not in df.columns:
        raise ValueError(
            "Cannot find a 'symbol' or 'canonical_symbol' column in the dataset. "
            f"Columns present: {list(df.columns)}"
        )

    df["canonical_symbol"] = df["canonical_symbol"].apply(_canonicalise_symbol)

    # ------------------------------------------------------------------
    # 3. Numeric / date coercion
    # ------------------------------------------------------------------
    numeric_cols = [
        "price_usd", "market_cap_usd", "volume_24h_usd", "change_24h_pct",
        "circulating_supply", "max_supply", "all_time_high_usd",
        "all_time_low_usd", "community_rank",
    ]
    df = _coerce_numeric(df, numeric_cols)
    df = _coerce_dates(df, ["launch_date", "last_updated"])

    # ------------------------------------------------------------------
    # 4. Null filling / counting
    # ------------------------------------------------------------------
    existing_numeric = [c for c in numeric_cols if c in df.columns]
    nulls_before = df[existing_numeric].isnull().sum().sum() if existing_numeric else 0
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    nulls_filled = int(nulls_before)
    logger.info("Nulls filled: %d", nulls_filled)

    # ------------------------------------------------------------------
    # 5. Deduplication by canonical_symbol (keep first / highest market cap)
    # ------------------------------------------------------------------
    pre_dedup = len(df)
    if "market_cap_usd" in df.columns:
        df = df.sort_values("market_cap_usd", ascending=False)
    df = df.drop_duplicates(subset=["canonical_symbol"], keep="first")
    duplicates_removed = pre_dedup - len(df)
    rows_kept = len(df)
    logger.info(
        "Rows kept: %d  (duplicates removed: %d)", rows_kept, duplicates_removed,
    )

    # ------------------------------------------------------------------
    # 6. Asset master
    # ------------------------------------------------------------------
    asset_master_cols = [c for c in _ASSET_MASTER_COLS if c in df.columns]
    asset_master = df[asset_master_cols].copy().reset_index(drop=True)
    am_path = data_dir / _ASSET_MASTER_FILE
    asset_master.to_parquet(am_path, engine="pyarrow", index=False)
    logger.info("Wrote asset_master → %s (%d rows)", am_path, len(asset_master))

    # ------------------------------------------------------------------
    # 7. Market snapshot (append-only)
    # ------------------------------------------------------------------
    snap_cols = [c for c in _MARKET_SNAPSHOT_COLS if c in df.columns]
    snapshot = df[snap_cols].copy().reset_index(drop=True)
    snapshot["as_of"] = as_of

    snap_path = data_dir / _MARKET_SNAPSHOT_FILE
    if append_snapshot and snap_path.exists():
        existing = pd.read_parquet(snap_path, engine="pyarrow")
        snapshot = pd.concat([existing, snapshot], ignore_index=True)
    snapshot.to_parquet(snap_path, engine="pyarrow", index=False)
    logger.info("Wrote market_snapshot → %s (%d total rows)", snap_path, len(snapshot))

    # ------------------------------------------------------------------
    # 8. Symbol map
    # ------------------------------------------------------------------
    sym_map = pd.DataFrame({
        "canonical_symbol": df["canonical_symbol"],
        "yfinance_symbol": df["canonical_symbol"].apply(
            lambda s: _build_yfinance_symbol(s, overrides)
        ),
        "exchange_symbol": df["canonical_symbol"].apply(_build_exchange_symbol),
        "display_name": df.get("name", df["canonical_symbol"]),
        "category": df.get("category", pd.Series([""] * len(df))),
        "network_type": df.get("network_type", pd.Series([""] * len(df))),
    }).reset_index(drop=True)

    smap_path = data_dir / _SYMBOL_MAP_FILE
    sym_map.to_parquet(smap_path, engine="pyarrow", index=False)
    logger.info("Wrote symbol_map → %s (%d rows)", smap_path, len(sym_map))

    summary: dict[str, Any] = {
        "rows_read": rows_read,
        "rows_kept": rows_kept,
        "duplicates_removed": duplicates_removed,
        "nulls_filled": nulls_filled,
        "asset_master_path": str(am_path),
        "market_snapshot_path": str(snap_path),
        "symbol_map_path": str(smap_path),
        "as_of": as_of.isoformat(),
    }
    logger.info("Ingestion complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Reader helpers
# ---------------------------------------------------------------------------

def load_asset_master(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return the asset master table, or empty DataFrame if not found."""
    path = (Path(data_dir) if data_dir else _DEFAULT_DATA_DIR) / _ASSET_MASTER_FILE
    if not path.exists():
        logger.warning("asset_master not found at %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


def load_market_snapshot(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return the market snapshot table, or empty DataFrame if not found."""
    path = (Path(data_dir) if data_dir else _DEFAULT_DATA_DIR) / _MARKET_SNAPSHOT_FILE
    if not path.exists():
        logger.warning("market_snapshot not found at %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


def load_symbol_map(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return the symbol map table, or empty DataFrame if not found."""
    path = (Path(data_dir) if data_dir else _DEFAULT_DATA_DIR) / _SYMBOL_MAP_FILE
    if not path.exists():
        logger.warning("symbol_map not found at %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


def get_yfinance_symbol(
    canonical: str,
    data_dir: str | Path | None = None,
) -> str:
    """Resolve a canonical symbol to its yfinance ticker.

    Falls back to ``f"{canonical}-USD"`` when symbol map is unavailable.
    """
    sym_map = load_symbol_map(data_dir)
    if not sym_map.empty and "canonical_symbol" in sym_map.columns:
        row = sym_map[sym_map["canonical_symbol"] == canonical.upper()]
        if not row.empty:
            return str(row["yfinance_symbol"].iloc[0])
    return _build_yfinance_symbol(canonical.upper(), _YFINANCE_OVERRIDES)


def compute_liquidity_scores(df: pd.DataFrame) -> pd.Series:
    """Compute liquidity_score = log10(mcap+1)*0.6 + log10(vol+1)*0.4.

    Parameters
    ----------
    df:
        DataFrame with ``market_cap_usd`` and ``volume_24h_usd`` columns.

    Returns
    -------
    pd.Series
        Float scores indexed like *df*.
    """
    mcap = pd.to_numeric(df.get("market_cap_usd", 0), errors="coerce").fillna(0.0)
    vol = pd.to_numeric(df.get("volume_24h_usd", 0), errors="coerce").fillna(0.0)
    score = (
        mcap.apply(lambda x: math.log10(x + 1)) * 0.6
        + vol.apply(lambda x: math.log10(x + 1)) * 0.4
    )
    return score
