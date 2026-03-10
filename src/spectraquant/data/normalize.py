"""Helpers for normalizing price column layouts from yfinance outputs."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_column_name(name: Any) -> str:
    """Normalize potentially multi-indexed column labels from yfinance."""

    if isinstance(name, tuple):
        parts = [p for p in name if p not in {None, ""}]
        name = "_".join(str(p) for p in parts if p is not None)

    raw = str(name)
    if raw.startswith("(") and "," in raw:
        try:
            raw = raw.strip()[1:-1].split(",")[0]
        except Exception:  # noqa: BLE001 - defensive cleanup only
            raw = str(name)

    cleaned = raw.strip().strip("'").replace(" `", "").replace("`", "")
    cleaned = cleaned.replace(" ", "_").replace("__", "_").strip("_")
    return cleaned.lower()


def normalize_price_columns(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Flatten tuple/MultiIndex columns and surface price fields.

    Ensures standard labels (date, open, high, low, close, adj_close, volume)
    are present when available and logs when normalization occurred.
    """

    if df is None or df.empty:
        return df

    original_columns = list(df.columns)
    flattened = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [p for p in col if p not in {None, ""}]
            flattened.append("_".join(str(p).strip().lower() for p in parts if p is not None))
        else:
            flattened.append(_normalize_column_name(col))

    df = df.copy()
    df.columns = flattened
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        logger.warning("Dropping duplicated columns after normalization: %s", duplicates)
        df = df.loc[:, ~df.columns.duplicated()]

    standard_map = {
        "close": ["close", "adj_close", "close_close"],
        "adj_close": ["adj_close", "close_adj_close"],
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "volume": ["volume"],
    }

    if ticker:
        suffix = ticker.lower()
        for col in list(df.columns):
            if col.endswith(f"_{suffix}"):
                base = col.replace(f"_{suffix}", "")
                if base and base not in df.columns:
                    df[base] = df[col]

    if "date" not in df.columns and "datetime" in df.columns:
        df.rename(columns={"datetime": "date"}, inplace=True)

    applied = False
    for target, aliases in standard_map.items():
        if target in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df[target] = df[alias]
                applied = True
                break

    if applied:
        logger.info("Normalized price columns for %s; original columns: %s", ticker, original_columns)

    return df


def normalize_price_frame(
    df: pd.DataFrame,
    *,
    tz: str = "UTC",
    index_name: str = "date",
) -> pd.DataFrame:
    """Normalize price dataframes to a timezone-aware datetime index."""

    if df is None or df.empty:
        return df

    working = df.copy()

    if isinstance(working.index, pd.MultiIndex):
        if index_name in working.index.names:
            working = working.reset_index()
        else:
            working = working.reset_index()

    if index_name in working.columns:
        working[index_name] = pd.to_datetime(working[index_name], utc=True, errors="coerce")
        working = working.dropna(subset=[index_name])
        working = working.set_index(index_name, drop=True)
    elif isinstance(working.index, pd.DatetimeIndex):
        working.index = pd.to_datetime(working.index, utc=True, errors="coerce")
    else:
        coerced = pd.to_datetime(working.index, utc=True, errors="coerce")
        if coerced.isna().all():
            raise ValueError("Missing datetime column or index in price frame.")
        working.index = coerced

    if tz:
        if working.index.tz is None:
            working.index = working.index.tz_localize(tz)
        else:
            working.index = working.index.tz_convert(tz)

    working.index.name = index_name
    if index_name in working.columns:
        working = working.drop(columns=[index_name])

    working = working.loc[~working.index.isna()]
    if not working.empty:
        working = working[~working.index.duplicated(keep="last")]
        working = working.sort_index()

    return working


def assert_price_frame(df: pd.DataFrame, *, context: str = "") -> None:
    """Assert normalized price-frame invariants to prevent ambiguous date handling."""

    if df is None or df.empty:
        raise ValueError("Price dataframe is empty; cannot validate date/index state.")

    errors = []
    if "date" in df.columns:
        errors.append("unexpected 'date' column present")
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"index type is {type(df.index).__name__}, expected DatetimeIndex")
    if df.index.name != "date":
        errors.append(f"index name is {df.index.name!r}, expected 'date'")

    if errors:
        details = {
            "index_names": df.index.names,
            "index_dtype": str(df.index.dtype),
            "has_date_column": "date" in df.columns,
            "columns": list(df.columns[:10]),
        }
        message = "Price frame validation failed"
        if context:
            message = f"{message} ({context})"
        raise AssertionError(f"{message}: {errors} details={details}")


__all__ = ["normalize_price_columns", "normalize_price_frame", "assert_price_frame"]
