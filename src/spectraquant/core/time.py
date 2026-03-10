"""Centralized datetime handling utilities."""
from __future__ import annotations

import logging
from typing import Dict

from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

import pandas as pd

from spectraquant.data.normalize import normalize_price_frame

logger = logging.getLogger(__name__)

def ensure_datetime_column(
    df: pd.DataFrame,
    col: str = "date",
    *,
    tz: str = "UTC",
    forbid_epoch: bool = True,
) -> pd.DataFrame:
    """Ensure explicit, timezone-normalized datetime column exists."""

    if df is None or df.empty:
        raise ValueError("DataFrame is empty; cannot validate datetime column.")

    df = df.copy()
    if col not in df.columns and "datetime" in df.columns:
        df.rename(columns={"datetime": col}, inplace=True)

    if col not in df.columns:
        raise ValueError(f"Missing required datetime column: {col}")

    dt = pd.to_datetime(df[col], utc=True, errors="coerce")
    df[col] = dt
    df = df.dropna(subset=[col])

    if df.empty:
        raise ValueError("All datetime values are invalid or missing.")

    if forbid_epoch and (df[col].dt.normalize() == pd.Timestamp("1970-01-01", tz=tz)).any():
        raise ValueError("Epoch date 1970-01-01 detected in datetime column.")

    if tz and df[col].dt.tz is None:
        df[col] = df[col].dt.tz_localize(tz)
    else:
        df[col] = df[col].dt.tz_convert(tz)

    return df


def normalize_time_index(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Normalize dataframe indices to a UTC DatetimeIndex with validation."""

    if df is None:
        raise ValueError(f"{context}: dataframe is None")
    if df.empty:
        return df

    working = df.copy()

    if isinstance(working.index, pd.MultiIndex):
        if "date" not in working.index.names:
            raise ValueError(f"{context}: MultiIndex missing 'date' level")
        date_values = working.index.get_level_values("date")
        if is_numeric_dtype(date_values):
            raise ValueError(f"{context}: index 'date' level is numeric, expected datetime")
        if not isinstance(date_values, pd.DatetimeIndex):
            if not is_datetime64_any_dtype(date_values):
                date_values = pd.to_datetime(date_values, utc=True, errors="raise")
            else:
                date_values = pd.DatetimeIndex(date_values)
        if date_values.tz is None:
            date_values = date_values.tz_localize("UTC")
        else:
            date_values = date_values.tz_convert("UTC")
        index_arrays = []
        for name in working.index.names:
            if name == "date":
                index_arrays.append(date_values)
            else:
                index_arrays.append(working.index.get_level_values(name))
        working.index = pd.MultiIndex.from_arrays(index_arrays, names=working.index.names)
    elif isinstance(working.index, pd.DatetimeIndex):
        index = pd.to_datetime(working.index, utc=True, errors="raise")
        if index.tz is None:
            index = index.tz_localize("UTC")
        else:
            index = index.tz_convert("UTC")
        working.index = index
    elif "date" in working.columns:
        if is_numeric_dtype(working["date"]):
            raise ValueError(f"{context}: date column is numeric, expected datetime-like values")
        working["date"] = pd.to_datetime(working["date"], utc=True, errors="coerce")
        working = working.dropna(subset=["date"])
        if working.empty:
            raise ValueError(f"{context}: date column could not be parsed into datetime values")
        working = working.set_index("date", drop=False)
    else:
        raise ValueError(f"{context}: index is not datetime and no date column present")

    if not working.index.is_monotonic_increasing:
        working = working.sort_index()
    if working.index.has_duplicates:
        working = working.loc[~working.index.duplicated(keep="last")]

    return working


def resolve_prediction_date(ticker: str, price_data: Dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Return latest available price timestamp for a ticker."""

    price_df = price_data.get(ticker)
    if price_df is None or price_df.empty:
        raise ValueError(f"No price history available for {ticker}")

    price_df = normalize_price_frame(price_df)
    latest_date = price_df.index.max()
    if pd.isna(latest_date):
        raise ValueError(f"Price history for {ticker} has no valid dates")
    if latest_date.normalize() == pd.Timestamp("1970-01-01", tz="UTC"):
        raise ValueError(f"Invalid epoch date for {ticker} price history")
    return latest_date


def is_intraday_horizon(horizon: str) -> bool:
    return str(horizon).lower().endswith("m")


def _align_intraday_timestamp(ts: pd.Timestamp, horizon: str) -> tuple[pd.Timestamp, int]:
    minutes = int(str(horizon).lower().replace("m", ""))
    if ts.tzinfo is None:
        raise ValueError("Intraday timestamp missing timezone information")
    aligned = ts.floor("min") if (ts.second != 0 or ts.microsecond != 0) else ts
    aligned = aligned.floor(f"{minutes}min")
    return aligned, minutes


def resolve_prediction_date_for_horizon(
    ticker: str,
    horizon: str,
    daily_prices: Dict[str, pd.DataFrame],
    intraday_prices: Dict[str, pd.DataFrame] | None = None,
) -> pd.Timestamp:
    """Return latest available price timestamp for a ticker and horizon."""

    if is_intraday_horizon(horizon):
        if not intraday_prices:
            raise ValueError(f"Intraday price data missing for horizon {horizon}")
        latest = resolve_prediction_date(ticker, intraday_prices)
        aligned, minutes = _align_intraday_timestamp(latest, horizon)
        if aligned != latest:
            logger.warning(
                "Intraday timestamp %s not aligned to %sm boundary for %s; using %s",
                latest,
                minutes,
                horizon,
                aligned,
            )
        return aligned
    return resolve_prediction_date(ticker, daily_prices)
