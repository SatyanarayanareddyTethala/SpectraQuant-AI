"""Datetime normalization utilities for SpectraQuant-AI-V3.

These helpers enforce explicit UTC handling for all time-series artefacts that
flow through the V3 ingestion, feature, and reporting stack. Silent timezone
coercion is forbidden: callers must either receive a normalized UTC timestamp
column/index or a typed schema error.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from spectraquant_v3.core.errors import DataSchemaError, EmptyPriceDataError


def normalize_datetime_frame(
    df: pd.DataFrame,
    *,
    label: str,
    timestamp_column: str = "timestamp",
    allow_index: bool = True,
) -> pd.DataFrame:
    """Return a copy of *df* with an explicit UTC ``timestamp`` column.

    Rules enforced:
    - ``df`` must be a non-empty DataFrame.
    - A timestamp source must exist either as ``timestamp_column`` or, when
      ``allow_index`` is true, as a ``DatetimeIndex``.
    - All timestamps are parsed to timezone-aware UTC values.
    - Rows are sorted by timestamp and duplicate timestamps are rejected.

    Raises:
        EmptyPriceDataError: If *df* is empty.
        DataSchemaError: If timestamps are missing, unparsable, or duplicated.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataSchemaError(
            f"{label}: expected pandas DataFrame, got {type(df).__name__}."
        )
    if df.empty:
        raise EmptyPriceDataError(f"{label}: DataFrame is empty.")

    out = df.copy()
    if timestamp_column in out.columns:
        raw_ts: Any = out[timestamp_column]
    elif allow_index and isinstance(out.index, pd.DatetimeIndex):
        raw_ts = pd.Series(out.index, index=out.index)
        out = out.reset_index(drop=False)
        index_name = out.columns[0]
        out = out.rename(columns={index_name: timestamp_column})
    else:
        raise DataSchemaError(
            f"{label}: missing '{timestamp_column}' column and DatetimeIndex fallback."
        )

    try:
        parsed = pd.to_datetime(raw_ts, utc=True, errors="raise")
    except Exception as exc:  # noqa: BLE001
        raise DataSchemaError(
            f"{label}: could not parse '{timestamp_column}' values as datetimes: {exc}"
        ) from exc

    if parsed.isna().any():
        raise DataSchemaError(
            f"{label}: '{timestamp_column}' contains null/NaT values after parsing."
        )

    out[timestamp_column] = pd.Series(parsed.to_numpy(), index=out.index)
    out = out.sort_values(timestamp_column, kind="stable").reset_index(drop=True)

    if out[timestamp_column].duplicated().any():
        dupes = out.loc[out[timestamp_column].duplicated(), timestamp_column].astype(str).head(5).tolist()
        raise DataSchemaError(
            f"{label}: duplicate timestamps detected after UTC normalization: {dupes}"
        )

    return out
