from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

from dashboard.utils.diagnostics import (
    DATE_INDEX_NOT_DATETIME,
    SIGNAL_RETURN_MISALIGNMENT,
    TIMEZONE_MISMATCH,
    Diagnostic,
    DiagnosticError,
    make_diagnostic,
)
from dashboard.utils.logging import configure_logger


def _is_integer_like(value: Any) -> bool:
    return is_integer_dtype(value) or is_numeric_dtype(value)


def normalize_time_index(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    if df is None:
        return df
    if isinstance(df.index, pd.RangeIndex) or _is_integer_like(df.index):
        raise DiagnosticError(
            make_diagnostic(
                DATE_INDEX_NOT_DATETIME,
                detected={"index_type": type(df.index).__name__},
                suggestion="Ensure the index is a datetime index (no integer/range index).",
                message="Index is not datetime.",
            )
        )
    if not is_datetime64_any_dtype(df.index):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="raise")
        except Exception as exc:  # noqa: BLE001
            raise DiagnosticError(
                make_diagnostic(
                    DATE_INDEX_NOT_DATETIME,
                    detected={"index_sample": str(df.index[:3]), "error": str(exc)},
                    suggestion="Parse dates before setting the index to avoid epoch coercion.",
                    message="Index could not be parsed as datetime.",
                )
            ) from exc

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    df = df.copy()
    df.index = idx
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    return df


def align_on_time_index(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "inner",
    min_overlap: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Diagnostic]]:
    logger = configure_logger()
    diagnostics: list[Diagnostic] = []

    left_tz = getattr(left.index, "tz", None)
    right_tz = getattr(right.index, "tz", None)
    if (left_tz is None) != (right_tz is None):
        diagnostics.append(
            make_diagnostic(
                TIMEZONE_MISMATCH,
                detected={"left_tz": str(left_tz), "right_tz": str(right_tz)},
                suggestion="Ensure both datasets are timezone-aware and normalized to UTC.",
                message="Timezone mismatch detected.",
            )
        )

    left_norm = normalize_time_index(left)
    right_norm = normalize_time_index(right)

    if how == "left":
        left_aligned = left_norm
        right_aligned = right_norm.reindex(left_norm.index)
    elif how == "right":
        right_aligned = right_norm
        left_aligned = left_norm.reindex(right_norm.index)
    else:
        intersection = left_norm.index.intersection(right_norm.index)
        left_aligned = left_norm.loc[intersection]
        right_aligned = right_norm.loc[intersection]

    overlap_start = None
    overlap_end = None
    if not left_aligned.empty and not right_aligned.empty:
        overlap_start = max(left_aligned.index.min(), right_aligned.index.min())
        overlap_end = min(left_aligned.index.max(), right_aligned.index.max())

    logger.info(
        "Alignment stats: left=%s right=%s aligned_left=%s aligned_right=%s overlap_start=%s overlap_end=%s",
        len(left_norm.index),
        len(right_norm.index),
        len(left_aligned.index),
        len(right_aligned.index),
        overlap_start,
        overlap_end,
    )

    overlap_rows = min(len(left_aligned.index), len(right_aligned.index))
    if overlap_rows < min_overlap:
        diagnostics.append(
            make_diagnostic(
                SIGNAL_RETURN_MISALIGNMENT,
                detected={
                    "left_rows": len(left_norm.index),
                    "right_rows": len(right_norm.index),
                    "overlap_rows": overlap_rows,
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                },
                suggestion="Verify date ranges overlap for signals, returns, and prices.",
                message="Insufficient overlap between time indexes.",
            )
        )

    return left_aligned, right_aligned, diagnostics
