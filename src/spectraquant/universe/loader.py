from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

MISSING_UNIVERSE_FILE = "MISSING_UNIVERSE_FILE"
UNIVERSE_SCHEMA_INVALID = "UNIVERSE_SCHEMA_INVALID"
NO_VALID_TICKERS_AFTER_CLEAN = "NO_VALID_TICKERS_AFTER_CLEAN"


@dataclass(frozen=True)
class UniverseDiagnostics:
    code: str
    message: str
    details: dict[str, Any]


def load_nse_universe(
    path: str | Path,
    *,
    symbol_column: str = "SYMBOL",
    suffix: str = ".NS",
    filter_series_eq: bool = True,
) -> tuple[list[str], dict[str, Any], list[UniverseDiagnostics]]:
    file_path = Path(path)
    if not file_path.exists():
        return [], _empty_meta(), [
            UniverseDiagnostics(
                code=MISSING_UNIVERSE_FILE,
                message="Universe CSV not found.",
                details={"path": str(file_path)},
            )
        ]

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001
        return [], _empty_meta(), [
            UniverseDiagnostics(
                code=UNIVERSE_SCHEMA_INVALID,
                message="Failed to read universe CSV.",
                details={"path": str(file_path), "error": str(exc)},
            )
        ]

    normalized_cols = {str(c).strip().lower(): c for c in df.columns}
    resolved_symbol_column = normalized_cols.get(str(symbol_column).strip().lower(), symbol_column)
    if resolved_symbol_column not in df.columns:
        return [], _empty_meta(), [
            UniverseDiagnostics(
                code=UNIVERSE_SCHEMA_INVALID,
                message="Universe CSV missing SYMBOL column.",
                details={"columns": list(df.columns), "expected": symbol_column},
            )
        ]

    raw_symbols = df[resolved_symbol_column].astype(str).str.strip().str.upper()
    raw_count = int(raw_symbols.shape[0])

    eq_count = raw_count
    if filter_series_eq and "SERIES" in df.columns:
        series_mask = df["SERIES"].astype(str).str.upper() == "EQ"
        df = df[series_mask]
        raw_symbols = df[resolved_symbol_column].astype(str).str.strip().str.upper()
        eq_count = int(raw_symbols.shape[0])

    blank_mask = raw_symbols.isna() | (raw_symbols == "") | (raw_symbols == "NAN")
    cleaned = raw_symbols[~blank_mask]
    duplicates_mask = cleaned.duplicated()
    cleaned = cleaned[~duplicates_mask]
    tickers = [f"{symbol}{suffix}" for symbol in cleaned.tolist()]

    dropped_blank = int(blank_mask.sum())
    dropped_duplicates = int(duplicates_mask.sum())
    dropped_count = dropped_blank + dropped_duplicates
    examples_dropped = raw_symbols[blank_mask].head(5).tolist()

    meta = {
        "raw_count": raw_count,
        "eq_count": eq_count,
        "clean_count": len(cleaned),
        "cleaned_count": len(cleaned),
        "dropped_count": dropped_count,
        "examples_dropped": examples_dropped,
        "dropped_reasons": {
            "blank_or_nan": dropped_blank,
            "duplicates": dropped_duplicates,
        },
    }

    diagnostics: list[UniverseDiagnostics] = []
    if not tickers:
        diagnostics.append(
            UniverseDiagnostics(
                code=NO_VALID_TICKERS_AFTER_CLEAN,
                message="No valid tickers remain after cleaning.",
                details={"raw_count": raw_count, "eq_count": eq_count},
            )
        )

    return tickers, meta, diagnostics


def _empty_meta() -> dict[str, Any]:
    return {
        "raw_count": 0,
        "eq_count": 0,
        "clean_count": 0,
        "cleaned_count": 0,
        "dropped_count": 0,
        "examples_dropped": [],
        "dropped_reasons": {"blank_or_nan": 0, "duplicates": 0},
    }
