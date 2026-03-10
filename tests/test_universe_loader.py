from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectraquant.universe.loader import (
    NO_VALID_TICKERS_AFTER_CLEAN,
    UNIVERSE_SCHEMA_INVALID,
    load_nse_universe,
)


def test_missing_symbol_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"BAD": ["RELIANCE"]}).to_csv(path, index=False)
    tickers, meta, diagnostics = load_nse_universe(path)
    assert tickers == []
    assert meta["raw_count"] == 0
    assert any(diag.code == UNIVERSE_SCHEMA_INVALID for diag in diagnostics)


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    pd.DataFrame({"SYMBOL": []}).to_csv(path, index=False)
    tickers, meta, diagnostics = load_nse_universe(path)
    assert tickers == []
    assert meta["raw_count"] == 0
    assert any(diag.code == NO_VALID_TICKERS_AFTER_CLEAN for diag in diagnostics)


def test_duplicate_symbols(tmp_path: Path) -> None:
    path = tmp_path / "dupes.csv"
    pd.DataFrame({"SYMBOL": ["RELIANCE", "RELIANCE"]}).to_csv(path, index=False)
    tickers, meta, diagnostics = load_nse_universe(path)
    assert tickers == ["RELIANCE.NS"]
    assert meta["dropped_reasons"]["duplicates"] == 1
    assert diagnostics == []


def test_suffix_appending(tmp_path: Path) -> None:
    path = tmp_path / "symbols.csv"
    pd.DataFrame({"SYMBOL": ["RELIANCE"]}).to_csv(path, index=False)
    tickers, meta, diagnostics = load_nse_universe(path)
    assert tickers == ["RELIANCE.NS"]
    assert meta["cleaned_count"] == 1
    assert diagnostics == []
