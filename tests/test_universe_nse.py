from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

from spectraquant.core.universe import (
    CANONICAL_COLUMNS,
    get_symbols,
    load_universe,
    map_symbol_to_ticker,
    parse_nse_equity_list,
)


def test_universe_nse_parse_smoke() -> None:
    fixture = FIXTURES / "EQUITY_L_sample.csv"
    raw = pd.read_csv(fixture)

    parsed, dedup_removed = parse_nse_equity_list(raw, asof_date="2026-01-01")

    assert list(parsed.columns) == CANONICAL_COLUMNS
    assert not parsed.empty
    assert parsed["ticker"].str.endswith(".NS").all()
    assert parsed["symbol"].is_unique
    assert dedup_removed >= 1


def test_universe_loader() -> None:
    fixture = FIXTURES / "EQUITY_L_sample.csv"
    raw = pd.read_csv(fixture)
    parsed, _ = parse_nse_equity_list(raw, asof_date="2026-01-01")

    out = Path("/tmp/universe_nse_test.csv")
    parsed.to_csv(out, index=False)

    loaded = load_universe(out)
    symbols = get_symbols(loaded)
    assert "RELIANCE" in symbols
    assert map_symbol_to_ticker(loaded, "RELIANCE") == "RELIANCE.NS"
    assert map_symbol_to_ticker(loaded, "UNKNOWN") is None
