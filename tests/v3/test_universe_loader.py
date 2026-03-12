"""Unit tests for the hybrid universe loader.

Covers:
- Valid universe loading and partitioning by asset class
- Missing required column detection
- Duplicate symbol deduplication
- Invalid asset_class rejection
- Empty symbol field rejection
- Universe size cap enforcement
- get_symbols_by_class helper
- inject_universe_into_config integration shim
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from spectraquant_v3.core.errors import UniverseValidationError
from spectraquant_v3.core.universe_loader import (
    MAX_UNIVERSE_SIZE,
    REQUIRED_COLUMNS,
    UniverseAsset,
    get_symbols_by_class,
    inject_universe_into_config,
    load_universe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path: Path, content: str, filename: str = "test_universe.csv") -> Path:
    """Write *content* to a temp CSV file and return its path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content).strip())
    return p


_VALID_CSV = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes
equity,RELIANCE.NS,NSE,energy,high,medium,Test equity 1
equity,HDFCBANK.NS,NSE,banking,very_high,medium,Test equity 2
crypto,BTCUSDT,BINANCE,crypto,very_high,high,Bitcoin
crypto,ETHUSDT,BINANCE,crypto,very_high,high,Ethereum
forex,EURUSD,FX,currency,very_high,low,Euro USD
forex,GBPUSD,FX,currency,very_high,low,GBP USD
"""


# ---------------------------------------------------------------------------
# 1. Valid universe
# ---------------------------------------------------------------------------


class TestValidUniverse:
    def test_returns_dict_with_all_three_keys(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        result = load_universe(p)
        assert "equities" in result
        assert "crypto" in result
        assert "forex" in result

    def test_equities_count(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        result = load_universe(p)
        assert len(result["equities"]) == 2

    def test_crypto_count(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        result = load_universe(p)
        assert len(result["crypto"]) == 2

    def test_forex_count(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        result = load_universe(p)
        assert len(result["forex"]) == 2

    def test_returns_universe_asset_instances(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        result = load_universe(p)
        for cls_key in ("equities", "crypto", "forex"):
            for asset in result[cls_key]:
                assert isinstance(asset, UniverseAsset)

    def test_symbol_values_are_correct(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        result = load_universe(p)
        equity_symbols = [a.symbol for a in result["equities"]]
        assert "RELIANCE.NS" in equity_symbols
        assert "HDFCBANK.NS" in equity_symbols

    def test_asset_class_field_normalised_to_lowercase(self, tmp_path: Path) -> None:
        csv_content = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes
EQUITY,RELIANCE.NS,NSE,energy,high,medium,uppercase asset class
"""
        p = _write_csv(tmp_path, csv_content)
        result = load_universe(p)
        assert len(result["equities"]) == 1
        assert result["equities"][0].asset_class == "equity"

    def test_whitespace_stripped_from_values(self, tmp_path: Path) -> None:
        csv_content = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes
 equity , RELIANCE.NS , NSE , energy , high , medium , padded
"""
        p = _write_csv(tmp_path, csv_content)
        result = load_universe(p)
        assert len(result["equities"]) == 1
        assert result["equities"][0].symbol == "RELIANCE.NS"

    def test_optional_notes_column_missing_defaults_empty(self, tmp_path: Path) -> None:
        csv_content = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score
equity,RELIANCE.NS,NSE,energy,high,medium
"""
        p = _write_csv(tmp_path, csv_content)
        result = load_universe(p)
        assert result["equities"][0].notes == ""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_universe(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# 2. Missing required column
# ---------------------------------------------------------------------------


class TestMissingColumn:
    @pytest.mark.parametrize("missing_col", list(REQUIRED_COLUMNS))
    def test_missing_required_column_raises(self, tmp_path: Path, missing_col: str) -> None:
        # Build a CSV header that omits *missing_col*
        cols = [c for c in REQUIRED_COLUMNS if c != missing_col]
        header = ",".join(cols)
        row = ",".join(["equity" if c == "asset_class" else "TEST" for c in cols])
        csv_content = f"{header}\n{row}\n"
        p = _write_csv(tmp_path, csv_content)
        with pytest.raises(UniverseValidationError, match=missing_col):
            load_universe(p)

    def test_completely_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(UniverseValidationError):
            load_universe(p)


# ---------------------------------------------------------------------------
# 3. Duplicate symbols
# ---------------------------------------------------------------------------


class TestDuplicateSymbols:
    def test_duplicate_symbols_are_deduplicated(self, tmp_path: Path) -> None:
        csv_content = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes
equity,RELIANCE.NS,NSE,energy,high,medium,first
equity,RELIANCE.NS,NSE,energy,high,medium,duplicate
equity,HDFCBANK.NS,NSE,banking,very_high,medium,
"""
        p = _write_csv(tmp_path, csv_content)
        result = load_universe(p)
        symbols = [a.symbol for a in result["equities"]]
        # Should contain each symbol only once
        assert symbols.count("RELIANCE.NS") == 1
        assert len(result["equities"]) == 2

    def test_duplicate_tracking_is_reported(self, tmp_path: Path) -> None:
        csv_content = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes
crypto,BTCUSDT,BINANCE,crypto,very_high,high,
crypto,BTCUSDT,BINANCE,crypto,very_high,high,dupe
"""
        p = _write_csv(tmp_path, csv_content)
        result = load_universe(p)
        dropped: list[str] = result.get("_duplicates_dropped", [])  # type: ignore[assignment]
        assert "BTCUSDT" in dropped

    def test_first_occurrence_wins_on_duplicate(self, tmp_path: Path) -> None:
        csv_content = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes
equity,RELIANCE.NS,NSE,energy,high,medium,original
equity,RELIANCE.NS,BSE,energy,low,medium,duplicate_different_exchange
"""
        p = _write_csv(tmp_path, csv_content)
        result = load_universe(p)
        assert result["equities"][0].exchange == "NSE"


# ---------------------------------------------------------------------------
# 4. Invalid asset_class
# ---------------------------------------------------------------------------


class TestInvalidAssetClass:
    def test_unknown_asset_class_raises(self, tmp_path: Path) -> None:
        csv_content = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes
unknown_class,SOMETOKEN,EXCHANGE,sector,high,medium,
"""
        p = _write_csv(tmp_path, csv_content)
        with pytest.raises(UniverseValidationError, match="invalid asset_class"):
            load_universe(p)

    def test_mixed_valid_invalid_asset_class_raises(self, tmp_path: Path) -> None:
        csv_content = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes
equity,RELIANCE.NS,NSE,energy,high,medium,
futures,BTCFUT,CME,derivatives,very_high,very_high,invalid class
"""
        p = _write_csv(tmp_path, csv_content)
        with pytest.raises(UniverseValidationError):
            load_universe(p)


# ---------------------------------------------------------------------------
# 5. Empty symbol field
# ---------------------------------------------------------------------------


class TestEmptySymbol:
    def test_empty_symbol_raises(self, tmp_path: Path) -> None:
        csv_content = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes
equity,,NSE,energy,high,medium,missing symbol
"""
        p = _write_csv(tmp_path, csv_content)
        with pytest.raises(UniverseValidationError, match="empty symbol"):
            load_universe(p)


# ---------------------------------------------------------------------------
# 6. Universe size cap
# ---------------------------------------------------------------------------


class TestUniverseSizeCap:
    def test_exceeding_max_size_raises(self, tmp_path: Path) -> None:
        header = "asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes"
        rows = [
            f"equity,SYM{i:03d}.NS,NSE,sector,high,medium,"
            for i in range(MAX_UNIVERSE_SIZE + 1)
        ]
        p = _write_csv(tmp_path, header + "\n" + "\n".join(rows))
        with pytest.raises(UniverseValidationError, match=str(MAX_UNIVERSE_SIZE)):
            load_universe(p)

    def test_exactly_at_max_size_is_allowed(self, tmp_path: Path) -> None:
        header = "asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes"
        rows = [
            f"equity,SYM{i:03d}.NS,NSE,sector,high,medium,"
            for i in range(MAX_UNIVERSE_SIZE)
        ]
        p = _write_csv(tmp_path, header + "\n" + "\n".join(rows))
        result = load_universe(p)
        assert len(result["equities"]) == MAX_UNIVERSE_SIZE


# ---------------------------------------------------------------------------
# 7. get_symbols_by_class helper
# ---------------------------------------------------------------------------


class TestGetSymbolsByClass:
    def test_returns_equity_symbols(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        universe = load_universe(p)
        symbols = get_symbols_by_class(universe, "equities")
        assert isinstance(symbols, list)
        assert all(isinstance(s, str) for s in symbols)
        assert "RELIANCE.NS" in symbols

    def test_returns_crypto_symbols(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        universe = load_universe(p)
        symbols = get_symbols_by_class(universe, "crypto")
        assert "BTCUSDT" in symbols

    def test_returns_forex_symbols(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        universe = load_universe(p)
        symbols = get_symbols_by_class(universe, "forex")
        assert "EURUSD" in symbols

    def test_invalid_class_raises_key_error(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        universe = load_universe(p)
        with pytest.raises(KeyError):
            get_symbols_by_class(universe, "nonexistent")


# ---------------------------------------------------------------------------
# 8. inject_universe_into_config integration shim
# ---------------------------------------------------------------------------


class TestInjectUniverseIntoConfig:
    def _base_cfg(self) -> dict:
        return {
            "run": {"mode": "test"},
            "cache": {"root": "data/cache"},
            "qa": {},
            "execution": {},
            "portfolio": {},
            "equities": {"universe": {"tickers": ["OLD_EQUITY.NS"]}},
            "crypto": {"symbols": ["OLDBTC"]},
        }

    def test_equity_symbols_are_replaced(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        cfg, _ = inject_universe_into_config(self._base_cfg(), p)
        tickers = cfg["equities"]["universe"]["tickers"]
        assert "RELIANCE.NS" in tickers
        assert "OLD_EQUITY.NS" not in tickers

    def test_crypto_symbols_are_replaced(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        cfg, _ = inject_universe_into_config(self._base_cfg(), p)
        assert "BTCUSDT" in cfg["crypto"]["symbols"]
        assert "OLDBTC" not in cfg["crypto"]["symbols"]

    def test_original_config_is_not_mutated(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        original = self._base_cfg()
        original_equity_tickers = list(original["equities"]["universe"]["tickers"])
        inject_universe_into_config(original, p)
        assert original["equities"]["universe"]["tickers"] == original_equity_tickers

    def test_returns_universe_dict(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _VALID_CSV)
        _, universe = inject_universe_into_config(self._base_cfg(), p)
        assert "equities" in universe
        assert "crypto" in universe
        assert "forex" in universe

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            inject_universe_into_config(self._base_cfg(), tmp_path / "missing.csv")

    def test_validation_error_propagates(self, tmp_path: Path) -> None:
        bad_csv = """\
asset_class,symbol,exchange,sector,liquidity_score,volatility_score,notes
badclass,SYM,EX,sector,high,medium,
"""
        p = _write_csv(tmp_path, bad_csv)
        with pytest.raises(UniverseValidationError):
            inject_universe_into_config(self._base_cfg(), p)
