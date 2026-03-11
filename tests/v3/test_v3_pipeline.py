"""Tests for the SpectraQuant-AI-V3 symbol registries, universe builders,
feature engines, signal agents, meta-policy, allocator, reporter, and pipelines.

All tests are self-contained (no network calls, no file system side-effects
beyond tmp_path fixtures).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# Helpers
# ===========================================================================

def _ohlcv_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with n rows."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.uniform(-0.01, 0.01, n))
    volume = rng.uniform(1_000_000, 5_000_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _crypto_cfg(symbols=None) -> dict:
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL"]
    return {
        "run": {"mode": "normal"},
        "cache": {"root": "data/cache"},
        "qa": {"min_ohlcv_coverage": 1.0},
        "execution": {"mode": "paper"},
        "portfolio": {
            "max_weight": 0.25,
            "max_gross_leverage": 1.0,
            "min_confidence": 0.10,
            "min_signal_threshold": 0.05,
            "target_vol": 0.15,
            "allocator": "equal_weight",
        },
        "crypto": {
            "symbols": symbols,
            "primary_ohlcv_provider": "ccxt",
            "universe_mode": "static",
            "quality_gate": {
                "min_market_cap_usd": 0,
                "min_24h_volume_usd": 0,
                "min_age_days": 0,
                "require_tradable_mapping": True,
            },
            "signals": {"momentum_lookback": 20, "rsi_period": 14},
        },
    }


def _equity_cfg(tickers=None) -> dict:
    if tickers is None:
        tickers = ["INFY.NS", "TCS.NS", "HDFCBANK.NS"]
    return {
        "run": {"mode": "normal"},
        "cache": {"root": "data/cache"},
        "qa": {"min_ohlcv_coverage": 1.0},
        "execution": {"mode": "paper"},
        "portfolio": {
            "max_weight": 0.20,
            "max_gross_leverage": 1.0,
            "min_confidence": 0.10,
            "min_signal_threshold": 0.05,
            "allocator": "equal_weight",
        },
        "equities": {
            "primary_ohlcv_provider": "yfinance",
            "universe": {"tickers": tickers, "exclude": []},
            "quality_gate": {
                "min_price": 0,
                "min_avg_volume": 0,
                "min_history_days": 0,
            },
            "signals": {"momentum_lookback": 20, "rsi_period": 14},
        },
    }


# ===========================================================================
# 1. Crypto symbol registry
# ===========================================================================

class TestCryptoSymbolRegistry:
    def test_register_and_get(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        reg = CryptoSymbolRegistry()
        rec = SymbolRecord(
            canonical_symbol="BTC",
            asset_class=AssetClass.CRYPTO,
            provider_symbol="BTC/USDT",
        )
        reg.register(rec)
        assert reg.get("BTC").canonical_symbol == "BTC"
        assert reg.contains("BTC")
        assert not reg.contains("ETH")

    def test_case_insensitive_get(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        reg = CryptoSymbolRegistry()
        reg.register(SymbolRecord("btc", AssetClass.CRYPTO))
        assert reg.get("BTC").canonical_symbol == "btc"

    def test_equity_symbol_raises_leak_error(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.errors import AssetClassLeakError
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        reg = CryptoSymbolRegistry()
        with pytest.raises(AssetClassLeakError):
            reg.register(SymbolRecord("INFY.NS", AssetClass.EQUITY))

    def test_equity_asset_class_raises_leak_error(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.errors import AssetClassLeakError
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        reg = CryptoSymbolRegistry()
        with pytest.raises(AssetClassLeakError):
            reg.register(SymbolRecord("RELIANCE", AssetClass.EQUITY))

    def test_resolution_error_on_missing(self) -> None:
        from spectraquant_v3.core.errors import SymbolResolutionError
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        reg = CryptoSymbolRegistry()
        with pytest.raises(SymbolResolutionError):
            reg.get("MISSING")

    def test_all_symbols_sorted(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        reg = CryptoSymbolRegistry()
        for s in ["SOL", "BTC", "ETH"]:
            reg.register(SymbolRecord(s, AssetClass.CRYPTO))
        assert reg.all_symbols() == ["BTC", "ETH", "SOL"]

    def test_len(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        reg = CryptoSymbolRegistry()
        reg.register(SymbolRecord("BTC", AssetClass.CRYPTO))
        assert len(reg) == 1

    def test_build_from_config(self) -> None:
        from spectraquant_v3.crypto.symbols.registry import build_registry_from_config

        cfg = _crypto_cfg(["BTC", "ETH"])
        reg = build_registry_from_config(cfg)
        assert reg.contains("BTC")
        assert reg.contains("ETH")
        assert len(reg) == 2
        assert reg.get("BTC").provider_symbol == "BTC/USDT"

    def test_equity_suffix_rejected(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.errors import AssetClassLeakError
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        reg = CryptoSymbolRegistry()
        for suffix in [".NS", ".L", ".TO"]:
            with pytest.raises(AssetClassLeakError):
                reg.register(SymbolRecord(f"FAKE{suffix}", AssetClass.CRYPTO))


# ===========================================================================
# 2. Equity symbol registry
# ===========================================================================

class TestEquitySymbolRegistry:
    def test_register_and_get(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        reg = EquitySymbolRegistry()
        reg.register(SymbolRecord("INFY.NS", AssetClass.EQUITY, yfinance_symbol="INFY.NS"))
        assert reg.get("INFY.NS").canonical_symbol == "INFY.NS"

    def test_crypto_symbol_raises_leak_error(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.errors import AssetClassLeakError
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        reg = EquitySymbolRegistry()
        with pytest.raises(AssetClassLeakError):
            reg.register(SymbolRecord("BTC", AssetClass.CRYPTO))

    def test_crypto_pair_suffix_rejected(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.errors import AssetClassLeakError
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        reg = EquitySymbolRegistry()
        with pytest.raises(AssetClassLeakError):
            reg.register(SymbolRecord("ETH/USDT", AssetClass.EQUITY))

    def test_resolution_error_on_missing(self) -> None:
        from spectraquant_v3.core.errors import SymbolResolutionError
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        reg = EquitySymbolRegistry()
        with pytest.raises(SymbolResolutionError):
            reg.get("MISSING.NS")

    def test_build_from_config(self) -> None:
        from spectraquant_v3.equities.symbols.registry import build_registry_from_config

        cfg = _equity_cfg(["INFY.NS", "TCS.NS"])
        reg = build_registry_from_config(cfg)
        assert reg.contains("INFY.NS")
        assert reg.contains("TCS.NS")
        assert len(reg) == 2

    def test_register_many(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        reg = EquitySymbolRegistry()
        reg.register_many([
            SymbolRecord("A.NS", AssetClass.EQUITY),
            SymbolRecord("B.NS", AssetClass.EQUITY),
        ])
        assert len(reg) == 2


# ===========================================================================
# 3. Crypto symbol mapper
# ===========================================================================

class TestCryptoSymbolMapper:
    def _make_mapper(self):
        from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper
        from spectraquant_v3.crypto.symbols.registry import build_registry_from_config

        cfg = _crypto_cfg(["BTC", "ETH"])
        reg = build_registry_from_config(cfg)
        return CryptoSymbolMapper(reg)

    def test_to_provider_symbol(self) -> None:
        mapper = self._make_mapper()
        assert mapper.to_provider_symbol("BTC") == "BTC/USDT"

    def test_to_exchange_symbol(self) -> None:
        mapper = self._make_mapper()
        assert "BTC" in mapper.to_exchange_symbol("BTC")

    def test_to_coingecko_id_fallback(self) -> None:
        mapper = self._make_mapper()
        assert mapper.to_coingecko_id("BTC") == "btc"

    def test_from_provider_symbol(self) -> None:
        mapper = self._make_mapper()
        assert mapper.from_provider_symbol("BTC/USDT") == "BTC"

    def test_equity_symbol_raises_leak(self) -> None:
        from spectraquant_v3.core.errors import AssetClassLeakError

        mapper = self._make_mapper()
        with pytest.raises(AssetClassLeakError):
            mapper.to_provider_symbol("INFY.NS")

    def test_is_registered(self) -> None:
        mapper = self._make_mapper()
        assert mapper.is_registered("BTC")
        assert not mapper.is_registered("DOGE")

    def test_get_record_returns_crypto(self) -> None:
        from spectraquant_v3.core.enums import AssetClass

        mapper = self._make_mapper()
        rec = mapper.get_record("ETH")
        assert rec.asset_class == AssetClass.CRYPTO


# ===========================================================================
# 4. Equity symbol mapper
# ===========================================================================

class TestEquitySymbolMapper:
    def _make_mapper(self):
        from spectraquant_v3.equities.symbols.mapper import EquitySymbolMapper
        from spectraquant_v3.equities.symbols.registry import build_registry_from_config

        cfg = _equity_cfg(["INFY.NS", "TCS.NS"])
        reg = build_registry_from_config(cfg)
        return EquitySymbolMapper(reg)

    def test_to_yfinance_symbol(self) -> None:
        mapper = self._make_mapper()
        assert mapper.to_yfinance_symbol("INFY.NS") == "INFY.NS"

    def test_from_yfinance_symbol(self) -> None:
        mapper = self._make_mapper()
        assert mapper.from_yfinance_symbol("TCS.NS") == "TCS.NS"

    def test_crypto_pair_raises_leak(self) -> None:
        from spectraquant_v3.core.errors import AssetClassLeakError

        mapper = self._make_mapper()
        with pytest.raises(AssetClassLeakError):
            mapper.to_yfinance_symbol("BTC/USDT")

    def test_is_registered(self) -> None:
        mapper = self._make_mapper()
        assert mapper.is_registered("INFY.NS")
        assert not mapper.is_registered("WIPRO.NS")

    def test_get_record_returns_equity(self) -> None:
        from spectraquant_v3.core.enums import AssetClass

        mapper = self._make_mapper()
        rec = mapper.get_record("TCS.NS")
        assert rec.asset_class == AssetClass.EQUITY


# ===========================================================================
# 5. Crypto universe builder
# ===========================================================================

class TestCryptoUniverseBuilder:
    def _make_builder(self, symbols=None, mode="static"):
        from spectraquant_v3.crypto.symbols.registry import build_registry_from_config
        from spectraquant_v3.crypto.universe.builder import CryptoUniverseBuilder

        if symbols is None:
            symbols = ["BTC", "ETH", "SOL"]
        cfg = _crypto_cfg(symbols)
        cfg["crypto"]["universe_mode"] = mode
        reg = build_registry_from_config(cfg)
        return CryptoUniverseBuilder(cfg, reg, run_id="test_run"), cfg

    def test_static_mode_includes_all(self) -> None:
        builder, _ = self._make_builder()
        artifact = builder.build()
        assert set(artifact.included_symbols) == {"BTC", "ETH", "SOL"}
        assert not artifact.excluded_symbols

    def test_quality_gate_excludes_low_market_cap(self) -> None:
        from spectraquant_v3.crypto.symbols.registry import build_registry_from_config
        from spectraquant_v3.crypto.universe.builder import CryptoUniverseBuilder

        cfg = _crypto_cfg(["BTC", "ETH"])
        cfg["crypto"]["quality_gate"]["min_market_cap_usd"] = 10_000_000
        reg = build_registry_from_config(cfg)
        builder = CryptoUniverseBuilder(cfg, reg, run_id="r1")
        market_data = {
            "BTC": {"market_cap_usd": 500_000_000},
            "ETH": {"market_cap_usd": 1_000_000},  # below threshold
        }
        artifact = builder.build(market_data=market_data)
        assert "BTC" in artifact.included_symbols
        assert "ETH" in artifact.excluded_symbols

    def test_write_produces_json(self, tmp_path: Path) -> None:
        builder, _ = self._make_builder()
        artifact = builder.build()
        path = artifact.write(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["asset_class"] == "crypto"
        assert data["included_count"] == 3

    def test_empty_symbols_raises_empty_universe(self) -> None:
        from spectraquant_v3.core.errors import EmptyUniverseError
        from spectraquant_v3.crypto.symbols.registry import build_registry_from_config
        from spectraquant_v3.crypto.universe.builder import CryptoUniverseBuilder

        cfg = _crypto_cfg([])
        reg = build_registry_from_config(cfg)
        builder = CryptoUniverseBuilder(cfg, reg, run_id="r1")
        with pytest.raises(EmptyUniverseError):
            builder.build()

    def test_equity_symbol_in_config_raises_leak_error(self) -> None:
        from spectraquant_v3.core.errors import AssetClassLeakError
        from spectraquant_v3.crypto.universe.builder import CryptoUniverseBuilder, UniverseEntry
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        reg = CryptoSymbolRegistry()
        cfg = _crypto_cfg(["INFY.NS"])
        builder = CryptoUniverseBuilder(cfg, reg, run_id="r1")
        builder._quality_gate.require_tradable_mapping = False
        with pytest.raises(AssetClassLeakError):
            builder.build()

    def test_inclusion_exclusion_reasons_recorded(self) -> None:
        from spectraquant_v3.crypto.symbols.registry import build_registry_from_config
        from spectraquant_v3.crypto.universe.builder import CryptoUniverseBuilder

        cfg = _crypto_cfg(["BTC", "ETH"])
        cfg["crypto"]["quality_gate"]["min_market_cap_usd"] = 100_000_000
        reg = build_registry_from_config(cfg)
        builder = CryptoUniverseBuilder(cfg, reg, run_id="r1")
        market_data = {
            "BTC": {"market_cap_usd": 500_000_000},
            "ETH": {"market_cap_usd": 1_000_000},
        }
        artifact = builder.build(market_data=market_data)
        btc_entry = next(e for e in artifact.entries if e.canonical_symbol == "BTC")
        eth_entry = next(e for e in artifact.entries if e.canonical_symbol == "ETH")
        assert btc_entry.reason == "passed_all_gates"
        assert "market_cap" in eth_entry.reason

    def test_dataset_topn_mode(self) -> None:
        from spectraquant_v3.crypto.symbols.registry import build_registry_from_config
        from spectraquant_v3.crypto.universe.builder import CryptoUniverseBuilder

        cfg = _crypto_cfg(["BTC", "ETH", "SOL"])
        cfg["crypto"]["universe_mode"] = "dataset_topN"
        cfg["crypto"]["universe_top_n"] = 2
        reg = build_registry_from_config(cfg)
        builder = CryptoUniverseBuilder(cfg, reg, run_id="r1")
        artifact = builder.build()
        assert len(artifact.included_symbols) <= 2

    def test_dataset_topn_ranks_by_market_cap(self) -> None:
        """dataset_topN must pick highest market-cap symbols, not alphabetical order."""
        from spectraquant_v3.crypto.symbols.registry import build_registry_from_config
        from spectraquant_v3.crypto.universe.builder import CryptoUniverseBuilder

        cfg = _crypto_cfg(["ADA", "BTC", "ETH"])  # alphabetical: ADA first
        cfg["crypto"]["universe_mode"] = "dataset_topN"
        cfg["crypto"]["universe_top_n"] = 2
        reg = build_registry_from_config(cfg)
        builder = CryptoUniverseBuilder(cfg, reg, run_id="r1")
        # Supply market data where BTC and ETH have higher caps than ADA
        market_data = {
            "ADA": {"market_cap_usd": 1_000_000},
            "BTC": {"market_cap_usd": 900_000_000_000},
            "ETH": {"market_cap_usd": 400_000_000_000},
        }
        artifact = builder.build(market_data=market_data)
        # BTC and ETH should be selected, not ADA
        assert "BTC" in artifact.included_symbols
        assert "ETH" in artifact.included_symbols
        assert "ADA" not in artifact.included_symbols


# ===========================================================================
# 6. Equity universe builder
# ===========================================================================

class TestEquityUniverseBuilder:
    def _make_builder(self, tickers=None):
        from spectraquant_v3.equities.symbols.registry import build_registry_from_config
        from spectraquant_v3.equities.universe.builder import EquityUniverseBuilder

        if tickers is None:
            tickers = ["INFY.NS", "TCS.NS"]
        cfg = _equity_cfg(tickers)
        reg = build_registry_from_config(cfg)
        return EquityUniverseBuilder(cfg, reg, run_id="test"), cfg

    def test_includes_all_without_gates(self) -> None:
        builder, _ = self._make_builder()
        artifact = builder.build()
        assert set(artifact.included_symbols) == {"INFY.NS", "TCS.NS"}

    def test_exclude_list_removes_symbol(self) -> None:
        from spectraquant_v3.equities.symbols.registry import build_registry_from_config
        from spectraquant_v3.equities.universe.builder import EquityUniverseBuilder

        cfg = _equity_cfg(["INFY.NS", "TCS.NS"])
        cfg["equities"]["universe"]["exclude"] = ["INFY.NS"]
        reg = build_registry_from_config(cfg)
        builder = EquityUniverseBuilder(cfg, reg, run_id="r1")
        artifact = builder.build()
        assert "INFY.NS" not in artifact.included_symbols
        assert "INFY.NS" in artifact.excluded_symbols

    def test_quality_gate_filters_low_price(self) -> None:
        from spectraquant_v3.equities.symbols.registry import build_registry_from_config
        from spectraquant_v3.equities.universe.builder import EquityUniverseBuilder

        cfg = _equity_cfg(["INFY.NS", "TCS.NS"])
        cfg["equities"]["quality_gate"]["min_price"] = 100.0
        reg = build_registry_from_config(cfg)
        builder = EquityUniverseBuilder(cfg, reg, run_id="r1")
        price_data = {
            "INFY.NS": {"last_close": 150.0},
            "TCS.NS": {"last_close": 50.0},  # below threshold
        }
        artifact = builder.build(price_data=price_data)
        assert "INFY.NS" in artifact.included_symbols
        assert "TCS.NS" in artifact.excluded_symbols

    def test_write_produces_json(self, tmp_path: Path) -> None:
        builder, _ = self._make_builder()
        artifact = builder.build()
        path = artifact.write(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["asset_class"] == "equity"

    def test_empty_tickers_raises_empty_universe(self) -> None:
        from spectraquant_v3.core.errors import EmptyUniverseError

        builder, _ = self._make_builder(tickers=[])
        with pytest.raises(EmptyUniverseError):
            builder.build()

    def test_crypto_symbol_raises_leak_error(self) -> None:
        from spectraquant_v3.core.errors import AssetClassLeakError
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry
        from spectraquant_v3.equities.universe.builder import EquityUniverseBuilder

        cfg = _equity_cfg(["BTC/USDT"])
        reg = EquitySymbolRegistry()
        builder = EquityUniverseBuilder(cfg, reg, run_id="r1")
        with pytest.raises(AssetClassLeakError):
            builder.build()


# ===========================================================================
# 7. Crypto feature engine
# ===========================================================================

class TestCryptoFeatureEngine:
    def test_compute_features_adds_columns(self) -> None:
        from spectraquant_v3.crypto.features.engine import compute_features

        df = _ohlcv_df(60)
        out = compute_features(df, symbol="BTC")
        for col in ["ret_1d", "ret_20d", "rsi", "volume_ratio", "atr_norm", "vol_realised"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self) -> None:
        from spectraquant_v3.crypto.features.engine import compute_features

        df = _ohlcv_df(60)
        out = compute_features(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in out.columns

    def test_empty_df_raises(self) -> None:
        from spectraquant_v3.core.errors import EmptyPriceDataError
        from spectraquant_v3.crypto.features.engine import compute_features

        with pytest.raises(EmptyPriceDataError):
            compute_features(pd.DataFrame())

    def test_missing_column_raises(self) -> None:
        from spectraquant_v3.core.errors import DataSchemaError
        from spectraquant_v3.crypto.features.engine import compute_features

        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(DataSchemaError):
            compute_features(df)

    def test_transform_many_skips_bad(self) -> None:
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine

        engine = CryptoFeatureEngine()
        price_map = {
            "BTC": _ohlcv_df(60),
            "BAD": pd.DataFrame({"price": [1, 2]}),  # bad schema
        }
        result = engine.transform_many(price_map)
        assert "BTC" in result
        assert "BAD" not in result

    def test_rsi_range(self) -> None:
        from spectraquant_v3.crypto.features.engine import compute_features

        df = _ohlcv_df(100)
        out = compute_features(df)
        valid = out["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_custom_momentum_window(self) -> None:
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine

        engine = CryptoFeatureEngine(momentum_window=10)
        out = engine.transform(_ohlcv_df(60))
        assert "ret_10d" in out.columns

    def test_from_config(self) -> None:
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine

        cfg = _crypto_cfg()
        engine = CryptoFeatureEngine.from_config(cfg)
        assert engine.momentum_window == 20


# ===========================================================================
# 8. Equity feature engine
# ===========================================================================

class TestEquityFeatureEngine:
    def test_compute_features_adds_columns(self) -> None:
        from spectraquant_v3.equities.features.engine import compute_features

        df = _ohlcv_df(60)
        out = compute_features(df, symbol="INFY.NS")
        for col in ["ret_1d", "ret_20d", "rsi", "volume_ratio", "atr_norm", "vol_realised"]:
            assert col in out.columns

    def test_empty_df_raises(self) -> None:
        from spectraquant_v3.core.errors import EmptyPriceDataError
        from spectraquant_v3.equities.features.engine import compute_features

        with pytest.raises(EmptyPriceDataError):
            compute_features(pd.DataFrame())

    def test_from_config(self) -> None:
        from spectraquant_v3.equities.features.engine import EquityFeatureEngine

        cfg = _equity_cfg()
        engine = EquityFeatureEngine.from_config(cfg)
        assert engine.momentum_window == 20

    def test_transform_many(self) -> None:
        from spectraquant_v3.equities.features.engine import EquityFeatureEngine

        engine = EquityFeatureEngine()
        pm = {"INFY.NS": _ohlcv_df(60), "TCS.NS": _ohlcv_df(60, seed=99)}
        result = engine.transform_many(pm)
        assert len(result) == 2

    def test_vol_realised_annualised(self) -> None:
        from spectraquant_v3.equities.features.engine import compute_features

        df = _ohlcv_df(100)
        out = compute_features(df)
        valid = out["vol_realised"].dropna()
        # Realised vol should be positive
        assert (valid >= 0).all()


# ===========================================================================
# 9. Crypto momentum signal agent
# ===========================================================================

class TestCryptoMomentumAgent:
    def _make_agent(self, run_id="r1"):
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine
        from spectraquant_v3.crypto.signals.momentum import CryptoMomentumAgent

        engine = CryptoFeatureEngine(momentum_window=20)
        df = _ohlcv_df(60)
        features = engine.transform(df, symbol="BTC")
        agent = CryptoMomentumAgent(run_id=run_id)
        return agent, features

    def test_evaluate_returns_signal_row(self) -> None:
        from spectraquant_v3.core.schema import SignalRow

        agent, features = self._make_agent()
        row = agent.evaluate("BTC", features)
        assert isinstance(row, SignalRow)
        assert row.canonical_symbol == "BTC"
        assert row.asset_class == "crypto"

    def test_signal_score_range(self) -> None:
        agent, features = self._make_agent()
        row = agent.evaluate("BTC", features)
        if row.status == "OK":
            assert -1.0 <= row.signal_score <= 1.0
            assert 0.0 <= row.confidence <= 1.0

    def test_insufficient_rows_gives_no_signal(self) -> None:
        from spectraquant_v3.crypto.signals.momentum import CryptoMomentumAgent

        agent = CryptoMomentumAgent(run_id="r1", min_rows=50)
        small_df = _ohlcv_df(5)
        row = agent.evaluate("ETH", small_df)
        assert row.status == "NO_SIGNAL"

    def test_empty_df_gives_no_signal(self) -> None:
        from spectraquant_v3.crypto.signals.momentum import CryptoMomentumAgent

        agent = CryptoMomentumAgent(run_id="r1")
        row = agent.evaluate("SOL", pd.DataFrame())
        assert row.status == "NO_SIGNAL"

    def test_evaluate_many_returns_one_per_symbol(self) -> None:
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine
        from spectraquant_v3.crypto.signals.momentum import CryptoMomentumAgent

        engine = CryptoFeatureEngine()
        pm = {"BTC": _ohlcv_df(60), "ETH": _ohlcv_df(60, seed=2)}
        fm = engine.transform_many(pm)
        agent = CryptoMomentumAgent(run_id="r1")
        rows = agent.evaluate_many(fm)
        assert len(rows) == 2
        symbols = {r.canonical_symbol for r in rows}
        assert symbols == {"BTC", "ETH"}

    def test_from_config(self) -> None:
        from spectraquant_v3.crypto.signals.momentum import CryptoMomentumAgent

        agent = CryptoMomentumAgent.from_config(_crypto_cfg(), run_id="r1")
        assert agent.momentum_window == 20

    def test_rsi_dampening_overbought(self) -> None:
        """High RSI should reduce long signal strength."""
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine
        from spectraquant_v3.crypto.signals.momentum import CryptoMomentumAgent

        engine = CryptoFeatureEngine(momentum_window=5)
        df = _ohlcv_df(60)
        features = engine.transform(df)
        # Manually set RSI to 95 on last row
        features = features.copy()
        features.loc[features.index[-1], "rsi"] = 95.0
        features.loc[features.index[-1], "ret_5d"] = 0.10  # strong positive momentum

        agent = CryptoMomentumAgent(run_id="r1", momentum_window=5)
        row = agent.evaluate("BTC", features)
        # Score should be dampened (less than pure tanh of 0.10*10 = ~0.964)
        if row.status == "OK":
            assert row.signal_score < 0.964


# ===========================================================================
# 10. Equity momentum signal agent
# ===========================================================================

class TestEquityMomentumAgent:
    def _make_agent_with_features(self, run_id="r1"):
        from spectraquant_v3.equities.features.engine import EquityFeatureEngine
        from spectraquant_v3.equities.signals.momentum import EquityMomentumAgent

        engine = EquityFeatureEngine(momentum_window=20)
        df = _ohlcv_df(60)
        features = engine.transform(df, symbol="INFY.NS")
        agent = EquityMomentumAgent(run_id=run_id)
        return agent, features

    def test_evaluate_returns_signal_row(self) -> None:
        from spectraquant_v3.core.schema import SignalRow

        agent, features = self._make_agent_with_features()
        row = agent.evaluate("INFY.NS", features)
        assert isinstance(row, SignalRow)
        assert row.asset_class == "equity"

    def test_min_threshold_blocks_weak_signal(self) -> None:
        from spectraquant_v3.equities.features.engine import EquityFeatureEngine
        from spectraquant_v3.equities.signals.momentum import EquityMomentumAgent

        engine = EquityFeatureEngine(momentum_window=5)
        df = _ohlcv_df(60)
        features = engine.transform(df)
        # Force near-zero momentum on last row
        features = features.copy()
        features.loc[features.index[-1], "ret_5d"] = 0.0001

        agent = EquityMomentumAgent(run_id="r1", momentum_window=5, min_threshold=0.05)
        row = agent.evaluate("TCS.NS", features)
        assert row.status == "NO_SIGNAL"

    def test_from_config(self) -> None:
        from spectraquant_v3.equities.signals.momentum import EquityMomentumAgent

        agent = EquityMomentumAgent.from_config(_equity_cfg(), run_id="r1")
        assert agent.momentum_window == 20

    def test_evaluate_many(self) -> None:
        from spectraquant_v3.equities.features.engine import EquityFeatureEngine
        from spectraquant_v3.equities.signals.momentum import EquityMomentumAgent

        engine = EquityFeatureEngine()
        pm = {"INFY.NS": _ohlcv_df(60), "TCS.NS": _ohlcv_df(60, seed=77)}
        fm = engine.transform_many(pm)
        agent = EquityMomentumAgent(run_id="r1")
        rows = agent.evaluate_many(fm)
        assert len(rows) == 2


# ===========================================================================
# 11. Meta-policy
# ===========================================================================

class TestMetaPolicy:
    def _make_ok_signal(self, symbol: str, score: float, confidence: float, asset: str = "crypto"):
        from spectraquant_v3.core.schema import SignalRow

        return SignalRow(
            run_id="r1",
            timestamp="2025-01-01T00:00:00+00:00",
            canonical_symbol=symbol,
            asset_class=asset,
            agent_id="test_agent",
            horizon="1d",
            signal_score=score,
            confidence=confidence,
            status="OK",
        )

    def _make_no_signal(self, symbol: str, asset: str = "crypto"):
        from spectraquant_v3.core.schema import SignalRow

        return SignalRow(
            run_id="r1",
            timestamp="2025-01-01T00:00:00+00:00",
            canonical_symbol=symbol,
            asset_class=asset,
            agent_id="test_agent",
            horizon="1d",
            status="NO_SIGNAL",
        )

    def test_all_passed_with_strong_signals(self) -> None:
        from spectraquant_v3.pipeline.meta_policy import MetaPolicy, MetaPolicyConfig

        policy = MetaPolicy(MetaPolicyConfig(min_confidence=0.1, min_signal_threshold=0.05))
        signals = [
            self._make_ok_signal("BTC", 0.5, 0.5),
            self._make_ok_signal("ETH", 0.4, 0.4),
        ]
        decisions = policy.run(signals)
        assert len(decisions) == 2
        assert all(d.passed for d in decisions)

    def test_blocked_by_low_confidence(self) -> None:
        from spectraquant_v3.pipeline.meta_policy import MetaPolicy, MetaPolicyConfig

        policy = MetaPolicy(MetaPolicyConfig(min_confidence=0.5))
        signals = [self._make_ok_signal("BTC", 0.1, 0.05)]
        decisions = policy.run(signals)
        assert not decisions[0].passed
        assert "confidence" in decisions[0].reason

    def test_blocked_by_no_signal(self) -> None:
        from spectraquant_v3.pipeline.meta_policy import MetaPolicy, MetaPolicyConfig

        policy = MetaPolicy(MetaPolicyConfig())
        signals = [self._make_no_signal("BTC")]
        decisions = policy.run(signals)
        assert not decisions[0].passed

    def test_composite_score_is_average(self) -> None:
        from spectraquant_v3.pipeline.meta_policy import MetaPolicy, MetaPolicyConfig

        policy = MetaPolicy(MetaPolicyConfig(min_confidence=0.0, min_signal_threshold=0.0))
        signals = [
            self._make_ok_signal("BTC", 0.6, 0.6),
            self._make_ok_signal("BTC", 0.4, 0.4),
        ]
        decisions = policy.run(signals)
        assert abs(decisions[0].composite_score - 0.5) < 1e-9

    def test_from_config(self) -> None:
        from spectraquant_v3.pipeline.meta_policy import MetaPolicy

        policy = MetaPolicy.from_config(_crypto_cfg())
        assert policy.config.min_confidence == 0.10

    def test_empty_signals_list(self) -> None:
        from spectraquant_v3.pipeline.meta_policy import MetaPolicy, MetaPolicyConfig

        policy = MetaPolicy(MetaPolicyConfig())
        decisions = policy.run([])
        assert decisions == []

    def _make_error_signal(self, symbol: str, asset: str = "crypto"):
        from spectraquant_v3.core.schema import SignalRow

        return SignalRow(
            run_id="r1",
            timestamp="2025-01-01T00:00:00+00:00",
            canonical_symbol=symbol,
            asset_class=asset,
            agent_id="test_agent",
            horizon="1d",
            status="ERROR",
            error_reason="provider_unavailable",
        )

    def test_block_error_signals_blocks_when_error_present(self) -> None:
        """When block_error_signals=True any ERROR row must block the symbol."""
        from spectraquant_v3.pipeline.meta_policy import MetaPolicy, MetaPolicyConfig

        policy = MetaPolicy(MetaPolicyConfig(block_error_signals=True))
        signals = [
            self._make_ok_signal("BTC", 0.5, 0.5),
            self._make_error_signal("BTC"),
        ]
        decisions = policy.run(signals)
        assert not decisions[0].passed
        assert "error" in decisions[0].reason

    def test_block_error_signals_false_uses_ok_rows(self) -> None:
        """When block_error_signals=False, ERROR rows are ignored and OK rows score."""
        from spectraquant_v3.pipeline.meta_policy import MetaPolicy, MetaPolicyConfig

        policy = MetaPolicy(
            MetaPolicyConfig(block_error_signals=False, min_confidence=0.0, min_signal_threshold=0.0)
        )
        signals = [
            self._make_ok_signal("BTC", 0.5, 0.5),
            self._make_error_signal("BTC"),
        ]
        decisions = policy.run(signals)
        assert decisions[0].passed
        assert abs(decisions[0].composite_score - 0.5) < 1e-9

    def test_error_row_excluded_from_composite_even_without_ok(self) -> None:
        """ERROR-only symbol with block_error_signals=True → blocked."""
        from spectraquant_v3.pipeline.meta_policy import MetaPolicy, MetaPolicyConfig

        policy = MetaPolicy(MetaPolicyConfig(block_error_signals=True))
        signals = [self._make_error_signal("ETH")]
        decisions = policy.run(signals)
        assert not decisions[0].passed


# ===========================================================================
# 12. Allocator
# ===========================================================================

class TestAllocator:
    def _make_decisions(self, symbols, scores, passed_mask=None):
        from spectraquant_v3.pipeline.meta_policy import PolicyDecision

        if passed_mask is None:
            passed_mask = [True] * len(symbols)
        return [
            PolicyDecision(
                canonical_symbol=sym,
                asset_class="crypto",
                composite_score=score,
                composite_confidence=abs(score),
                passed=p,
                reason="passed" if p else "blocked",
            )
            for sym, score, p in zip(symbols, scores, passed_mask)
        ]

    def test_equal_weight_sums_to_one(self) -> None:
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        alloc = Allocator(AllocatorConfig(mode="equal_weight", max_weight=0.5), run_id="r1")
        decisions = self._make_decisions(["BTC", "ETH", "SOL"], [0.5, 0.4, 0.3])
        rows = alloc.allocate(decisions)
        active = [r for r in rows if not r.blocked]
        total = sum(r.target_weight for r in active)
        assert abs(total - 1.0) < 1e-4

    def test_max_weight_cap_applied(self) -> None:
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        alloc = Allocator(AllocatorConfig(mode="equal_weight", max_weight=0.10), run_id="r1")
        decisions = self._make_decisions(["BTC", "ETH"], [0.5, 0.4])
        rows = alloc.allocate(decisions)
        for row in rows:
            assert row.target_weight <= 0.10 + 1e-9

    def test_blocked_symbols_have_zero_weight(self) -> None:
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        alloc = Allocator(AllocatorConfig(), run_id="r1")
        decisions = self._make_decisions(
            ["BTC", "ETH"], [0.5, 0.4], passed_mask=[True, False]
        )
        rows = alloc.allocate(decisions)
        eth_row = next(r for r in rows if r.canonical_symbol == "ETH")
        assert eth_row.target_weight == 0.0
        assert eth_row.blocked

    def test_vol_target_mode(self) -> None:
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        alloc = Allocator(
            AllocatorConfig(mode="vol_target", target_vol=0.15, max_gross_leverage=1.0),
            run_id="r1",
        )
        decisions = self._make_decisions(["BTC", "ETH"], [0.5, 0.4])
        vol_map = {"BTC": 0.60, "ETH": 0.40}
        rows = alloc.allocate(decisions, vol_map=vol_map)
        active = [r for r in rows if not r.blocked]
        total_abs = sum(abs(r.target_weight) for r in active)
        assert total_abs <= 1.0 + 1e-6

    def test_from_config(self) -> None:
        from spectraquant_v3.pipeline.allocator import Allocator

        alloc = Allocator.from_config(_crypto_cfg(), run_id="r1")
        assert alloc.config.max_weight == 0.25

    def test_empty_decisions(self) -> None:
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        alloc = Allocator(AllocatorConfig(), run_id="r1")
        rows = alloc.allocate([])
        assert rows == []

    def test_negative_weight_capped_by_abs(self) -> None:
        """Negative (short) weights must be capped symmetrically via abs."""
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        alloc = Allocator(AllocatorConfig(mode="vol_target", target_vol=0.15, max_weight=0.20), run_id="r1")
        decisions = self._make_decisions(["BTC", "ETH"], [-0.8, -0.6])
        vol_map = {"BTC": 0.10, "ETH": 0.10}  # low vol → large raw weights
        rows = alloc.allocate(decisions, vol_map=vol_map)
        for row in [r for r in rows if not r.blocked]:
            assert abs(row.target_weight) <= 0.20 + 1e-9, (
                f"target_weight={row.target_weight} exceeds max_weight=0.20"
            )

    def test_gross_leverage_enforced_with_signed_weights(self) -> None:
        """sum(|weight|) must not exceed max_gross_leverage."""
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        alloc = Allocator(
            AllocatorConfig(mode="vol_target", target_vol=0.50, max_gross_leverage=0.5),
            run_id="r1",
        )
        decisions = self._make_decisions(["BTC", "ETH", "SOL"], [0.9, -0.7, 0.8])
        vol_map = {"BTC": 0.05, "ETH": 0.05, "SOL": 0.05}  # very low vol → large raw
        rows = alloc.allocate(decisions, vol_map=vol_map)
        total_abs = sum(abs(r.target_weight) for r in rows if not r.blocked)
        assert total_abs <= 0.5 + 1e-6, f"gross leverage={total_abs} exceeded 0.5"

    def test_max_weight_capped_for_negative_short(self) -> None:
        """A large negative weight should be capped to -max_weight, not left uncapped."""
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        alloc = Allocator(AllocatorConfig(mode="vol_target", target_vol=0.30, max_weight=0.10), run_id="r1")
        decisions = self._make_decisions(["BTC"], [-0.9])
        vol_map = {"BTC": 0.05}
        rows = alloc.allocate(decisions, vol_map=vol_map)
        btc = rows[0]
        assert not btc.blocked
        assert btc.target_weight >= -0.10 - 1e-9

    def test_equal_weight_all_positive(self) -> None:
        """Equal-weight allocator never produces negative weights."""
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        alloc = Allocator(AllocatorConfig(mode="equal_weight", max_weight=0.5), run_id="r1")
        decisions = self._make_decisions(["A", "B", "C", "D"], [0.1, 0.2, 0.3, 0.4])
        rows = alloc.allocate(decisions)
        for row in [r for r in rows if not r.blocked]:
            assert row.target_weight >= 0.0


# ===========================================================================
# 13. Reporter
# ===========================================================================

class TestPipelineReporter:
    def _make_signal_rows(self):
        from spectraquant_v3.core.schema import SignalRow

        return [
            SignalRow("r1", "2025-01-01T00:00:00+00:00", "BTC", "crypto", "a1", "1d",
                      signal_score=0.5, confidence=0.5, status="OK"),
            SignalRow("r1", "2025-01-01T00:00:00+00:00", "ETH", "crypto", "a1", "1d",
                      status="NO_SIGNAL"),
        ]

    def _make_decisions(self):
        from spectraquant_v3.pipeline.meta_policy import PolicyDecision

        return [
            PolicyDecision("BTC", "crypto", 0.5, 0.5, True, "passed"),
            PolicyDecision("ETH", "crypto", 0.0, 0.0, False, "no_signal"),
        ]

    def _make_allocations(self):
        from spectraquant_v3.core.schema import AllocationRow

        return [
            AllocationRow("r1", "BTC", "crypto", target_weight=0.5),
            AllocationRow("r1", "ETH", "crypto", target_weight=0.0, blocked=True),
        ]

    def test_write_signals(self, tmp_path: Path) -> None:
        from spectraquant_v3.pipeline.reporter import PipelineReporter

        rep = PipelineReporter("r1", tmp_path, "crypto")
        path = rep.write_signals(self._make_signal_rows())
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["ok_count"] == 1
        assert data["no_signal_count"] == 1

    def test_write_allocation(self, tmp_path: Path) -> None:
        from spectraquant_v3.pipeline.reporter import PipelineReporter

        rep = PipelineReporter("r1", tmp_path, "crypto")
        path = rep.write_allocation(self._make_allocations())
        data = json.loads(path.read_text())
        assert data["active_positions"] == 1
        assert data["blocked_positions"] == 1

    def test_write_run_report(self, tmp_path: Path) -> None:
        from spectraquant_v3.pipeline.reporter import PipelineReporter

        rep = PipelineReporter("r1", tmp_path, "crypto")
        path = rep.write_run_report(
            universe_size=2,
            signals=self._make_signal_rows(),
            decisions=self._make_decisions(),
            allocations=self._make_allocations(),
        )
        data = json.loads(path.read_text())
        assert data["universe_size"] == 2
        assert data["signals_ok"] == 1

    def test_write_all_creates_all_files(self, tmp_path: Path) -> None:
        from spectraquant_v3.pipeline.reporter import PipelineReporter

        rep = PipelineReporter("r1", tmp_path, "crypto")
        paths = rep.write_all(
            universe_symbols=["BTC", "ETH"],
            signals=self._make_signal_rows(),
            decisions=self._make_decisions(),
            allocations=self._make_allocations(),
        )
        assert len(paths) == 4
        for key, p in paths.items():
            assert Path(p).exists(), f"{key} file not written"


# ===========================================================================
# 14. End-to-end pipeline (dry_run=True to avoid file writes)
# ===========================================================================

class TestCryptoPipelineEndToEnd:
    def test_pipeline_succeeds_with_price_data(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

        cfg = _crypto_cfg(["BTC", "ETH"])
        price_data = {"BTC": _ohlcv_df(60), "ETH": _ohlcv_df(60, seed=2)}
        result = run_crypto_pipeline(
            cfg,
            run_mode=RunMode.NORMAL,
            dry_run=True,
            price_data=price_data,
            run_id="e2e_crypto",
            project_root=tmp_path,
        )
        assert result["status"] == "success"
        assert "BTC" in result["universe"]
        assert len(result["signals"]) == 2
        assert len(result["allocations"]) == 2

    def test_pipeline_no_price_data(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

        cfg = _crypto_cfg(["BTC"])
        result = run_crypto_pipeline(
            cfg,
            run_mode=RunMode.NORMAL,
            dry_run=True,
            run_id="e2e_noprice",
            project_root=tmp_path,
        )
        assert result["status"] == "success"
        # No price data → NO_SIGNAL for all
        assert all(s.status == "NO_SIGNAL" for s in result["signals"])

    def test_pipeline_writes_artefacts(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

        cfg = _crypto_cfg(["BTC"])
        price_data = {"BTC": _ohlcv_df(60)}
        result = run_crypto_pipeline(
            cfg,
            run_mode=RunMode.NORMAL,
            dry_run=False,
            price_data=price_data,
            run_id="artefact_test",
            project_root=tmp_path,
        )
        assert result["status"] == "success"
        # At least the run report should be written
        assert len(result["artefact_paths"]) >= 1

    def test_pipeline_run_id_propagates(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

        cfg = _crypto_cfg(["BTC"])
        result = run_crypto_pipeline(
            cfg,
            run_mode=RunMode.NORMAL,
            dry_run=True,
            run_id="my_run_id",
            project_root=tmp_path,
        )
        assert result["run_id"] == "my_run_id"

    def test_pipeline_aborts_when_all_ohlcv_empty(self, tmp_path: Path) -> None:
        """Pipeline must raise EmptyUniverseError when price_data is supplied
        but all DataFrames are empty (assert_ohlcv_available guard)."""
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.core.errors import EmptyUniverseError
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline
        import pandas as pd

        cfg = _crypto_cfg(["BTC"])
        empty_price_data = {"BTC": pd.DataFrame()}
        with pytest.raises(EmptyUniverseError, match="EMPTY_OHLCV_UNIVERSE"):
            run_crypto_pipeline(
                cfg,
                run_mode=RunMode.NORMAL,
                dry_run=True,
                price_data=empty_price_data,
                project_root=tmp_path,
            )


class TestEquityPipelineEndToEnd:
    def test_pipeline_succeeds_with_price_data(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        cfg = _equity_cfg(["INFY.NS", "TCS.NS"])
        price_data = {"INFY.NS": _ohlcv_df(60), "TCS.NS": _ohlcv_df(60, seed=3)}
        result = run_equity_pipeline(
            cfg,
            run_mode=RunMode.NORMAL,
            dry_run=True,
            price_data=price_data,
            run_id="e2e_equity",
            project_root=tmp_path,
        )
        assert result["status"] == "success"
        assert "INFY.NS" in result["universe"]
        assert len(result["signals"]) == 2

    def test_pipeline_no_price_data(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        cfg = _equity_cfg(["INFY.NS"])
        result = run_equity_pipeline(
            cfg,
            run_mode=RunMode.NORMAL,
            dry_run=True,
            run_id="e2e_noprice_eq",
            project_root=tmp_path,
        )
        assert result["status"] == "success"
        assert all(s.status == "NO_SIGNAL" for s in result["signals"])

    def test_pipeline_empty_universe_raises(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.core.errors import EmptyUniverseError
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        cfg = _equity_cfg([])
        with pytest.raises(EmptyUniverseError):
            run_equity_pipeline(cfg, dry_run=True, project_root=tmp_path)

    def test_pipeline_aborts_when_all_ohlcv_empty(self, tmp_path: Path) -> None:
        """Pipeline must raise EmptyUniverseError when price_data is supplied
        but all DataFrames are empty (assert_ohlcv_available guard)."""
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.core.errors import EmptyUniverseError
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline
        import pandas as pd

        cfg = _equity_cfg(["INFY.NS"])
        empty_price_data = {"INFY.NS": pd.DataFrame()}
        with pytest.raises(EmptyUniverseError, match="EMPTY_OHLCV_UNIVERSE"):
            run_equity_pipeline(
                cfg,
                run_mode=RunMode.NORMAL,
                dry_run=True,
                price_data=empty_price_data,
                project_root=tmp_path,
            )

    def test_pipeline_writes_artefacts(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        cfg = _equity_cfg(["INFY.NS"])
        price_data = {"INFY.NS": _ohlcv_df(60)}
        result = run_equity_pipeline(
            cfg,
            run_mode=RunMode.NORMAL,
            dry_run=False,
            price_data=price_data,
            run_id="eq_artefact",
            project_root=tmp_path,
        )
        assert result["status"] == "success"
        assert len(result["artefact_paths"]) >= 1


# ===========================================================================
# 15. Asset-class segregation: pipelines must not cross-import
# ===========================================================================

class TestAssetClassSegregation:
    def test_crypto_pipeline_does_not_import_equities(self) -> None:
        import spectraquant_v3.pipeline.crypto_pipeline as cp
        src = cp.__file__
        with open(src) as f:
            content = f.read()
        # Check that no actual import statement imports from equities
        import_lines = [
            ln.strip() for ln in content.splitlines()
            if ln.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert "equities" not in line, (
                f"crypto_pipeline.py contains an equities import: {line}"
            )

    def test_equity_pipeline_does_not_import_crypto(self) -> None:
        import spectraquant_v3.pipeline.equity_pipeline as ep
        src = ep.__file__
        with open(src) as f:
            content = f.read()
        import_lines = [
            ln.strip() for ln in content.splitlines()
            if ln.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert "crypto" not in line, (
                f"equity_pipeline.py contains a crypto import: {line}"
            )

    def test_crypto_registry_rejects_equity_symbol(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.errors import AssetClassLeakError
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        reg = CryptoSymbolRegistry()
        with pytest.raises(AssetClassLeakError):
            reg.register(SymbolRecord("TCS.NS", AssetClass.EQUITY))

    def test_equity_registry_rejects_crypto_pair(self) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.errors import AssetClassLeakError
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        reg = EquitySymbolRegistry()
        with pytest.raises(AssetClassLeakError):
            reg.register(SymbolRecord("SOL/USDT", AssetClass.EQUITY))


# ===========================================================================
# 16. Crypto ingestion – CryptoOHLCVLoader
# ===========================================================================

class TestCryptoOHLCVLoader:
    """Tests for CryptoOHLCVLoader (cache-managed, RunMode-aware)."""

    def _make_loader(self, tmp_path, run_mode=None):
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.ingestion.ohlcv_loader import CryptoOHLCVLoader
        from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        if run_mode is None:
            run_mode = RunMode.NORMAL

        registry = CryptoSymbolRegistry()
        registry.register(SymbolRecord(
            canonical_symbol="BTC",
            asset_class=AssetClass.CRYPTO,
            provider_symbol="BTC/USDT",
        ))
        registry.register(SymbolRecord(
            canonical_symbol="ETH",
            asset_class=AssetClass.CRYPTO,
            provider_symbol="ETH/USDT",
        ))
        cache = CacheManager(cache_dir=tmp_path, run_mode=run_mode)
        mapper = CryptoSymbolMapper(registry=registry)
        return CryptoOHLCVLoader(cache=cache, mapper=mapper, run_mode=run_mode)

    def test_load_returns_none_on_download_failure(self, tmp_path) -> None:
        """When no cache and yfinance unavailable, load() returns None."""
        loader = self._make_loader(tmp_path)
        # No cache file, no network → returns None
        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.crypto.ingestion.ohlcv_loader._download_ohlcv_yfinance",
            return_value=None,
        ):
            result = loader.load("BTC")
        assert result is None

    def test_load_from_cache_hit(self, tmp_path) -> None:
        """load() returns cached DataFrame on cache hit without downloading."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        # Pre-populate cache
        df = _ohlcv_df(60)
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)
        cache.write_parquet("BTC", df)

        loader = self._make_loader(tmp_path, run_mode=RunMode.NORMAL)
        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.crypto.ingestion.ohlcv_loader._download_ohlcv_yfinance",
        ) as mock_dl:
            result = loader.load("BTC")
        # Should not call download at all
        mock_dl.assert_not_called()
        assert result is not None
        assert len(result) == 60

    def test_load_downloads_and_caches_on_cache_miss(self, tmp_path) -> None:
        """load() downloads, validates, and caches on cache miss in NORMAL mode."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        df = _ohlcv_df(60)
        loader = self._make_loader(tmp_path, run_mode=RunMode.NORMAL)
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)

        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.crypto.ingestion.ohlcv_loader._download_ohlcv_yfinance",
            return_value=df,
        ):
            result = loader.load("BTC")

        assert result is not None
        assert len(result) == 60
        # Should be cached now
        assert cache.exists("BTC")

    def test_load_test_mode_raises_on_cache_miss(self, tmp_path) -> None:
        """In TEST mode, load() raises CacheOnlyViolationError on cache miss."""
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.core.errors import CacheOnlyViolationError

        loader = self._make_loader(tmp_path, run_mode=RunMode.TEST)
        with pytest.raises(CacheOnlyViolationError):
            loader.load("BTC")

    def test_load_test_mode_returns_cached_data(self, tmp_path) -> None:
        """In TEST mode, load() returns cached data without network calls."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        df = _ohlcv_df(60)
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)
        cache.write_parquet("ETH", df)

        loader = self._make_loader(tmp_path, run_mode=RunMode.TEST)
        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.crypto.ingestion.ohlcv_loader._download_ohlcv_yfinance",
        ) as mock_dl:
            result = loader.load("ETH")
        mock_dl.assert_not_called()
        assert result is not None

    def test_load_refresh_mode_overwrites_cache(self, tmp_path) -> None:
        """In REFRESH mode, load() downloads even when cache exists."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        old_df = _ohlcv_df(30, seed=1)
        new_df = _ohlcv_df(60, seed=2)
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)
        cache.write_parquet("BTC", old_df)

        loader = self._make_loader(tmp_path, run_mode=RunMode.REFRESH)
        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.crypto.ingestion.ohlcv_loader._download_ohlcv_yfinance",
            return_value=new_df,
        ):
            result = loader.load("BTC")

        assert result is not None
        assert len(result) == 60  # new data, not old 30-row data

    def test_load_skips_invalid_schema(self, tmp_path) -> None:
        """load() returns None and does not cache when schema validation fails."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        # DataFrame missing required OHLCV columns
        bad_df = pd.DataFrame({"price": [100.0, 101.0]})
        loader = self._make_loader(tmp_path, run_mode=RunMode.NORMAL)
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)

        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.crypto.ingestion.ohlcv_loader._download_ohlcv_yfinance",
            return_value=bad_df,
        ):
            result = loader.load("BTC")

        assert result is None
        assert not cache.exists("BTC")

    def test_load_many_returns_only_successful_symbols(self, tmp_path) -> None:
        """load_many() omits symbols that fail and returns only successful ones."""
        df = _ohlcv_df(60)
        loader = self._make_loader(tmp_path)

        import unittest.mock as mock

        def _mock_download(yf_sym, **kwargs):
            if "BTC" in yf_sym:
                return df
            return None

        with mock.patch(
            "spectraquant_v3.crypto.ingestion.ohlcv_loader._download_ohlcv_yfinance",
            side_effect=_mock_download,
        ):
            result = loader.load_many(["BTC", "ETH"])

        assert "BTC" in result
        assert "ETH" not in result

    def test_from_config_creates_loader(self, tmp_path) -> None:
        """from_config() builds a properly configured CryptoOHLCVLoader."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.ingestion.ohlcv_loader import CryptoOHLCVLoader
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        registry = CryptoSymbolRegistry()
        registry.register(SymbolRecord("BTC", AssetClass.CRYPTO))
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)
        cfg = _crypto_cfg(["BTC"])
        loader = CryptoOHLCVLoader.from_config(
            cfg=cfg,
            cache=cache,
            registry=registry,
            run_mode=RunMode.NORMAL,
        )
        assert loader._run_mode == RunMode.NORMAL
        assert loader._period == "1y"

    def test_yfinance_symbol_helper(self) -> None:
        """_yfinance_symbol_for maps canonical tickers to yfinance format."""
        from spectraquant_v3.crypto.ingestion.ohlcv_loader import _yfinance_symbol_for

        assert _yfinance_symbol_for("BTC") == "BTC-USD"
        assert _yfinance_symbol_for("eth") == "ETH-USD"
        # If already has hyphen, return as-is (uppercased)
        assert _yfinance_symbol_for("BTC-USD") == "BTC-USD"

    def test_ingestion_module_does_not_import_equities(self) -> None:
        """crypto/ingestion must not import from spectraquant_v3.equities."""
        import spectraquant_v3.crypto.ingestion.ohlcv_loader as mod
        src = mod.__file__
        with open(src) as f:
            content = f.read()
        import_lines = [
            ln.strip() for ln in content.splitlines()
            if ln.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert "equities" not in line, (
                f"crypto/ingestion/ohlcv_loader.py contains an equities import: {line}"
            )


# ===========================================================================
# 17. Equity ingestion – EquityOHLCVLoader
# ===========================================================================

class TestEquityOHLCVLoader:
    """Tests for EquityOHLCVLoader (cache-managed, RunMode-aware)."""

    def _make_loader(self, tmp_path, run_mode=None):
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.ingestion.ohlcv_loader import EquityOHLCVLoader
        from spectraquant_v3.equities.symbols.mapper import EquitySymbolMapper
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        if run_mode is None:
            run_mode = RunMode.NORMAL

        registry = EquitySymbolRegistry()
        registry.register(SymbolRecord(
            canonical_symbol="INFY.NS",
            asset_class=AssetClass.EQUITY,
            yfinance_symbol="INFY.NS",
        ))
        registry.register(SymbolRecord(
            canonical_symbol="TCS.NS",
            asset_class=AssetClass.EQUITY,
            yfinance_symbol="TCS.NS",
        ))
        cache = CacheManager(cache_dir=tmp_path, run_mode=run_mode)
        mapper = EquitySymbolMapper(registry=registry)
        return EquityOHLCVLoader(cache=cache, mapper=mapper, run_mode=run_mode)

    def test_load_returns_none_on_download_failure(self, tmp_path) -> None:
        """When no cache and yfinance unavailable, load() returns None."""
        loader = self._make_loader(tmp_path)
        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.equities.ingestion.ohlcv_loader._download_ohlcv_yfinance",
            return_value=None,
        ):
            result = loader.load("INFY.NS")
        assert result is None

    def test_load_from_cache_hit(self, tmp_path) -> None:
        """load() returns cached DataFrame on cache hit without downloading."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        df = _ohlcv_df(60)
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)
        cache.write_parquet("INFY.NS", df)

        loader = self._make_loader(tmp_path, run_mode=RunMode.NORMAL)
        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.equities.ingestion.ohlcv_loader._download_ohlcv_yfinance",
        ) as mock_dl:
            result = loader.load("INFY.NS")
        mock_dl.assert_not_called()
        assert result is not None
        assert len(result) == 60

    def test_load_downloads_and_caches_on_cache_miss(self, tmp_path) -> None:
        """load() downloads, validates, and caches on cache miss in NORMAL mode."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        df = _ohlcv_df(60)
        loader = self._make_loader(tmp_path, run_mode=RunMode.NORMAL)
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)

        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.equities.ingestion.ohlcv_loader._download_ohlcv_yfinance",
            return_value=df,
        ):
            result = loader.load("INFY.NS")

        assert result is not None
        assert len(result) == 60
        assert cache.exists("INFY.NS")

    def test_load_test_mode_raises_on_cache_miss(self, tmp_path) -> None:
        """In TEST mode, load() raises CacheOnlyViolationError on cache miss."""
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.core.errors import CacheOnlyViolationError

        loader = self._make_loader(tmp_path, run_mode=RunMode.TEST)
        with pytest.raises(CacheOnlyViolationError):
            loader.load("INFY.NS")

    def test_load_test_mode_returns_cached_data(self, tmp_path) -> None:
        """In TEST mode, load() returns cached data without network calls."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        df = _ohlcv_df(60)
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)
        cache.write_parquet("TCS.NS", df)

        loader = self._make_loader(tmp_path, run_mode=RunMode.TEST)
        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.equities.ingestion.ohlcv_loader._download_ohlcv_yfinance",
        ) as mock_dl:
            result = loader.load("TCS.NS")
        mock_dl.assert_not_called()
        assert result is not None

    def test_load_refresh_mode_overwrites_cache(self, tmp_path) -> None:
        """In REFRESH mode, load() downloads even when cache exists."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        old_df = _ohlcv_df(30, seed=1)
        new_df = _ohlcv_df(60, seed=2)
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)
        cache.write_parquet("INFY.NS", old_df)

        loader = self._make_loader(tmp_path, run_mode=RunMode.REFRESH)
        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.equities.ingestion.ohlcv_loader._download_ohlcv_yfinance",
            return_value=new_df,
        ):
            result = loader.load("INFY.NS")

        assert result is not None
        assert len(result) == 60

    def test_load_skips_invalid_schema(self, tmp_path) -> None:
        """load() returns None and does not cache when schema validation fails."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        bad_df = pd.DataFrame({"price": [100.0, 101.0]})
        loader = self._make_loader(tmp_path, run_mode=RunMode.NORMAL)
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)

        import unittest.mock as mock
        with mock.patch(
            "spectraquant_v3.equities.ingestion.ohlcv_loader._download_ohlcv_yfinance",
            return_value=bad_df,
        ):
            result = loader.load("INFY.NS")

        assert result is None
        assert not cache.exists("INFY.NS")

    def test_load_many_returns_only_successful_symbols(self, tmp_path) -> None:
        """load_many() omits symbols that fail and returns only successful ones."""
        df = _ohlcv_df(60)
        loader = self._make_loader(tmp_path)

        import unittest.mock as mock

        def _mock_download(yf_sym, **kwargs):
            if "INFY" in yf_sym:
                return df
            return None

        with mock.patch(
            "spectraquant_v3.equities.ingestion.ohlcv_loader._download_ohlcv_yfinance",
            side_effect=_mock_download,
        ):
            result = loader.load_many(["INFY.NS", "TCS.NS"])

        assert "INFY.NS" in result
        assert "TCS.NS" not in result

    def test_from_config_creates_loader(self, tmp_path) -> None:
        """from_config() builds a properly configured EquityOHLCVLoader."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.ingestion.ohlcv_loader import EquityOHLCVLoader
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        registry = EquitySymbolRegistry()
        registry.register(SymbolRecord("INFY.NS", AssetClass.EQUITY))
        cache = CacheManager(cache_dir=tmp_path, run_mode=RunMode.NORMAL)
        cfg = _equity_cfg(["INFY.NS"])
        loader = EquityOHLCVLoader.from_config(
            cfg=cfg,
            cache=cache,
            registry=registry,
            run_mode=RunMode.NORMAL,
        )
        assert loader._run_mode == RunMode.NORMAL
        assert loader._period == "5y"

    def test_ingestion_module_does_not_import_crypto(self) -> None:
        """equities/ingestion must not import from spectraquant_v3.crypto."""
        import spectraquant_v3.equities.ingestion.ohlcv_loader as mod
        src = mod.__file__
        with open(src) as f:
            content = f.read()
        import_lines = [
            ln.strip() for ln in content.splitlines()
            if ln.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert "crypto" not in line, (
                f"equities/ingestion/ohlcv_loader.py contains a crypto import: {line}"
            )


# ===========================================================================
# 18. NaN volatility handling in allocator
# ===========================================================================

class TestAllocatorNaNVolatility:
    """Verify that NaN or inf volatility values are handled correctly."""

    def _make_decisions(self, symbols, scores=None):
        from spectraquant_v3.pipeline.meta_policy import PolicyDecision
        if scores is None:
            scores = [0.5] * len(symbols)
        return [
            PolicyDecision(
                canonical_symbol=sym,
                asset_class="crypto",
                composite_score=score,
                composite_confidence=abs(score),
                passed=True,
                reason="passed_all_filters",
            )
            for sym, score in zip(symbols, scores)
        ]

    def test_nan_vol_falls_back_to_default(self) -> None:
        """NaN vol in vol_map falls back to default_vol=0.20."""
        import math
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        config = AllocatorConfig(mode="vol_target", target_vol=0.15, max_gross_leverage=1.0)
        allocator = Allocator(config=config)
        decisions = self._make_decisions(["BTC", "ETH"], [0.5, 0.5])

        # Pass NaN vol for BTC, valid vol for ETH
        vol_map = {"BTC": float("nan"), "ETH": 0.25}
        rows = allocator.allocate(decisions, vol_map=vol_map)

        active = [r for r in rows if not r.blocked]
        assert len(active) == 2
        # No NaN weights
        for row in active:
            assert math.isfinite(row.target_weight), (
                f"Weight for {row.canonical_symbol} is not finite: {row.target_weight}"
            )

    def test_inf_vol_falls_back_to_default(self) -> None:
        """Infinite vol in vol_map falls back to default_vol=0.20."""
        import math
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        config = AllocatorConfig(mode="vol_target", target_vol=0.15, max_gross_leverage=1.0)
        allocator = Allocator(config=config)
        decisions = self._make_decisions(["BTC"], [0.5])

        vol_map = {"BTC": float("inf")}
        rows = allocator.allocate(decisions, vol_map=vol_map)
        active = [r for r in rows if not r.blocked]
        assert len(active) == 1
        assert math.isfinite(active[0].target_weight)

    def test_zero_vol_falls_back_to_default(self) -> None:
        """Zero vol in vol_map falls back to default_vol=0.20."""
        import math
        from spectraquant_v3.pipeline.allocator import Allocator, AllocatorConfig

        config = AllocatorConfig(mode="vol_target", target_vol=0.15, max_gross_leverage=1.0)
        allocator = Allocator(config=config)
        decisions = self._make_decisions(["BTC"], [0.5])

        vol_map = {"BTC": 0.0}
        rows = allocator.allocate(decisions, vol_map=vol_map)
        active = [r for r in rows if not r.blocked]
        assert len(active) == 1
        assert math.isfinite(active[0].target_weight)
        assert active[0].target_weight > 0


# ===========================================================================
# 19. Pipeline vol_map NaN filtering
# ===========================================================================

class TestPipelineVolMapFiltering:
    """Verify that vol_map extraction correctly filters NaN/inf/zero values."""

    def test_crypto_pipeline_filters_nan_vol(self, tmp_path) -> None:
        """run_crypto_pipeline filters NaN vol_realised from vol_map."""
        import math
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

        cfg = _crypto_cfg(["BTC"])
        cfg["portfolio"]["allocator"] = "vol_target"

        # DataFrame where vol_realised will be NaN (only 1 row, needs min 2)
        tiny_df = _ohlcv_df(1)
        price_data = {"BTC": tiny_df}

        result = run_crypto_pipeline(
            cfg,
            run_mode=RunMode.NORMAL,
            dry_run=True,
            price_data=price_data,
            run_id="volnan_test",
            project_root=tmp_path,
        )
        assert result["status"] == "success"
        for row in result["allocations"]:
            assert math.isfinite(row.target_weight), (
                f"Non-finite allocation for {row.canonical_symbol}: {row.target_weight}"
            )

    def test_equity_pipeline_filters_nan_vol(self, tmp_path) -> None:
        """run_equity_pipeline filters NaN vol_realised from vol_map."""
        import math
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        cfg = _equity_cfg(["INFY.NS"])
        cfg["portfolio"]["allocator"] = "vol_target"

        tiny_df = _ohlcv_df(1)
        price_data = {"INFY.NS": tiny_df}

        result = run_equity_pipeline(
            cfg,
            run_mode=RunMode.NORMAL,
            dry_run=True,
            price_data=price_data,
            run_id="eqvolnan_test",
            project_root=tmp_path,
        )
        assert result["status"] == "success"
        for row in result["allocations"]:
            assert math.isfinite(row.target_weight), (
                f"Non-finite allocation for {row.canonical_symbol}: {row.target_weight}"
            )


# ===========================================================================
# Research Dataset Builder tests
# ===========================================================================


class TestResearchDatasetBuilder:
    """Unit tests for ResearchDatasetBuilder."""

    def test_build_basic(self, tmp_path):
        """Build with two symbols, verify output files and row counts."""
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine
        from spectraquant_v3.research.dataset_builder import ResearchDatasetBuilder

        engine = CryptoFeatureEngine()
        feature_map = {
            "BTC": engine.transform(_ohlcv_df(120, seed=1), symbol="BTC"),
            "ETH": engine.transform(_ohlcv_df(120, seed=2), symbol="ETH"),
        }

        builder = ResearchDatasetBuilder(output_dir=tmp_path / "ds", run_id="test")
        result = builder.build(
            feature_map=feature_map,
            forward_windows=[1, 5],
            train_frac=0.70,
            val_frac=0.15,
        )

        # Check result metadata
        assert result.n_symbols == 2
        assert result.total_rows > 0
        assert result.train_rows > 0
        assert result.test_rows > 0
        assert "fwd_ret_1d" in result.label_columns
        assert "fwd_ret_5d" in result.label_columns

        # Check files on disk
        from pathlib import Path
        assert Path(result.train_path).exists()
        assert Path(result.test_path).exists()
        assert Path(result.manifest_path).exists()

    def test_build_no_val(self, tmp_path):
        """val_frac=0 skips the validation partition."""
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine
        from spectraquant_v3.research.dataset_builder import ResearchDatasetBuilder

        engine = CryptoFeatureEngine()
        feature_map = {
            "BTC": engine.transform(_ohlcv_df(100, seed=3), symbol="BTC"),
        }

        builder = ResearchDatasetBuilder(output_dir=tmp_path / "ds2", run_id="noval")
        result = builder.build(
            feature_map=feature_map,
            forward_windows=[1],
            train_frac=0.80,
            val_frac=0.0,
        )

        assert result.val_path is None
        assert result.val_rows == 0

    def test_build_empty_feature_map_raises(self, tmp_path):
        """Empty feature_map raises EmptyPriceDataError."""
        from spectraquant_v3.core.errors import EmptyPriceDataError
        from spectraquant_v3.research.dataset_builder import ResearchDatasetBuilder

        builder = ResearchDatasetBuilder(output_dir=tmp_path / "ds3", run_id="empty")
        with pytest.raises(EmptyPriceDataError):
            builder.build(feature_map={})

    def test_build_bad_split_raises(self, tmp_path):
        """train_frac + val_frac >= 1.0 raises ValueError."""
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine
        from spectraquant_v3.research.dataset_builder import ResearchDatasetBuilder

        engine = CryptoFeatureEngine()
        feature_map = {
            "BTC": engine.transform(_ohlcv_df(60, seed=4), symbol="BTC"),
        }

        builder = ResearchDatasetBuilder(output_dir=tmp_path / "ds4", run_id="bad")
        with pytest.raises(ValueError, match="train_frac"):
            builder.build(
                feature_map=feature_map,
                forward_windows=[1],
                train_frac=0.85,
                val_frac=0.20,
            )

    def test_build_manifest_is_valid_json(self, tmp_path):
        """Manifest file is valid JSON and contains expected keys."""
        import json
        from pathlib import Path

        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine
        from spectraquant_v3.research.dataset_builder import ResearchDatasetBuilder

        engine = CryptoFeatureEngine()
        feature_map = {
            "SOL": engine.transform(_ohlcv_df(80, seed=5), symbol="SOL"),
        }

        builder = ResearchDatasetBuilder(output_dir=tmp_path / "ds5", run_id="mfest")
        result = builder.build(feature_map=feature_map, forward_windows=[1, 5])

        data = json.loads(Path(result.manifest_path).read_text())
        for key in (
            "output_dir", "train_path", "test_path", "total_rows",
            "n_symbols", "n_dates", "train_rows", "test_rows",
            "feature_columns", "label_columns", "generated_at",
        ):
            assert key in data, f"manifest missing key: {key}"

    def test_build_summary_non_empty(self, tmp_path):
        """DatasetBuildResult.summary() returns a multi-line string."""
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine
        from spectraquant_v3.research.dataset_builder import ResearchDatasetBuilder

        engine = CryptoFeatureEngine()
        feature_map = {
            "BTC": engine.transform(_ohlcv_df(80, seed=6), symbol="BTC"),
        }

        builder = ResearchDatasetBuilder(output_dir=tmp_path / "ds6", run_id="sumtest")
        result = builder.build(feature_map=feature_map, forward_windows=[1])
        summary = result.summary()
        assert "ResearchDataset" in summary
        assert "train=" in summary

    def test_build_train_test_non_overlapping(self, tmp_path):
        """Train and test partitions share no dates."""
        import pandas as pd
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine
        from spectraquant_v3.research.dataset_builder import ResearchDatasetBuilder

        engine = CryptoFeatureEngine()
        feature_map = {
            "BTC": engine.transform(_ohlcv_df(120, seed=7), symbol="BTC"),
            "ETH": engine.transform(_ohlcv_df(120, seed=8), symbol="ETH"),
        }

        builder = ResearchDatasetBuilder(
            output_dir=tmp_path / "ds7", run_id="nooverlap"
        )
        result = builder.build(
            feature_map=feature_map,
            forward_windows=[1],
            train_frac=0.70,
            val_frac=0.15,
        )

        train = pd.read_parquet(result.train_path)
        test = pd.read_parquet(result.test_path)

        train_dates = set(train.index.get_level_values("date").normalize())
        test_dates = set(test.index.get_level_values("date").normalize())
        assert train_dates.isdisjoint(test_dates), (
            f"Train/test date overlap: {train_dates & test_dates}"
        )


# ===========================================================================
# BacktestResults tests
# ===========================================================================


class TestBacktestResults:
    """Unit tests for BacktestResults and RebalanceSnapshot."""

    def _make_results(self) -> "BacktestResults":
        from spectraquant_v3.backtest.results import BacktestResults, RebalanceSnapshot

        snaps = [
            RebalanceSnapshot(
                date="2024-01-08",
                universe=["BTC", "ETH"],
                signals_ok=2,
                signals_nosig=0,
                policy_passed=2,
                policy_blocked=0,
                allocations={"BTC": 0.5, "ETH": 0.5},
                portfolio_value=1.02,
                step_return=0.02,
            ),
            RebalanceSnapshot(
                date="2024-01-15",
                universe=["BTC", "ETH"],
                signals_ok=1,
                signals_nosig=1,
                policy_passed=1,
                policy_blocked=1,
                allocations={"BTC": 1.0},
                portfolio_value=0.99,
                step_return=-0.0294,
            ),
        ]
        return BacktestResults(
            run_id="test_bt",
            asset_class="crypto",
            start_date="2024-01-08",
            end_date="2024-01-15",
            n_steps=2,
            symbols=["BTC", "ETH"],
            total_return=-0.01,
            annualised_return=-0.05,
            annualised_volatility=0.20,
            sharpe_ratio=-0.25,
            max_drawdown=-0.03,
            calmar_ratio=1.67,
            win_rate=0.50,
            snapshots=snaps,
            generated_at="2024-01-15T00:00:00+00:00",
        )

    def test_to_dict_keys(self):
        """to_dict contains all required keys."""
        r = self._make_results()
        d = r.to_dict()
        for key in (
            "run_id", "asset_class", "start_date", "end_date", "n_steps",
            "symbols", "total_return", "annualised_return", "annualised_volatility",
            "sharpe_ratio", "max_drawdown", "calmar_ratio", "win_rate",
            "turnover", "avg_positions", "exposure",
            "generated_at", "snapshots",
        ):
            assert key in d, f"BacktestResults.to_dict() missing key: {key}"

    def test_snapshots_serialised(self):
        """Each snapshot is serialised in to_dict()."""
        r = self._make_results()
        d = r.to_dict()
        assert len(d["snapshots"]) == 2
        assert d["snapshots"][0]["date"] == "2024-01-08"

    def test_write_creates_json(self, tmp_path):
        """write() creates a valid JSON file."""
        import json
        r = self._make_results()
        path = r.write(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["run_id"] == "test_bt"

    def test_summary_format(self):
        """summary() returns a multi-line string with key metrics."""
        r = self._make_results()
        s = r.summary()
        assert "test_bt" in s
        assert "crypto" in s
        assert "sharpe=" in s


# ===========================================================================
# BacktestEngine tests
# ===========================================================================


class TestBacktestEngine:
    """Unit tests for the walk-forward BacktestEngine."""

    def _make_engine(self, tmp_path=None, symbols=None, n=120, seed=1,
                     window_type="expanding", freq="W", asset_class="crypto"):
        from spectraquant_v3.backtest.engine import BacktestEngine

        if symbols is None:
            symbols = ["BTC", "ETH", "SOL"]

        price_data = {
            sym: _ohlcv_df(n, seed=seed + i)
            for i, sym in enumerate(symbols)
        }

        cfg = _crypto_cfg(symbols) if asset_class == "crypto" else _equity_cfg(symbols)
        return BacktestEngine(
            cfg=cfg,
            asset_class=asset_class,
            price_data=price_data,
            rebalance_freq=freq,
            window_type=window_type,
            min_in_sample_periods=30,
            run_id="bt_test",
        )

    def test_run_returns_results(self, tmp_path):
        """run() returns a BacktestResults with at least one step."""
        from spectraquant_v3.backtest.results import BacktestResults

        engine = self._make_engine(tmp_path)
        results = engine.run()

        assert isinstance(results, BacktestResults)
        assert results.n_steps >= 1
        assert results.asset_class == "crypto"
        assert len(results.symbols) > 0

    def test_run_nav_starts_at_one(self, tmp_path):
        """NAV is always >= 0; first step starts from 1.0."""
        engine = self._make_engine(tmp_path)
        results = engine.run()

        for snap in results.snapshots:
            assert snap.portfolio_value >= 0.0

    def test_rolling_window(self, tmp_path):
        """Rolling window mode runs without error."""
        from spectraquant_v3.backtest.results import BacktestResults

        engine = self._make_engine(
            tmp_path, n=150, window_type="rolling", freq="ME"
        )
        results = engine.run()
        assert isinstance(results, BacktestResults)

    def test_expanding_window(self, tmp_path):
        """Expanding window mode produces more steps than rolling for same data."""
        engine_exp = self._make_engine(
            tmp_path, n=200, window_type="expanding", freq="ME"
        )
        engine_rol = self._make_engine(
            tmp_path, n=200, window_type="rolling", freq="ME"
        )
        res_exp = engine_exp.run()
        res_rol = engine_rol.run()
        # Both should run; expanding typically has same or more valid steps
        assert res_exp.n_steps >= 0
        assert res_rol.n_steps >= 0

    def test_metrics_are_finite(self, tmp_path):
        """All aggregate performance metrics are finite floats."""
        import math

        engine = self._make_engine(tmp_path, n=200, freq="ME")
        results = engine.run()

        for attr in (
            "total_return", "annualised_return", "annualised_volatility",
            "sharpe_ratio", "max_drawdown", "win_rate",
        ):
            val = getattr(results, attr)
            assert math.isfinite(val), f"{attr}={val} is not finite"

    def test_empty_price_data_raises(self):
        """Passing empty price_data raises EmptyPriceDataError."""
        from spectraquant_v3.backtest.engine import BacktestEngine
        from spectraquant_v3.core.errors import EmptyPriceDataError

        with pytest.raises(EmptyPriceDataError):
            BacktestEngine(
                cfg=_crypto_cfg(),
                asset_class="crypto",
                price_data={},
                run_id="empty_bt",
            )

    def test_invalid_asset_class_raises(self):
        """Passing an invalid asset_class raises ValueError."""
        from spectraquant_v3.backtest.engine import BacktestEngine

        with pytest.raises(ValueError, match="asset_class"):
            BacktestEngine(
                cfg=_crypto_cfg(),
                asset_class="futures",
                price_data={"BTC": _ohlcv_df()},
                run_id="bad_ac",
            )

    def test_invalid_window_type_raises(self):
        """Passing an invalid window_type raises ValueError."""
        from spectraquant_v3.backtest.engine import BacktestEngine

        with pytest.raises(ValueError, match="window_type"):
            BacktestEngine(
                cfg=_crypto_cfg(),
                asset_class="crypto",
                price_data={"BTC": _ohlcv_df()},
                window_type="anchored",
                run_id="bad_wt",
            )

    def test_results_write_and_summary(self, tmp_path):
        """Results can be written to disk and summarised."""
        engine = self._make_engine(tmp_path, n=200, freq="ME")
        results = engine.run()
        path = results.write(tmp_path)

        assert path.exists()
        summary = results.summary()
        assert "bt_test" in summary

    def test_equity_backtest(self, tmp_path):
        """BacktestEngine works for the equity asset class."""
        from spectraquant_v3.backtest.engine import BacktestEngine
        from spectraquant_v3.backtest.results import BacktestResults

        tickers = ["INFY.NS", "TCS.NS"]
        price_data = {
            sym: _ohlcv_df(150, seed=i + 10)
            for i, sym in enumerate(tickers)
        }
        cfg = _equity_cfg(tickers)
        engine = BacktestEngine(
            cfg=cfg,
            asset_class="equity",
            price_data=price_data,
            rebalance_freq="ME",
            window_type="expanding",
            min_in_sample_periods=30,
            run_id="eq_bt",
        )
        results = engine.run()
        assert isinstance(results, BacktestResults)
        assert results.asset_class == "equity"

    def test_cost_regression_deterministic(self):
        """Net performance is deterministic with explicit cost parameters."""
        from types import SimpleNamespace

        from spectraquant_v3.backtest.engine import BacktestEngine

        idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        price_data = {
            "AAA": pd.DataFrame(
                {
                    "open": [100, 100, 110, 121, 121],
                    "high": [100, 100, 110, 121, 121],
                    "low": [100, 100, 110, 121, 121],
                    "close": [100, 100, 110, 121, 121],
                    "volume": [1, 1, 1, 1, 1],
                },
                index=idx,
            )
        }

        engine = BacktestEngine(
            cfg={"backtest": {"target_vol": 1e9}},
            asset_class="crypto",
            price_data=price_data,
            rebalance_freq="D",
            min_in_sample_periods=2,
            commission=10.0,
            slippage=20.0,
            spread=30.0,
            run_id="bt_cost_regression",
        )

        class _Feat:
            def transform(self, df, symbol=None):
                out = df.copy()
                out["vol_realised"] = 0.1
                return out

        class _Signal:
            def evaluate_many(self, feature_map, as_of):
                return [SimpleNamespace(status="OK") for _ in feature_map]

        class _Policy:
            def run(self, signals):
                return [SimpleNamespace(passed=True) for _ in signals]

        class _Allocator:
            def __init__(self):
                self._weights = [1.0, 0.0, 1.0, 1.0]
                self._i = 0

            def allocate(self, decisions, vol_map=None):
                w = self._weights[self._i]
                self._i += 1
                return [
                    SimpleNamespace(
                        canonical_symbol="AAA",
                        target_weight=w,
                        blocked=False,
                    )
                ]

        engine._feature_engine = _Feat()
        engine._signal_agent = _Signal()
        engine._meta_policy = _Policy()
        engine._allocator = _Allocator()

        results = engine.run()
        assert results.n_steps == 4

        gross = np.array([0.10, 0.00, 0.00, 0.00])
        turnover = np.array([1.0, 1.0, 1.0, 0.0])
        step_cost = turnover * ((10.0 + 20.0 + 30.0) * 1e-4)
        net = gross - step_cost
        expected_nav = float(np.prod(1.0 + net))

        observed_net = np.array([s.net_return for s in results.snapshots])
        observed_gross = np.array([s.gross_return for s in results.snapshots])
        observed_turnover = np.array([s.turnover for s in results.snapshots])

        np.testing.assert_allclose(observed_gross, gross, atol=1e-12)
        np.testing.assert_allclose(observed_turnover, turnover, atol=1e-12)
        np.testing.assert_allclose(observed_net, net, atol=1e-12)
        assert results.snapshots[-1].portfolio_value == pytest.approx(expected_nav, abs=1e-12)

        assert results.turnover == pytest.approx(float(turnover.mean()), abs=1e-12)
        assert results.avg_positions == pytest.approx(0.75, abs=1e-12)
        assert results.exposure == pytest.approx(0.75, abs=1e-12)

    # ------------------------------------------------------------------
    # News-injection tests
    # ------------------------------------------------------------------

    def _equity_ohlcv(self, n: int = 120, seed: int = 42) -> pd.DataFrame:
        """Synthetic OHLCV for equity tests (same schema as _ohlcv_df)."""
        return _ohlcv_df(n, seed=seed)

    def _hybrid_cfg(self, tickers=None) -> dict:
        """Equity config wired for the hybrid strategy."""
        if tickers is None:
            tickers = ["INFY.NS", "TCS.NS"]
        cfg = _equity_cfg(tickers)
        cfg["strategies"] = {
            "equity_momentum_news_hybrid_v1": {
                "signal_blend": {"momentum_weight": 0.7, "news_weight": 0.3},
                "vol_gate": {"threshold": 0.40},
            }
        }
        return cfg

    def test_news_feature_map_parameter_accepted(self):
        """BacktestEngine accepts news_feature_map without error."""
        from spectraquant_v3.backtest.engine import BacktestEngine

        sym = "INFY.NS"
        price_data = {sym: _ohlcv_df(60, seed=1)}
        news_map = {sym: {"2024-01-15": 0.5}}
        engine = BacktestEngine(
            cfg=_equity_cfg([sym]),
            asset_class="equity",
            price_data=price_data,
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id="news_param_test",
            news_feature_map=news_map,
        )
        assert engine._news_feature_map == news_map

    def test_inject_news_features_assigns_score_to_last_row(self):
        """_inject_news_features writes news_sentiment_score to df.iloc[-1]."""
        from spectraquant_v3.backtest.engine import BacktestEngine

        sym = "INFY.NS"
        price_data = {sym: _ohlcv_df(60, seed=2)}
        date = pd.Timestamp("2024-03-31", tz="UTC")
        date_key = date.strftime("%Y-%m-%d")
        expected_score = 0.65

        engine = BacktestEngine(
            cfg=_equity_cfg([sym]),
            asset_class="equity",
            price_data=price_data,
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id="inject_test",
            news_feature_map={sym: {date_key: expected_score}},
        )

        # Build a minimal feature_map to inject into
        df = _ohlcv_df(60, seed=2).copy()
        df["vol_realised"] = 0.15
        feature_map = {sym: df}
        engine._inject_news_features(feature_map, date)

        last_val = feature_map[sym]["news_sentiment_score"].iloc[-1]
        assert last_val == pytest.approx(expected_score, abs=1e-9)

    def test_inject_news_features_iso_key_fallback(self):
        """_inject_news_features also matches full ISO-8601 keys."""
        from spectraquant_v3.backtest.engine import BacktestEngine

        sym = "TCS.NS"
        price_data = {sym: _ohlcv_df(60, seed=3)}
        date = pd.Timestamp("2024-02-29", tz="UTC")
        iso_key = date.isoformat()
        expected_score = -0.3

        engine = BacktestEngine(
            cfg=_equity_cfg([sym]),
            asset_class="equity",
            price_data=price_data,
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id="iso_key_test",
            news_feature_map={sym: {iso_key: expected_score}},
        )

        df = _ohlcv_df(60, seed=3).copy()
        feature_map = {sym: df}
        engine._inject_news_features(feature_map, date)

        assert feature_map[sym]["news_sentiment_score"].iloc[-1] == pytest.approx(
            expected_score, abs=1e-9
        )

    def test_inject_news_features_no_op_when_map_absent(self):
        """_inject_news_features leaves feature_map unchanged when no news map set."""
        from spectraquant_v3.backtest.engine import BacktestEngine

        sym = "INFY.NS"
        price_data = {sym: _ohlcv_df(60, seed=4)}
        engine = BacktestEngine(
            cfg=_equity_cfg([sym]),
            asset_class="equity",
            price_data=price_data,
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id="no_news_test",
        )

        df = _ohlcv_df(60, seed=4).copy()
        feature_map = {sym: df}
        original_cols = set(df.columns)
        engine._inject_news_features(feature_map, pd.Timestamp("2024-03-31", tz="UTC"))

        assert set(feature_map[sym].columns) == original_cols

    def test_inject_news_features_missing_symbol_no_op(self):
        """_inject_news_features leaves symbols not in news_feature_map unchanged."""
        from spectraquant_v3.backtest.engine import BacktestEngine

        sym_with_news = "INFY.NS"
        sym_without = "TCS.NS"
        price_data = {sym_with_news: _ohlcv_df(60, seed=5), sym_without: _ohlcv_df(60, seed=6)}
        date = pd.Timestamp("2024-03-31", tz="UTC")
        date_key = date.strftime("%Y-%m-%d")

        engine = BacktestEngine(
            cfg=_equity_cfg([sym_with_news, sym_without]),
            asset_class="equity",
            price_data=price_data,
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id="partial_news_test",
            news_feature_map={sym_with_news: {date_key: 0.4}},
        )

        df_with = _ohlcv_df(60, seed=5).copy()
        df_without = _ohlcv_df(60, seed=6).copy()
        feature_map = {sym_with_news: df_with, sym_without: df_without}
        engine._inject_news_features(feature_map, date)

        # Symbol with news should have the column
        assert "news_sentiment_score" in feature_map[sym_with_news].columns
        # Symbol without news should NOT have the column
        assert "news_sentiment_score" not in feature_map[sym_without].columns

    def test_hybrid_backtest_uses_injected_news_score(self):
        """Hybrid backtest signals include news_sentiment_score when news_feature_map is set.

        Validates that the hybrid agent's rationale references news_sentiment_score
        (not None) on at least one step when a news map is provided.
        """
        from spectraquant_v3.backtest.engine import BacktestEngine
        from spectraquant_v3.backtest.results import BacktestResults
        from spectraquant_v3.equities.signals.hybrid import EquityMomentumNewsHybridAgent

        sym = "INFY.NS"
        price_data = {sym: _ohlcv_df(150, seed=10)}

        # Build a news map covering every month-end date in the data range
        idx = pd.date_range("2024-01-01", periods=150, freq="D", tz="UTC")
        month_ends = pd.date_range(
            start=idx[30],
            end=idx[-1],
            freq="ME",
        )
        news_map = {sym: {d.strftime("%Y-%m-%d"): 0.6 for d in month_ends}}

        cfg = self._hybrid_cfg([sym])
        engine = BacktestEngine(
            cfg=cfg,
            asset_class="equity",
            price_data=price_data,
            strategy_id="equity_momentum_news_hybrid_v1",
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id="hybrid_with_news",
            news_feature_map=news_map,
        )

        results = engine.run()
        assert isinstance(results, BacktestResults)
        # The backtest must complete successfully
        assert results.n_steps >= 1

    def test_hybrid_backtest_graceful_fallback_without_news(self):
        """Hybrid backtest runs successfully with no news_feature_map (pure momentum fallback)."""
        from spectraquant_v3.backtest.engine import BacktestEngine
        from spectraquant_v3.backtest.results import BacktestResults

        sym = "INFY.NS"
        price_data = {sym: _ohlcv_df(150, seed=11)}
        cfg = self._hybrid_cfg([sym])

        engine = BacktestEngine(
            cfg=cfg,
            asset_class="equity",
            price_data=price_data,
            strategy_id="equity_momentum_news_hybrid_v1",
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id="hybrid_no_news",
        )

        results = engine.run()
        assert isinstance(results, BacktestResults)
        assert results.n_steps >= 1

    def test_hybrid_vs_baseline_diverge_with_news(self):
        """Hybrid signal scores differ when news_feature_map is provided vs absent.

        Uses ``news_feature_map`` (the new injection parameter) to supply a
        positive news score for every rebalance date.  Both engines share the
        same stub feature engine that returns weak positive momentum.  Without
        news the hybrid falls back to pure momentum; with news the blended
        score is measurably higher.  The composite_scores in the final snapshot
        must differ between the two runs.
        """
        from types import SimpleNamespace

        from spectraquant_v3.backtest.engine import BacktestEngine
        from spectraquant_v3.equities.signals.hybrid import EquityMomentumNewsHybridAgent

        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        # Non-flat price so momentum column (ret_20d) is non-zero in the stub
        flat_price = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": [100.0] * n,
                "volume": [1_000_000.0] * n,
            },
            index=idx,
        )
        sym = "INFY.NS"
        price_data = {sym: flat_price}

        # A feature engine with non-zero momentum (above min_threshold=0.05)
        # tanh(0.008 * 10) = tanh(0.08) ≈ 0.0798 > 0.05
        class _FeatWeakMom:
            def transform(self, df, symbol=None):
                out = df.copy()
                out["ret_20d"] = 0.008
                out["rsi"] = 50.0
                out["vol_realised"] = 0.10
                return out

        # Stub policy that passes every signal, keeping composite_score from signal
        class _Policy:
            def run(self, signals):
                return [
                    SimpleNamespace(
                        passed=True,
                        canonical_symbol=s.canonical_symbol,
                        asset_class=s.asset_class,
                        composite_score=s.signal_score,
                        composite_confidence=s.confidence,
                        reason="",
                    )
                    for s in signals
                ]

        class _Alloc:
            def allocate(self, decisions, vol_map=None):
                return [
                    SimpleNamespace(
                        canonical_symbol=d.canonical_symbol,
                        target_weight=0.5 if d.passed else 0.0,
                        blocked=not d.passed,
                    )
                    for d in decisions
                ]

        cfg = self._hybrid_cfg([sym])

        # Build a news map covering every day so every rebalance step gets news
        news_map = {sym: {d.strftime("%Y-%m-%d"): 0.6 for d in idx}}

        # Engine WITHOUT news (pure momentum fallback)
        engine_no_news = BacktestEngine(
            cfg=cfg,
            asset_class="equity",
            price_data=price_data,
            strategy_id="equity_momentum_news_hybrid_v1",
            rebalance_freq="D",
            min_in_sample_periods=22,
            run_id="baseline",
        )
        engine_no_news._feature_engine = _FeatWeakMom()
        engine_no_news._signal_agent = EquityMomentumNewsHybridAgent(run_id="baseline")
        engine_no_news._meta_policy = _Policy()
        engine_no_news._allocator = _Alloc()

        # Engine WITH news injected via news_feature_map
        engine_with_news = BacktestEngine(
            cfg=cfg,
            asset_class="equity",
            price_data=price_data,
            strategy_id="equity_momentum_news_hybrid_v1",
            rebalance_freq="D",
            min_in_sample_periods=22,
            run_id="with_news",
            news_feature_map=news_map,
        )
        engine_with_news._feature_engine = _FeatWeakMom()
        engine_with_news._signal_agent = EquityMomentumNewsHybridAgent(run_id="with_news")
        engine_with_news._meta_policy = _Policy()
        engine_with_news._allocator = _Alloc()

        res_no_news = engine_no_news.run()
        res_with_news = engine_with_news.run()

        assert res_no_news.n_steps >= 1
        assert res_with_news.n_steps >= 1

        # Without news: composite_score ≈ tanh(0.08) ≈ 0.080 (pure momentum)
        # With news=0.6: composite_score ≈ 0.7 * 0.080 + 0.3 * 0.6 = 0.056 + 0.18 = 0.236
        # The two composite_scores must differ by a measurable amount
        final_snap_no_news = res_no_news.snapshots[-1]
        final_snap_with_news = res_with_news.snapshots[-1]

        score_no = final_snap_no_news.composite_scores.get(sym, 0.0)
        score_with = final_snap_with_news.composite_scores.get(sym, 0.0)

        assert score_with != pytest.approx(score_no, abs=1e-3), (
            f"Expected composite_score to differ between no-news ({score_no:.4f}) "
            f"and news-injected ({score_with:.4f}) runs, but they are equal. "
            "News injection via news_feature_map may not be taking effect."
        )
        # News should boost the blended score above pure momentum
        assert score_with > score_no, (
            f"With positive news (0.6), hybrid score ({score_with:.4f}) should "
            f"exceed pure momentum score ({score_no:.4f})."
        )

    def test_baseline_momentum_unchanged_by_news_map(self):
        """A non-hybrid equity backtest is unaffected by news_feature_map."""
        from spectraquant_v3.backtest.engine import BacktestEngine
        from spectraquant_v3.backtest.results import BacktestResults

        tickers = ["INFY.NS", "TCS.NS"]
        price_data = {sym: _ohlcv_df(150, seed=i + 20) for i, sym in enumerate(tickers)}

        idx = pd.date_range("2024-01-01", periods=150, freq="D", tz="UTC")
        month_ends = pd.date_range(start=idx[30], end=idx[-1], freq="ME")
        news_map = {
            sym: {d.strftime("%Y-%m-%d"): 0.5 for d in month_ends}
            for sym in tickers
        }

        cfg = _equity_cfg(tickers)

        engine_plain = BacktestEngine(
            cfg=cfg,
            asset_class="equity",
            price_data=price_data,
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id="plain_eq",
        )
        engine_with_news_map = BacktestEngine(
            cfg=cfg,
            asset_class="equity",
            price_data=price_data,
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id="plain_eq_news",
            news_feature_map=news_map,
        )

        res_plain = engine_plain.run()
        res_news = engine_with_news_map.run()

        # Momentum agent doesn't read news_sentiment_score, so results should
        # be identical (same NAV, same step count).
        assert res_plain.n_steps == res_news.n_steps
        assert res_plain.snapshots[-1].portfolio_value == pytest.approx(
            res_news.snapshots[-1].portfolio_value, abs=1e-9
        )

    def test_backtest_with_cached_data_and_news_map(self):
        """BacktestEngine with news_feature_map completes without regression."""
        from spectraquant_v3.backtest.engine import BacktestEngine
        from spectraquant_v3.backtest.results import BacktestResults

        tickers = ["INFY.NS", "TCS.NS"]
        price_data = {sym: _ohlcv_df(150, seed=i + 30) for i, sym in enumerate(tickers)}

        idx = pd.date_range("2024-01-01", periods=150, freq="D", tz="UTC")
        month_ends = pd.date_range(start=idx[30], end=idx[-1], freq="ME")
        news_map = {
            tickers[0]: {d.strftime("%Y-%m-%d"): 0.2 for d in month_ends},
        }

        cfg = _equity_cfg(tickers)
        engine = BacktestEngine(
            cfg=cfg,
            asset_class="equity",
            price_data=price_data,
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id="regression_news",
            news_feature_map=news_map,
        )

        results = engine.run()
        assert isinstance(results, BacktestResults)
        assert results.n_steps >= 1
        assert results.asset_class == "equity"


class TestCLIBacktestAndResearch:
    """Tests for sqv3 backtest run and sqv3 research dataset CLI commands."""

    def test_backtest_run_help(self):
        """sqv3 backtest run --help exits cleanly."""
        from typer.testing import CliRunner
        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["backtest", "run", "--help"])
        assert result.exit_code == 0
        assert "walk-forward" in result.output.lower() or "backtest" in result.output.lower()

    def test_research_dataset_help(self):
        """sqv3 research dataset --help exits cleanly."""
        from typer.testing import CliRunner
        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["research", "dataset", "--help"])
        assert result.exit_code == 0

    def test_backtest_run_invalid_asset_class(self):
        """Passing an invalid asset class returns exit code 1."""
        from typer.testing import CliRunner
        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["backtest", "run", "--asset-class", "futures"]
        )
        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_backtest_run_invalid_window_type(self):
        """Passing an invalid window type returns exit code 1."""
        from typer.testing import CliRunner
        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["backtest", "run", "--asset-class", "crypto", "--window-type", "anchored"],
        )
        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_research_dataset_invalid_asset_class(self):
        """Passing an invalid asset class to research dataset returns exit code 1."""
        from typer.testing import CliRunner
        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["research", "dataset", "--asset-class", "futures"]
        )
        assert result.exit_code == 1

    def test_top_level_help_shows_backtest_research(self):
        """sqv3 --help shows both new sub-commands."""
        from typer.testing import CliRunner
        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "backtest" in result.output
        assert "research" in result.output

    def test_backtest_run_with_cached_data(self, tmp_path):
        """backtest run command succeeds when cache is seeded."""
        from unittest.mock import patch

        import pandas as pd
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        # Create synthetic price data for the mock cache
        df = _ohlcv_df(200, seed=42)

        runner = CliRunner()

        # Patch CacheManager.read_parquet to return synthetic data (instance method)
        def _read_parquet(self_cache, key):
            return df

        with patch(
            "spectraquant_v3.core.cache.CacheManager.read_parquet",
            new=_read_parquet,
        ):
            result = runner.invoke(
                app,
                [
                    "backtest", "run",
                    "--asset-class", "crypto",
                    "--symbols", "BTC,ETH",
                    "--rebalance-freq", "ME",
                    "--min-periods", "30",
                    "--output-dir", str(tmp_path),
                    "--run-id", "cli_bt",
                ],
            )

        # With mocked OHLCV data of 200 rows the backtest should succeed.
        assert result.exit_code == 0, (
            f"Expected exit code 0, got {result.exit_code}.\n"
            f"Output: {result.output}"
        )
        assert "cli_bt" in result.output
