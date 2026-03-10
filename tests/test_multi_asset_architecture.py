"""Tests for the multi-asset architecture.

Validates the non-negotiable segregation rules:
- Crypto symbols cannot enter the equity pipeline.
- Equity symbols cannot enter the crypto pipeline.
- SymbolResolutionError is raised on wrong-class resolution.
- AssetClassLeakError is raised on cross-class contamination.
"""
from __future__ import annotations

import pytest

from spectraquant.core.enums import AssetClass, RunMode, SignalStatus
from spectraquant.core.errors import (
    AssetClassLeakError,
    CacheOnlyViolationError,
    EmptyOHLCVError,
    EmptyUniverseError,
    MixedAssetRunError,
    SymbolResolutionError,
)


# ---------------------------------------------------------------------------
# AssetClass enum
# ---------------------------------------------------------------------------

class TestAssetClassEnum:
    def test_crypto_value(self):
        assert AssetClass.CRYPTO.value == "crypto"
        assert str(AssetClass.CRYPTO) == "crypto"

    def test_equity_value(self):
        assert AssetClass.EQUITY.value == "equity"
        assert str(AssetClass.EQUITY) == "equity"

    def test_both_values_distinct(self):
        assert AssetClass.CRYPTO != AssetClass.EQUITY


# ---------------------------------------------------------------------------
# RunMode enum
# ---------------------------------------------------------------------------

class TestRunModeEnum:
    def test_all_modes_present(self):
        assert RunMode.NORMAL.value == "normal"
        assert RunMode.TEST.value == "test"
        assert RunMode.REFRESH.value == "refresh"


# ---------------------------------------------------------------------------
# Core error types
# ---------------------------------------------------------------------------

class TestSymbolResolutionError:
    def test_basic_message(self):
        err = SymbolResolutionError("INFY.NS", expected_asset_class=AssetClass.CRYPTO)
        assert "INFY.NS" in str(err)
        assert "crypto" in str(err)

    def test_with_actual_class(self):
        err = SymbolResolutionError(
            "INFY.NS",
            expected_asset_class=AssetClass.CRYPTO,
            actual_asset_class=AssetClass.EQUITY,
        )
        assert "equity" in str(err)

    def test_attributes_set(self):
        err = SymbolResolutionError("BTC", expected_asset_class=AssetClass.EQUITY)
        assert err.symbol == "BTC"
        assert err.expected_asset_class == AssetClass.EQUITY


class TestAssetClassLeakError:
    def test_message_contains_class_info(self):
        err = AssetClassLeakError(
            contaminating_symbols=["BTC", "ETH"],
            pipeline_asset_class=AssetClass.EQUITY,
            contaminating_asset_class=AssetClass.CRYPTO,
        )
        msg = str(err)
        assert "BTC" in msg
        assert "equity" in msg.lower()
        assert "crypto" in msg.lower()
        assert "ASSET CLASS LEAK" in msg


class TestEmptyOHLCVError:
    def test_crypto_message(self):
        err = EmptyOHLCVError(
            asset_class=AssetClass.CRYPTO,
            symbols=["BTC", "ETH"],
            run_mode="normal",
        )
        msg = str(err)
        assert "EMPTY_CRYPTO_OHLCV_UNIVERSE" in msg
        assert "0 symbols" in msg

    def test_equity_message(self):
        err = EmptyOHLCVError(
            asset_class=AssetClass.EQUITY,
            symbols=["INFY.NS"],
            run_mode="test",
        )
        assert "EMPTY_EQUITY_OHLCV_UNIVERSE" in str(err)


class TestCacheOnlyViolationError:
    def test_message(self):
        err = CacheOnlyViolationError("BTC", data_type="OHLCV")
        msg = str(err)
        assert "TEST MODE" in msg
        assert "BTC" in msg


class TestMixedAssetRunError:
    def test_message(self):
        err = MixedAssetRunError()
        msg = str(err)
        assert "MIXED ASSET RUN FORBIDDEN" in msg
        assert "equity-run" in msg
        assert "crypto-run" in msg


# ---------------------------------------------------------------------------
# Crypto symbol registry
# ---------------------------------------------------------------------------

class TestCryptoSymbolRegistry:
    def _registry(self):
        from spectraquant.crypto.symbols.crypto_symbol_registry import CryptoSymbolRegistry
        return CryptoSymbolRegistry()

    def test_btc_resolves(self):
        reg = self._registry()
        rec = reg.get("BTC")
        assert rec.canonical_symbol == "BTC"
        assert rec.asset_class == AssetClass.CRYPTO

    def test_case_insensitive(self):
        reg = self._registry()
        assert reg.get("btc").canonical_symbol == "BTC"

    def test_equity_symbol_raises(self):
        reg = self._registry()
        with pytest.raises(SymbolResolutionError) as exc_info:
            reg.get("INFY.NS")
        assert exc_info.value.expected_asset_class == AssetClass.CRYPTO
        assert exc_info.value.actual_asset_class == AssetClass.EQUITY

    def test_equity_symbol_binance_spot_raises(self):
        reg = self._registry()
        with pytest.raises(SymbolResolutionError):
            reg.get_provider_symbol("TCS.NS", "binance_spot")

    def test_unknown_symbol_raises(self):
        reg = self._registry()
        with pytest.raises(SymbolResolutionError):
            reg.get("UNKNOWN_TOKEN_XYZ")

    def test_binance_spot_btc(self):
        reg = self._registry()
        assert reg.get_provider_symbol("BTC", "binance_spot") == "BTC/USDT"

    def test_coinbase_eth(self):
        reg = self._registry()
        assert reg.get_provider_symbol("ETH", "coinbase") == "ETH-USD"

    def test_binance_perp_sol(self):
        reg = self._registry()
        assert reg.get_provider_symbol("SOL", "binance_perp") == "SOLUSDT"

    def test_unknown_provider_raises(self):
        reg = self._registry()
        with pytest.raises(SymbolResolutionError):
            reg.get_provider_symbol("BTC", "unknown_exchange")

    def test_register_custom_symbol(self):
        from spectraquant.crypto.symbols.crypto_symbol_registry import CryptoSymbolRecord
        reg = self._registry()
        rec = CryptoSymbolRecord(
            canonical_symbol="PEPE",
            ccxt_binance_spot="PEPE/USDT",
            coingecko_id="pepe",
        )
        reg.register(rec)
        assert reg.get("PEPE").coingecko_id == "pepe"

    def test_register_equity_symbol_raises(self):
        from spectraquant.crypto.symbols.crypto_symbol_registry import CryptoSymbolRecord
        reg = self._registry()
        with pytest.raises(SymbolResolutionError):
            reg.register(CryptoSymbolRecord(canonical_symbol="INFY.NS"))

    def test_list_canonical_sorted(self):
        reg = self._registry()
        symbols = reg.list_canonical()
        assert symbols == sorted(symbols)


# ---------------------------------------------------------------------------
# Crypto symbol mapper
# ---------------------------------------------------------------------------

class TestCryptoSymbolMapper:
    def _mapper(self):
        from spectraquant.crypto.symbols.crypto_symbol_mapper import CryptoSymbolMapper
        return CryptoSymbolMapper()

    def test_to_binance_spot(self):
        mapper = self._mapper()
        assert mapper.to_binance_spot("BTC") == "BTC/USDT"

    def test_to_coinbase(self):
        mapper = self._mapper()
        assert mapper.to_coinbase("ETH") == "ETH-USD"

    def test_equity_leak_raises(self):
        mapper = self._mapper()
        with pytest.raises(SymbolResolutionError):
            mapper.to_binance_spot("INFY.NS")

    def test_validate_no_equity_leak_raises(self):
        mapper = self._mapper()
        with pytest.raises(AssetClassLeakError) as exc_info:
            mapper.validate_no_equity_leak(["BTC", "ETH", "RELIANCE.NS"])
        assert "RELIANCE.NS" in exc_info.value.contaminating_symbols

    def test_validate_no_equity_leak_passes(self):
        mapper = self._mapper()
        # Should not raise
        mapper.validate_no_equity_leak(["BTC", "ETH", "SOL"])

    def test_map_to_binance_spot_batch(self):
        mapper = self._mapper()
        result = mapper.map_to_binance_spot(["BTC", "ETH"])
        assert result["BTC"] == "BTC/USDT"
        assert result["ETH"] == "ETH/USDT"


# ---------------------------------------------------------------------------
# Equity symbol registry
# ---------------------------------------------------------------------------

class TestEquitySymbolRegistry:
    def _registry(self):
        from spectraquant.equities.symbols.equity_symbol_registry import EquitySymbolRegistry
        return EquitySymbolRegistry()

    def test_infy_resolves(self):
        reg = self._registry()
        rec = reg.get("INFY")
        assert rec.canonical_symbol == "INFY"
        assert rec.asset_class == AssetClass.EQUITY
        assert rec.yfinance_symbol == "INFY.NS"

    def test_case_insensitive(self):
        reg = self._registry()
        assert reg.get("infy").yfinance_symbol == "INFY.NS"

    def test_crypto_symbol_raises(self):
        reg = self._registry()
        with pytest.raises(SymbolResolutionError) as exc_info:
            reg.get("BTC")
        assert exc_info.value.actual_asset_class == AssetClass.CRYPTO

    def test_eth_raises_as_crypto(self):
        reg = self._registry()
        with pytest.raises(SymbolResolutionError):
            reg.get("ETH")

    def test_unknown_symbol_raises(self):
        reg = self._registry()
        with pytest.raises(SymbolResolutionError):
            reg.get("UNKNOWN_STOCK_XYZ")

    def test_get_yfinance_symbol(self):
        reg = self._registry()
        assert reg.get_yfinance_symbol("TCS") == "TCS.NS"

    def test_register_from_yfinance(self):
        reg = self._registry()
        reg.register_from_yfinance_symbol("ZOMATO.NS")
        assert reg.get("ZOMATO").yfinance_symbol == "ZOMATO.NS"

    def test_list_canonical_sorted(self):
        reg = self._registry()
        symbols = reg.list_canonical()
        assert symbols == sorted(symbols)


# ---------------------------------------------------------------------------
# Equity symbol mapper
# ---------------------------------------------------------------------------

class TestEquitySymbolMapper:
    def _mapper(self):
        from spectraquant.equities.symbols.equity_symbol_mapper import EquitySymbolMapper
        return EquitySymbolMapper()

    def test_to_yfinance_infy(self):
        mapper = self._mapper()
        assert mapper.to_yfinance("INFY") == "INFY.NS"

    def test_crypto_symbol_raises_leak_error(self):
        mapper = self._mapper()
        with pytest.raises(AssetClassLeakError) as exc_info:
            mapper.to_yfinance("BTC")
        assert "BTC" in exc_info.value.contaminating_symbols

    def test_validate_no_crypto_leak_raises(self):
        mapper = self._mapper()
        with pytest.raises(AssetClassLeakError) as exc_info:
            mapper.validate_no_crypto_leak(["INFY", "BTC", "ETH"])
        assert "BTC" in exc_info.value.contaminating_symbols

    def test_validate_no_crypto_leak_passes(self):
        mapper = self._mapper()
        mapper.validate_no_crypto_leak(["INFY", "TCS", "RELIANCE"])

    def test_bootstrap_from_yfinance_symbols(self):
        mapper = self._mapper()
        mapper.bootstrap_from_yfinance_symbols(["ZOMATO.NS", "PAYTM.NS"])
        assert mapper.to_yfinance("ZOMATO") == "ZOMATO.NS"

    def test_map_to_yfinance_batch(self):
        mapper = self._mapper()
        result = mapper.map_to_yfinance(["INFY", "TCS"])
        assert result["INFY"] == "INFY.NS"
        assert result["TCS"] == "TCS.NS"


# ---------------------------------------------------------------------------
# Equity universe builder
# ---------------------------------------------------------------------------

class TestEquityUniverseBuilder:
    def test_builds_from_config_tickers(self):
        from spectraquant.equities.universe.equity_universe_builder import EquityUniverseBuilder
        builder = EquityUniverseBuilder(config={"tickers": ["INFY.NS", "TCS.NS"]})
        symbols = builder.build()
        assert "INFY.NS" in symbols
        assert "TCS.NS" in symbols

    def test_deduplicates_symbols(self):
        from spectraquant.equities.universe.equity_universe_builder import EquityUniverseBuilder
        builder = EquityUniverseBuilder(
            config={"tickers": ["INFY.NS", "INFY.NS", "TCS.NS"]}
        )
        symbols = builder.build()
        assert symbols.count("INFY.NS") == 1

    def test_empty_universe_raises(self):
        from spectraquant.equities.universe.equity_universe_builder import EquityUniverseBuilder
        builder = EquityUniverseBuilder(config={"tickers": []})
        with pytest.raises(EmptyUniverseError):
            builder.build()

    def test_crypto_symbols_raise_leak_error(self):
        from spectraquant.equities.universe.equity_universe_builder import EquityUniverseBuilder
        builder = EquityUniverseBuilder(
            config={"tickers": ["INFY.NS", "BTC", "ETH"]}
        )
        with pytest.raises(AssetClassLeakError) as exc_info:
            builder.build()
        assert "BTC" in exc_info.value.contaminating_symbols

    def test_crypto_pair_symbols_raise_leak_error(self):
        from spectraquant.equities.universe.equity_universe_builder import EquityUniverseBuilder
        builder = EquityUniverseBuilder(
            config={"tickers": ["BTC/USDT"]}
        )
        with pytest.raises(AssetClassLeakError):
            builder.build()

    def test_exclude_filter(self):
        from spectraquant.equities.universe.equity_universe_builder import EquityUniverseBuilder
        builder = EquityUniverseBuilder(
            config={"tickers": ["INFY.NS", "TCS.NS"], "exclude": ["INFY.NS"]}
        )
        symbols = builder.build()
        assert "INFY.NS" not in symbols
        assert "TCS.NS" in symbols

    def test_reads_csv_file(self, tmp_path):
        from spectraquant.equities.universe.equity_universe_builder import EquityUniverseBuilder
        csv_path = tmp_path / "tickers.csv"
        csv_path.write_text("ticker\nINFY.NS\nTCS.NS\n")
        builder = EquityUniverseBuilder(config={"tickers_file": str(csv_path)})
        symbols = builder.build()
        assert "INFY.NS" in symbols


# ---------------------------------------------------------------------------
# Equity signal agents
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n: int = 60, trend: float = 1.0) -> "pd.DataFrame":
    import pandas as pd
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    base = 1000.0
    close = [base + i * trend for i in range(n)]
    return pd.DataFrame({
        "open": [c - 0.5 for c in close],
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
        "volume": [1_000_000] * n,
    }, index=idx)


class TestEquitySignalAgents:
    def test_momentum_ok(self):
        from spectraquant.equities.signals.momentum_agent import MomentumAgent
        agent = MomentumAgent()
        df = _make_ohlcv_df(60, trend=1.0)
        out = agent.run(df, "INFY.NS")
        assert out.status == SignalStatus.OK
        assert -1 <= out.signal_score <= 1
        assert 0 <= out.confidence <= 1

    def test_momentum_no_data(self):
        from spectraquant.equities.signals.momentum_agent import MomentumAgent
        import pandas as pd
        agent = MomentumAgent()
        out = agent.run(pd.DataFrame(), "INFY.NS")
        assert out.status == SignalStatus.NO_SIGNAL

    def test_mean_reversion_ok(self):
        from spectraquant.equities.signals.mean_reversion_agent import MeanReversionAgent
        agent = MeanReversionAgent()
        df = _make_ohlcv_df(60)
        out = agent.run(df, "TCS.NS")
        assert out.status == SignalStatus.OK

    def test_volatility_ok(self):
        from spectraquant.equities.signals.volatility_agent import VolatilityAgent
        agent = VolatilityAgent()
        df = _make_ohlcv_df(60)
        out = agent.run(df, "RELIANCE.NS")
        assert out.status == SignalStatus.OK

    def test_breakout_ok(self):
        from spectraquant.equities.signals.breakout_agent import BreakoutAgent
        agent = BreakoutAgent()
        df = _make_ohlcv_df(60)
        out = agent.run(df, "WIPRO.NS")
        assert out.status == SignalStatus.OK

    def test_regime_ok(self):
        from spectraquant.equities.signals.regime_agent import RegimeAgent
        agent = RegimeAgent()
        df = _make_ohlcv_df(100)
        out = agent.run(df, "HDFC.NS")
        assert out.status == SignalStatus.OK

    def test_quality_passes(self):
        from spectraquant.equities.signals.quality_agent import QualityAgent
        agent = QualityAgent(min_rows=20)
        df = _make_ohlcv_df(60)
        out = agent.run(df, "SBIN.NS")
        assert out.status == SignalStatus.OK

    def test_quality_fails_insufficient_history(self):
        from spectraquant.equities.signals.quality_agent import QualityAgent
        agent = QualityAgent(min_rows=100)
        df = _make_ohlcv_df(30)
        out = agent.run(df, "SBIN.NS")
        assert out.status == SignalStatus.NO_SIGNAL

    def test_news_sentiment_no_data_returns_no_signal(self):
        from spectraquant.equities.signals.news_sentiment_agent import NewsSentimentAgent
        agent = NewsSentimentAgent(news_data={})
        df = _make_ohlcv_df(60)
        out = agent.run(df, "INFY.NS")
        assert out.status == SignalStatus.NO_SIGNAL
        assert "NO_NEWS_DATA" in out.error_reason

    def test_news_sentiment_with_data(self):
        from spectraquant.equities.signals.news_sentiment_agent import NewsSentimentAgent
        agent = NewsSentimentAgent(
            news_data={"INFY.NS": {"sentiment_score": 0.7, "confidence": 0.8}}
        )
        df = _make_ohlcv_df(60)
        out = agent.run(df, "INFY.NS")
        assert out.status == SignalStatus.OK
        assert out.signal_score == pytest.approx(0.7)

    def test_insufficient_rows_returns_no_signal(self):
        from spectraquant.equities.signals.momentum_agent import MomentumAgent
        agent = MomentumAgent()
        df = _make_ohlcv_df(5)  # below MIN_ROWS=20
        out = agent.run(df, "INFY.NS")
        assert out.status == SignalStatus.NO_SIGNAL


# ---------------------------------------------------------------------------
# Equity meta-policy
# ---------------------------------------------------------------------------

class TestEquityMetaPolicy:
    def _policy(self):
        from spectraquant.equities.policy.meta_policy import EquityMetaPolicy
        return EquityMetaPolicy()

    def _ok_output(self, score=0.5, agent_id="equity_momentum"):
        from spectraquant.equities.signals._base_agent import AgentOutput
        return AgentOutput(
            canonical_symbol="INFY.NS",
            agent_id=agent_id,
            signal_score=score,
            confidence=0.8,
            status=SignalStatus.OK,
        )

    def test_single_agent_blends(self):
        policy = self._policy()
        out = self._ok_output(0.6)
        decision = policy.blend([out], "INFY.NS")
        assert not decision.blocked
        assert abs(decision.blended_score) <= 1.0

    def test_all_no_signal_blocked(self):
        from spectraquant.equities.signals._base_agent import AgentOutput
        policy = self._policy()
        out = AgentOutput(
            canonical_symbol="INFY.NS",
            status=SignalStatus.NO_SIGNAL,
            error_reason="NO_PRICE_DATA",
        )
        decision = policy.blend([out], "INFY.NS")
        assert decision.blocked

    def test_blended_score_in_range(self):
        policy = self._policy()
        outputs = [self._ok_output(s) for s in [0.3, -0.1, 0.7, 0.5]]
        decision = policy.blend(outputs, "INFY.NS")
        if not decision.blocked:
            assert -1.0 <= decision.blended_score <= 1.0


# ---------------------------------------------------------------------------
# Equity allocator
# ---------------------------------------------------------------------------

class TestEquityAllocator:
    def _allocator(self):
        from spectraquant.equities.policy.allocator import EquityAllocator
        return EquityAllocator(max_weight=0.25)

    def _decision(self, score=0.5, confidence=0.8, blocked=False):
        from spectraquant.equities.policy.meta_policy import PolicyDecision
        return PolicyDecision(
            canonical_symbol="X",
            blended_score=score,
            confidence=confidence,
            blocked=blocked,
        )

    def test_basic_allocation(self):
        allocator = self._allocator()
        from spectraquant.equities.policy.meta_policy import PolicyDecision
        decisions = {
            "INFY.NS": PolicyDecision("INFY.NS", blended_score=0.6, confidence=0.8),
            "TCS.NS": PolicyDecision("TCS.NS", blended_score=0.4, confidence=0.7),
        }
        result = allocator.allocate(decisions)
        assert len(result.target_weights) > 0
        total = sum(result.target_weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_blocked_assets_excluded(self):
        allocator = self._allocator()
        from spectraquant.equities.policy.meta_policy import PolicyDecision
        decisions = {
            "INFY.NS": PolicyDecision("INFY.NS", blended_score=0.6, confidence=0.8),
            "TCS.NS": PolicyDecision(
                "TCS.NS", blended_score=0.0, confidence=0.0,
                blocked=True, block_reason="test"
            ),
        }
        result = allocator.allocate(decisions)
        assert "TCS.NS" not in result.target_weights
        assert "TCS.NS" in result.blocked_assets

    def test_max_weight_capped(self):
        from spectraquant.equities.policy.allocator import EquityAllocator
        allocator = EquityAllocator(max_weight=0.20)
        from spectraquant.equities.policy.meta_policy import PolicyDecision
        decisions = {
            f"SYM{i}.NS": PolicyDecision(
                f"SYM{i}.NS", blended_score=0.5 + i * 0.1, confidence=0.8
            )
            for i in range(10)
        }
        result = allocator.allocate(decisions)
        for w in result.target_weights.values():
            assert w <= 0.20 + 1e-6


# ---------------------------------------------------------------------------
# Equity paper executor
# ---------------------------------------------------------------------------

class TestEquityPaperExecutor:
    def test_generates_buy_orders(self):
        from spectraquant.equities.execution.paper_executor import EquityPaperExecutor
        executor = EquityPaperExecutor()
        orders = executor.execute({"INFY.NS": 0.2, "TCS.NS": 0.3})
        assert len(orders) == 2
        actions = {o.symbol: o.action for o in orders}
        assert actions["INFY.NS"] == "BUY"

    def test_sell_when_reducing(self):
        from spectraquant.equities.execution.paper_executor import EquityPaperExecutor
        executor = EquityPaperExecutor()
        orders = executor.execute(
            target_weights={"INFY.NS": 0.1},
            current_weights={"INFY.NS": 0.3},
        )
        assert orders[0].action == "SELL"

    def test_no_order_for_unchanged_weight(self):
        from spectraquant.equities.execution.paper_executor import EquityPaperExecutor
        executor = EquityPaperExecutor()
        orders = executor.execute(
            target_weights={"INFY.NS": 0.2},
            current_weights={"INFY.NS": 0.2},
        )
        assert len(orders) == 0

    def test_order_log_accumulates(self):
        from spectraquant.equities.execution.paper_executor import EquityPaperExecutor
        executor = EquityPaperExecutor()
        executor.execute({"A.NS": 0.2})
        executor.execute({"B.NS": 0.3})
        assert len(executor.order_log) == 2


# ---------------------------------------------------------------------------
# EquityOHLCVResult.assert_ohlcv_available
# ---------------------------------------------------------------------------

class TestEquityOHLCVResultAssertOhlcvAvailable:
    """assert_ohlcv_available() is the hard guard that aborts the pipeline
    when no symbol has usable OHLCV data after QA population."""

    def test_raises_when_qa_is_empty(self):
        from spectraquant.equities.ingestion.price_downloader import EquityOHLCVResult
        result = EquityOHLCVResult(
            symbols_requested=["INFY.NS"],
        )
        with pytest.raises(EmptyOHLCVError):
            result.assert_ohlcv_available()

    def test_raises_when_all_symbols_have_no_ohlcv(self):
        from spectraquant.equities.ingestion.price_downloader import EquityOHLCVResult
        result = EquityOHLCVResult(
            symbols_requested=["INFY.NS", "TCS.NS"],
            symbols_loaded=[],
            symbols_failed=["INFY.NS", "TCS.NS"],
            qa={
                "INFY.NS": {"has_ohlcv": False, "rows_loaded": 0},
                "TCS.NS": {"has_ohlcv": False, "rows_loaded": 0},
            },
        )
        with pytest.raises(EmptyOHLCVError):
            result.assert_ohlcv_available()

    def test_does_not_raise_when_at_least_one_symbol_has_ohlcv(self):
        import pandas as pd
        from spectraquant.equities.ingestion.price_downloader import EquityOHLCVResult
        idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame({"open": [1.0] * 5, "close": [1.1] * 5}, index=idx)
        result = EquityOHLCVResult(
            symbols_requested=["INFY.NS", "TCS.NS"],
            symbols_loaded=["INFY.NS"],
            symbols_failed=["TCS.NS"],
            prices={"INFY.NS": df},
            qa={
                "INFY.NS": {"has_ohlcv": True, "rows_loaded": 5},
                "TCS.NS": {"has_ohlcv": False, "rows_loaded": 0},
            },
        )
        result.assert_ohlcv_available()  # must not raise

    def test_error_contains_requested_symbols(self):
        from spectraquant.equities.ingestion.price_downloader import EquityOHLCVResult
        result = EquityOHLCVResult(
            symbols_requested=["INFY.NS", "TCS.NS"],
            qa={
                "INFY.NS": {"has_ohlcv": False},
                "TCS.NS": {"has_ohlcv": False},
            },
        )
        with pytest.raises(EmptyOHLCVError) as exc_info:
            result.assert_ohlcv_available()
        assert exc_info.value.asset_class == AssetClass.EQUITY
        assert "INFY.NS" in exc_info.value.symbols or "TCS.NS" in exc_info.value.symbols


# ---------------------------------------------------------------------------
# Equity reporter
# ---------------------------------------------------------------------------

class TestEquityReporter:
    def test_write_report(self, tmp_path):
        from spectraquant.equities.reporting.reporter import EquityReporter, EquityRunReport
        reporter = EquityReporter(reports_dir=tmp_path)
        report = EquityRunReport(
            run_id="test123",
            symbols_requested=["INFY.NS"],
            symbols_loaded=["INFY.NS"],
        )
        path = reporter.write(report)
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["run_id"] == "test123"

    def test_write_qa_matrix(self, tmp_path):
        from spectraquant.equities.reporting.reporter import EquityReporter
        reporter = EquityReporter(reports_dir=tmp_path)
        qa = {"INFY.NS": {"has_ohlcv": True, "rows_loaded": 252}}
        path = reporter.write_qa_matrix(qa, run_id="qa001")
        assert path.exists()


# ---------------------------------------------------------------------------
# yfinance provider test-mode
# ---------------------------------------------------------------------------

class TestYFinanceEquityProviderTestMode:
    def test_cache_miss_raises_in_test_mode(self, tmp_path):
        from spectraquant.equities.ingestion.providers.yfinance_provider import (
            YFinanceEquityProvider,
        )
        from spectraquant.core.enums import RunMode
        provider = YFinanceEquityProvider(
            cache_dir=tmp_path / "prices",
            run_mode=RunMode.TEST,
        )
        with pytest.raises(CacheOnlyViolationError):
            provider.fetch(["INFY.NS"])

    def test_cache_hit_returns_data(self, tmp_path):
        """Provider returns cached data without network call."""
        import pandas as pd
        from spectraquant.equities.ingestion.providers.yfinance_provider import (
            YFinanceEquityProvider,
        )
        from spectraquant.core.enums import RunMode

        cache_dir = tmp_path / "prices"
        cache_dir.mkdir()
        # Write a fake cache entry
        idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1_000_000] * 10,
        }, index=idx)
        df.to_parquet(cache_dir / "INFY.NS.parquet", engine="pyarrow")

        provider = YFinanceEquityProvider(
            cache_dir=cache_dir,
            run_mode=RunMode.TEST,
        )
        result = provider.fetch(["INFY.NS"])
        assert "INFY.NS" in result
        assert len(result["INFY.NS"]) == 10


# ---------------------------------------------------------------------------
# Cross-asset isolation tests
# ---------------------------------------------------------------------------

class TestCrossAssetIsolation:
    """Verify that crypto symbols cannot enter equity pipeline and vice versa."""

    def test_btc_in_equity_universe_raises(self):
        from spectraquant.equities.universe.equity_universe_builder import EquityUniverseBuilder
        builder = EquityUniverseBuilder(config={"tickers": ["BTC"]})
        with pytest.raises(AssetClassLeakError):
            builder.build()

    def test_eth_usdt_in_equity_universe_raises(self):
        from spectraquant.equities.universe.equity_universe_builder import EquityUniverseBuilder
        builder = EquityUniverseBuilder(config={"tickers": ["ETH/USDT"]})
        with pytest.raises(AssetClassLeakError):
            builder.build()

    def test_nse_ticker_in_crypto_registry_raises(self):
        from spectraquant.crypto.symbols.crypto_symbol_registry import CryptoSymbolRegistry
        reg = CryptoSymbolRegistry()
        with pytest.raises(SymbolResolutionError) as exc_info:
            reg.get("RELIANCE.NS")
        assert exc_info.value.actual_asset_class == AssetClass.EQUITY

    def test_equity_symbol_in_crypto_mapper_raises(self):
        from spectraquant.crypto.symbols.crypto_symbol_mapper import CryptoSymbolMapper
        mapper = CryptoSymbolMapper()
        with pytest.raises(SymbolResolutionError):
            mapper.to_binance_spot("TCS.NS")

    def test_btc_in_equity_registry_raises(self):
        from spectraquant.equities.symbols.equity_symbol_registry import EquitySymbolRegistry
        reg = EquitySymbolRegistry()
        with pytest.raises(SymbolResolutionError) as exc_info:
            reg.get("BTC")
        assert exc_info.value.actual_asset_class == AssetClass.CRYPTO

    def test_crypto_in_equity_mapper_raises(self):
        from spectraquant.equities.symbols.equity_symbol_mapper import EquitySymbolMapper
        mapper = EquitySymbolMapper()
        with pytest.raises(AssetClassLeakError):
            mapper.to_yfinance("ETH")


class TestCryptoPipelineOHLCVGuard:
    """Verify the hard EmptyOHLCVError guard in the crypto pipeline."""

    def test_raises_empty_ohlcv_when_no_data_in_normal_mode(self, tmp_path):
        """Non-test-mode pipeline must raise EmptyOHLCVError when n_ohlcv == 0."""
        import os
        from spectraquant.core.errors import EmptyOHLCVError
        from spectraquant.pipeline.crypto_run import run_crypto_pipeline

        # Ensure test-mode env var is NOT set
        os.environ.pop("SPECTRAQUANT_TEST_MODE", None)

        cfg = {
            "crypto": {
                "enabled": True,
                "symbols": ["BTC-USD"],
                "prices_dir": str(tmp_path / "empty_prices"),  # no cached prices
                "universe_mode": "static",
                "news_first": False,
                "universe_csv": str(
                    __import__("pathlib").Path(__file__).resolve().parents[1]
                    / "src/spectraquant/crypto/universe/crypto_universe.csv"
                ),
            },
            "crypto_dataset": {"data_dir": str(tmp_path / "crypto")},
            # test_mode intentionally absent (normal mode)
            "news_ai": {"enabled": False},
            "onchain_ai": {"enabled": False},
            "agents": {"enabled": False},
            "crypto_meta_policy": {"enabled": False},
            "crypto_portfolio": {},
        }

        with pytest.raises(EmptyOHLCVError) as exc_info:
            run_crypto_pipeline(cfg=cfg, dry_run=True)

        err_msg = str(exc_info.value)
        assert "EMPTY_CRYPTO_OHLCV_UNIVERSE" in err_msg

    def test_does_not_raise_in_test_mode(self, tmp_path, monkeypatch):
        """Test-mode pipeline must NOT raise even when all prices are missing."""
        from spectraquant.pipeline.crypto_run import run_crypto_pipeline

        monkeypatch.setenv("SPECTRAQUANT_TEST_MODE", "true")

        cfg = {
            "crypto": {
                "enabled": True,
                "symbols": ["BTC-USD"],
                "prices_dir": str(tmp_path / "empty_prices"),  # no cached prices
                "universe_mode": "static",
                "news_first": False,
                "universe_csv": str(
                    __import__("pathlib").Path(__file__).resolve().parents[1]
                    / "src/spectraquant/crypto/universe/crypto_universe.csv"
                ),
            },
            "crypto_dataset": {"data_dir": str(tmp_path / "crypto")},
            "test_mode": {"enabled": True},
            "news_ai": {"enabled": False},
            "onchain_ai": {"enabled": False},
            "agents": {"enabled": False},
            "crypto_meta_policy": {"enabled": False},
            "crypto_portfolio": {},
        }

        result = run_crypto_pipeline(cfg=cfg, dry_run=True)
        assert isinstance(result, dict)
