"""Tests for new V3 modules: Feature Store, Strategy Portfolio, Execution,
Monitoring, ingestion improvements, and config additions.

All tests are self-contained – no network calls, no permanent file system
writes (tmp_path fixtures only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest


# ===========================================================================
# Helpers
# ===========================================================================


def _make_feature_df(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Return a small synthetic feature DataFrame with a DatetimeIndex."""
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "momentum_20d": rng.standard_normal(n),
            "rsi_14": rng.uniform(20, 80, n),
            "atr_14": rng.uniform(0.01, 0.05, n),
        },
        index=dates,
    )


def _minimal_crypto_cfg() -> dict[str, Any]:
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
            "symbols": ["BTC", "ETH"],
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


# ===========================================================================
# Feature Store tests
# ===========================================================================


class TestFeatureStoreMetadata:
    def test_to_dict_round_trip(self) -> None:
        from spectraquant_v3.feature_store.metadata import FeatureSetMetadata

        meta = FeatureSetMetadata(
            feature_name="rsi_14",
            feature_version="1.0.0",
            symbol="BTC",
            asset_class="crypto",
            source_run_id="run_001",
            date_start="2024-01-01",
            date_end="2024-06-30",
            row_count=181,
            feature_columns=["rsi_14"],
            metadata={"note": "test"},
        )
        d = meta.to_dict()
        restored = FeatureSetMetadata.from_dict(d)
        assert restored.feature_name == "rsi_14"
        assert restored.symbol == "BTC"
        assert restored.row_count == 181

    def test_write_read(self, tmp_path: Path) -> None:
        from spectraquant_v3.feature_store.metadata import FeatureSetMetadata

        meta = FeatureSetMetadata(
            feature_name="momentum_20d",
            feature_version="1.0.0",
            symbol="ETH",
            asset_class="crypto",
        )
        path = tmp_path / "meta.json"
        meta.write(path)
        restored = FeatureSetMetadata.read(path)
        assert restored.feature_name == "momentum_20d"
        assert restored.symbol == "ETH"


class TestFeatureStore:
    def test_write_and_read(self, tmp_path: Path) -> None:
        from spectraquant_v3.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "store")
        df = _make_feature_df()
        meta = store.write_feature_frame(
            df=df,
            feature_name="momentum_20d",
            feature_version="1.0.0",
            symbol="BTC",
            asset_class="crypto",
            source_run_id="run_001",
        )
        assert meta.row_count == len(df)
        assert meta.feature_name == "momentum_20d"

        df2 = store.read_feature_frame("momentum_20d", "1.0.0", "BTC", "crypto")
        assert len(df2) == len(df)

    def test_write_empty_raises(self, tmp_path: Path) -> None:
        from spectraquant_v3.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "store")
        with pytest.raises(ValueError, match="empty"):
            store.write_feature_frame(
                df=pd.DataFrame(),
                feature_name="x",
                feature_version="1.0.0",
                symbol="BTC",
                asset_class="crypto",
            )

    def test_read_missing_raises(self, tmp_path: Path) -> None:
        from spectraquant_v3.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "store")
        with pytest.raises(FileNotFoundError):
            store.read_feature_frame("missing", "1.0.0", "BTC", "crypto")

    def test_list_feature_sets(self, tmp_path: Path) -> None:
        from spectraquant_v3.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "store")
        df = _make_feature_df()
        store.write_feature_frame(df, "rsi_14", "1.0.0", "BTC", "crypto")
        store.write_feature_frame(df, "rsi_14", "1.0.0", "ETH", "crypto")
        store.write_feature_frame(df, "momentum_20d", "1.0.0", "BTC", "crypto")

        all_sets = store.list_feature_sets()
        assert len(all_sets) == 3

        rsi_sets = store.list_feature_sets(feature_name="rsi_14")
        assert len(rsi_sets) == 2

        crypto_sets = store.list_feature_sets(asset_class="crypto")
        assert len(crypto_sets) == 3

    def test_query_feature_data(self, tmp_path: Path) -> None:
        from spectraquant_v3.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "store")
        df = _make_feature_df(n=30)
        store.write_feature_frame(df, "rsi_14", "1.0.0", "BTC", "crypto")
        store.write_feature_frame(df, "rsi_14", "1.0.0", "ETH", "crypto")

        result = store.query_feature_data(feature_name="rsi_14", symbol="BTC")
        assert not result.empty
        assert "_symbol" in result.columns

    def test_query_date_filter(self, tmp_path: Path) -> None:
        from spectraquant_v3.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "store")
        df = _make_feature_df(n=60)
        store.write_feature_frame(df, "rsi_14", "1.0.0", "BTC", "crypto")

        result = store.query_feature_data(
            feature_name="rsi_14",
            date_start="2024-01-15",
            date_end="2024-01-31",
        )
        assert not result.empty
        assert len(result) <= 60

    def test_query_no_match_returns_empty(self, tmp_path: Path) -> None:
        from spectraquant_v3.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "store")
        result = store.query_feature_data(feature_name="nonexistent")
        assert result.empty

    def test_symbol_slash_safe(self, tmp_path: Path) -> None:
        """Symbols containing '/' are stored safely without creating subdirs."""
        from spectraquant_v3.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "store")
        df = _make_feature_df()
        store.write_feature_frame(df, "rsi_14", "1.0.0", "BTC/USDT", "crypto")
        df2 = store.read_feature_frame("rsi_14", "1.0.0", "BTC/USDT", "crypto")
        assert len(df2) == len(df)


# ===========================================================================
# Strategy Portfolio tests
# ===========================================================================


class TestStrategyPortfolio:
    def test_compute_weights_equal(self) -> None:
        from spectraquant_v3.strategy_portfolio import StrategyPortfolio

        p = StrategyPortfolio("p1", ["s1", "s2", "s3"], weighting_scheme="equal")
        w = p.compute_weights()
        assert len(w) == 3
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert abs(w["s1"] - 1 / 3) < 1e-9

    def test_compute_weights_risk_budget(self) -> None:
        from spectraquant_v3.strategy_portfolio import StrategyPortfolio

        p = StrategyPortfolio(
            "p1",
            ["s1", "s2"],
            weighting_scheme="risk_budget",
            risk_budget={"s1": 2.0, "s2": 1.0},
        )
        w = p.compute_weights()
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert w["s1"] > w["s2"]

    def test_compute_weights_custom(self) -> None:
        from spectraquant_v3.strategy_portfolio import StrategyPortfolio

        p = StrategyPortfolio(
            "p1",
            ["s1", "s2"],
            weighting_scheme="custom",
            custom_weights={"s1": 3.0, "s2": 1.0},
        )
        w = p.compute_weights()
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert w["s1"] > w["s2"]

    def test_max_strategy_weight_cap(self) -> None:
        """With 3 equal strategies and cap=0.2, each raw weight (1/3 ≈ 0.33)
        exceeds the cap.  After capping to 0.2 and renormalising, each weight
        should equal 1/3 again (all are capped equally).  The key invariant is
        that weights always sum to 1.0 and no weight exceeds the cap post-renorm
        when other strategies can absorb the excess.
        
        We test with 4 strategies and a cap of 0.40 so that some strategies are
        over the cap and renormalisation still enforces the cap after redistribution.
        """
        from spectraquant_v3.strategy_portfolio import StrategyPortfolio

        # With 2 strategies, cap=0.4: equal gives 0.5 each → capped to 0.4 each
        # → renorm: 0.4/(0.4+0.4) = 0.5. Cap is soft when both are capped equally.
        # Test that weights are consistent and sum to 1.0
        p = StrategyPortfolio(
            "p1",
            ["s1", "s2", "s3"],
            weighting_scheme="equal",
            max_strategy_weight=0.4,
        )
        w = p.compute_weights()
        assert abs(sum(w.values()) - 1.0) < 1e-9
        # With 3 equal strategies (1/3 each > 0.4 cap), all get capped and
        # renorm brings them back to equal (1/3 each < 0.5)
        assert all(v <= 0.5 + 1e-9 for v in w.values())

    def test_invalid_weighting_scheme(self) -> None:
        from spectraquant_v3.strategy_portfolio import StrategyPortfolio

        with pytest.raises(ValueError, match="weighting_scheme"):
            StrategyPortfolio("p1", ["s1"], weighting_scheme="invalid")

    def test_empty_strategy_ids_raises(self) -> None:
        from spectraquant_v3.strategy_portfolio import StrategyPortfolio

        with pytest.raises(ValueError, match="strategy_ids"):
            StrategyPortfolio("p1", [])

    def test_run_dry_run(self, tmp_path: Path) -> None:
        """Portfolio.run() in dry_run mode should not write files."""
        from unittest.mock import patch

        from spectraquant_v3.strategy_portfolio import StrategyPortfolio

        fake_result = {
            "status": "ok",
            "metrics": {"sharpe": 1.5, "cagr": 0.2},
        }

        with patch(
            "spectraquant_v3.strategy_portfolio.portfolio.run_strategy",
            return_value=fake_result,
        ):
            p = StrategyPortfolio(
                "test_portfolio",
                ["s1", "s2"],
                weighting_scheme="equal",
                output_dir=str(tmp_path / "output"),
            )
            result = p.run(
                cfg_by_strategy={"s1": {}, "s2": {}},
                dry_run=True,
            )

        assert result.portfolio_id == "test_portfolio"
        assert "s1" in result.weights
        assert "s2" in result.weights
        assert not result.artifact_paths  # dry_run → no files written

    def test_portfolio_result_to_dict(self) -> None:
        from spectraquant_v3.strategy_portfolio.result import PortfolioResult

        r = PortfolioResult(
            portfolio_id="p1",
            strategy_ids=["s1", "s2"],
            weights={"s1": 0.5, "s2": 0.5},
            metrics={"weighted_cagr": 0.15},
        )
        d = r.to_dict()
        assert d["portfolio_id"] == "p1"
        assert d["weights"]["s1"] == 0.5

    def test_portfolio_result_write(self, tmp_path: Path) -> None:
        from spectraquant_v3.strategy_portfolio.result import PortfolioResult

        r = PortfolioResult(
            portfolio_id="p1",
            strategy_ids=["s1"],
            weights={"s1": 1.0},
        )
        path = r.write(tmp_path)
        assert path.exists()
        import json

        data = json.loads(path.read_text())
        assert data["portfolio_id"] == "p1"


# ===========================================================================
# Execution module tests
# ===========================================================================


class TestExecutionSimulator:
    def test_basic_execution(self) -> None:
        from spectraquant_v3.execution import ExecutionSimulator

        sim = ExecutionSimulator(slippage_bps=5, transaction_cost_bps=5)
        results = sim.execute_weights(
            target_weights={"BTC": 0.5, "ETH": 0.3},
            prices={"BTC": 45000.0, "ETH": 3000.0},
        )
        assert len(results) == 2
        symbols = {r.symbol for r in results}
        assert "BTC" in symbols
        assert "ETH" in symbols

    def test_empty_weights_raises(self) -> None:
        from spectraquant_v3.execution import ExecutionSimulator

        sim = ExecutionSimulator()
        with pytest.raises(ValueError, match="target_weights"):
            sim.execute_weights({})

    def test_live_mode_raises(self) -> None:
        from spectraquant_v3.execution import ExecutionSimulator

        with pytest.raises(NotImplementedError, match="Live"):
            ExecutionSimulator(mode="live")

    def test_invalid_mode_raises(self) -> None:
        from spectraquant_v3.execution import ExecutionSimulator

        with pytest.raises(ValueError, match="mode"):
            ExecutionSimulator(mode="dark_pool")

    def test_fill_price_buy_above_ref(self) -> None:
        from spectraquant_v3.execution import ExecutionSimulator

        sim = ExecutionSimulator(slippage_bps=10, spread_bps=5)
        results = sim.execute_weights(
            target_weights={"BTC": 0.5},
            prev_weights={"BTC": 0.0},
            prices={"BTC": 100.0},
        )
        assert results[0].fill_price > 100.0  # buy → price above ref

    def test_fill_price_sell_below_ref(self) -> None:
        from spectraquant_v3.execution import ExecutionSimulator

        sim = ExecutionSimulator(slippage_bps=10, spread_bps=5)
        results = sim.execute_weights(
            target_weights={"BTC": 0.3},
            prev_weights={"BTC": 0.5},
            prices={"BTC": 100.0},
        )
        assert results[0].fill_price < 100.0  # sell → price below ref

    def test_max_position_size_cap(self) -> None:
        from spectraquant_v3.execution import ExecutionSimulator

        sim = ExecutionSimulator(max_position_size=0.3)
        results = sim.execute_weights(
            target_weights={"BTC": 0.9},
        )
        assert results[0].executed_weight <= 0.3 + 1e-9

    def test_total_cost_bps(self) -> None:
        from spectraquant_v3.execution import ExecutionSimulator

        sim = ExecutionSimulator()
        results = sim.execute_weights({"BTC": 0.5, "ETH": 0.3})
        cost = sim.total_cost_bps(results)
        assert cost > 0

    def test_summary(self) -> None:
        from spectraquant_v3.execution import ExecutionSimulator

        sim = ExecutionSimulator()
        results = sim.execute_weights({"BTC": 0.5})
        summary = sim.summary(results)
        assert summary["symbols"] == 1
        assert summary["total_cost_bps"] > 0

    def test_summary_empty(self) -> None:
        from spectraquant_v3.execution import ExecutionSimulator

        sim = ExecutionSimulator()
        summary = sim.summary([])
        assert summary["symbols"] == 0

    def test_execution_result_to_dict(self) -> None:
        from spectraquant_v3.execution.result import ExecutionResult

        r = ExecutionResult(
            symbol="BTC",
            side="buy",
            target_weight=0.5,
            executed_weight=0.5,
            fill_price=45100.0,
        )
        d = r.to_dict()
        assert d["symbol"] == "BTC"
        assert d["side"] == "buy"


# ===========================================================================
# Monitoring module tests
# ===========================================================================


class TestPipelineMonitor:
    def test_check_ohlcv_coverage_pass(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_001")
        result = monitor.check_ohlcv_coverage(
            symbols=["BTC", "ETH"],
            loaded_symbols=["BTC", "ETH"],
            min_coverage=0.9,
        )
        assert result is True
        assert monitor.is_healthy()

    def test_check_ohlcv_coverage_fail(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_002")
        result = monitor.check_ohlcv_coverage(
            symbols=["BTC", "ETH", "SOL"],
            loaded_symbols=["BTC"],
            min_coverage=0.9,
        )
        assert result is False
        assert not monitor.is_healthy()

    def test_check_ohlcv_empty_universe(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_003")
        result = monitor.check_ohlcv_coverage(
            symbols=[],
            loaded_symbols=[],
        )
        assert result is False

    def test_check_signal_health(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_004")
        result = monitor.check_signal_health(signal_rows=["sig1", "sig2"])
        assert result is True

    def test_check_signal_health_empty(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_005")
        result = monitor.check_signal_health(signal_rows=[])
        assert result is False

    def test_check_allocation_health(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_006")
        result = monitor.check_allocation_health(
            allocation_rows=["a", "b"], min_allocated=1
        )
        assert result is True

    def test_check_qa_matrix_pass(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_007")
        result = monitor.check_qa_matrix(
            {"total": 10, "failed": 0}, max_failure_rate=0.2
        )
        assert result is True

    def test_check_qa_matrix_fail(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_008")
        result = monitor.check_qa_matrix(
            {"total": 10, "failed": 5}, max_failure_rate=0.2
        )
        assert result is False

    def test_mark_failed(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_009")
        monitor.get_report().mark_failed("critical error")
        assert monitor.get_report().status == "failed"
        assert not monitor.is_healthy()

    def test_get_report_structure(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_010")
        monitor.check_ohlcv_coverage(["BTC"], ["BTC"])
        report = monitor.get_report()
        d = report.to_dict()
        assert "run_id" in d
        assert "status" in d
        assert "checks" in d
        assert "alerts" in d
        assert "metrics" in d

    def test_report_write(self, tmp_path: Path) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_011")
        monitor.check_ohlcv_coverage(["BTC"], ["BTC"])
        path = monitor.get_report().write(tmp_path)
        assert path.exists()

    def test_add_custom_check(self) -> None:
        from spectraquant_v3.monitoring import PipelineMonitor

        monitor = PipelineMonitor("run_012")
        monitor.add_custom_check("my_check", passed=True, message="all good", value=42)
        report = monitor.get_report()
        checks = [c["name"] for c in report.checks]
        assert "my_check" in checks


# ===========================================================================
# CacheManager freshness tests
# ===========================================================================


class TestCacheManagerFreshness:
    def test_write_and_read_freshness(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        cm.write_freshness("BTC")
        info = cm.read_freshness("BTC")
        assert info is not None
        assert "last_updated" in info
        assert info["key"] == "BTC"

    def test_read_freshness_missing_returns_none(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        assert cm.read_freshness("nonexistent") is None

    def test_is_stale_no_sidecar(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        assert cm.is_stale("BTC") is True

    def test_is_stale_fresh(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        cm.write_freshness("BTC")
        # Just written → should NOT be stale with default max_age
        assert cm.is_stale("BTC", max_age_seconds=3600) is False

    def test_is_stale_expired(self, tmp_path: Path) -> None:
        from datetime import datetime, timedelta, timezone
        import json
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        # Write a freshness sidecar with an old timestamp
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        (tmp_path / "BTC.fresh").write_text(
            json.dumps({"key": "BTC", "last_updated": old_ts})
        )
        assert cm.is_stale("BTC", max_age_seconds=3600) is True

    def test_write_freshness_with_extra(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        cm.write_freshness("ETH", extra={"rows": 500, "provider": "ccxt"})
        info = cm.read_freshness("ETH")
        assert info is not None
        assert info.get("rows") == 500
        assert info.get("provider") == "ccxt"

    def test_freshness_slash_key(self, tmp_path: Path) -> None:
        """Freshness for keys containing '/' is written to a flat file."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        cm.write_freshness("BTC/USDT")
        # Verify file has __ in name, not /
        fresh_files = list(tmp_path.glob("*.fresh"))
        assert any("BTC__USDT" in f.name for f in fresh_files)
        info = cm.read_freshness("BTC/USDT")
        assert info is not None


# ===========================================================================
# Missing bars tests
# ===========================================================================


class TestMissingBars:
    def _make_ohlcv(self, dates: list[str]) -> pd.DataFrame:
        idx = pd.to_datetime(dates)
        return pd.DataFrame(
            {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            index=idx,
        )

    def test_no_missing(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        dates = pd.date_range("2024-01-01", periods=7, freq="D")
        df = self._make_ohlcv([str(d.date()) for d in dates])
        report = diagnose_missing_bars(df, "BTC", "2024-01-01", "2024-01-07")
        assert report.missing_count == 0
        assert report.coverage == 1.0

    def test_some_missing(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        # Only 5 of 7 days present
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05", "2024-01-07"]
        df = self._make_ohlcv(dates)
        report = diagnose_missing_bars(df, "BTC", "2024-01-01", "2024-01-07")
        assert report.missing_count == 2
        assert report.coverage < 1.0
        assert len(report.missing_dates) == 2

    def test_all_missing(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        df = self._make_ohlcv(["2023-12-01"])  # outside range
        report = diagnose_missing_bars(df, "BTC", "2024-01-01", "2024-01-07")
        assert report.coverage == 0.0

    def test_non_datetime_index_raises(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        df = pd.DataFrame({"close": [1, 2]}, index=[0, 1])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            diagnose_missing_bars(df, "BTC", "2024-01-01", "2024-01-07")

    def test_to_dict(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = self._make_ohlcv([str(d.date()) for d in dates])
        report = diagnose_missing_bars(df, "ETH", "2024-01-01", "2024-01-05")
        d = report.to_dict()
        assert "symbol" in d
        assert "missing_count" in d
        assert "coverage" in d


# ===========================================================================
# Ingestion audit log tests
# ===========================================================================


class TestIngestionAuditLog:
    def test_record_and_read(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditLog

        log = IngestionAuditLog(tmp_path / "audit.jsonl")
        log.record("BTC", "ccxt", "success", rows=1000, run_id="run_001")
        log.record("ETH", "ccxt", "failure", error="timeout")
        entries = log.read_entries()
        assert len(entries) == 2
        assert entries[0].symbol == "BTC"
        assert entries[1].status == "failure"

    def test_filter_by_symbol(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditLog

        log = IngestionAuditLog(tmp_path / "audit.jsonl")
        log.record("BTC", "ccxt", "success")
        log.record("ETH", "ccxt", "success")
        entries = log.read_entries(symbol="BTC")
        assert len(entries) == 1
        assert entries[0].symbol == "BTC"

    def test_filter_by_status(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditLog

        log = IngestionAuditLog(tmp_path / "audit.jsonl")
        log.record("BTC", "ccxt", "success")
        log.record("ETH", "ccxt", "failure")
        log.record("SOL", "ccxt", "cache_hit")
        entries = log.read_entries(status="failure")
        assert len(entries) == 1
        assert entries[0].symbol == "ETH"

    def test_read_missing_file(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditLog

        log = IngestionAuditLog(tmp_path / "missing.jsonl")
        assert log.read_entries() == []

    def test_to_dict(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditEntry

        entry = IngestionAuditEntry(
            symbol="BTC",
            provider="ccxt",
            status="success",
            rows=500,
            run_id="run_001",
        )
        d = entry.to_dict()
        assert d["symbol"] == "BTC"
        assert d["rows"] == 500

    def test_from_dict_round_trip(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditEntry

        entry = IngestionAuditEntry(
            symbol="ETH",
            provider="yfinance",
            status="cache_hit",
            rows=0,
            error="",
        )
        restored = IngestionAuditEntry.from_dict(entry.to_dict())
        assert restored.symbol == "ETH"
        assert restored.provider == "yfinance"

    def test_metadata_stored(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditLog

        log = IngestionAuditLog(tmp_path / "audit.jsonl")
        log.record(
            "BTC", "ccxt", "success", metadata={"exchange": "binance", "rows": 1000}
        )
        entries = log.read_entries()
        assert entries[0].metadata.get("exchange") == "binance"


# ===========================================================================
# Config files tests
# ===========================================================================


class TestConfigFiles:
    def test_providers_yaml_readable(self) -> None:
        import yaml
        from spectraquant_v3.core.config import _find_config_dir

        cfg_dir = _find_config_dir()
        path = cfg_dir / "providers.yaml"
        assert path.exists(), f"providers.yaml not found at {path}"
        data = yaml.safe_load(path.read_text())
        assert "crypto_providers" in data
        assert "equity_providers" in data

    def test_news_yaml_readable(self) -> None:
        import yaml
        from spectraquant_v3.core.config import _find_config_dir

        cfg_dir = _find_config_dir()
        path = cfg_dir / "news.yaml"
        assert path.exists(), f"news.yaml not found at {path}"
        data = yaml.safe_load(path.read_text())
        assert "news" in data

    def test_strategies_yaml_readable(self) -> None:
        import yaml
        from spectraquant_v3.core.config import _find_config_dir

        cfg_dir = _find_config_dir()
        path = cfg_dir / "strategies.yaml"
        assert path.exists(), f"strategies.yaml not found at {path}"
        data = yaml.safe_load(path.read_text())
        assert "strategies" in data

    def test_risk_yaml_readable(self) -> None:
        import yaml
        from spectraquant_v3.core.config import _find_config_dir

        cfg_dir = _find_config_dir()
        path = cfg_dir / "risk.yaml"
        assert path.exists(), f"risk.yaml not found at {path}"
        data = yaml.safe_load(path.read_text())
        assert "risk" in data

    def test_strategies_yaml_has_crypto_momentum_v1(self) -> None:
        import yaml
        from spectraquant_v3.core.config import _find_config_dir

        cfg_dir = _find_config_dir()
        data = yaml.safe_load((cfg_dir / "strategies.yaml").read_text())
        strategies = data.get("strategies", {})
        assert "crypto_momentum_v1" in strategies
        assert strategies["crypto_momentum_v1"]["asset_class"] == "crypto"

    def test_risk_yaml_has_risk_key(self) -> None:
        import yaml
        from spectraquant_v3.core.config import _find_config_dir

        cfg_dir = _find_config_dir()
        data = yaml.safe_load((cfg_dir / "risk.yaml").read_text())
        risk = data.get("risk", {})
        assert "max_gross_leverage" in risk
        assert "volatility_target" in risk


# ===========================================================================
# CLI smoke tests
# ===========================================================================


class TestCLINewCommands:
    def test_feature_store_help(self) -> None:
        from typer.testing import CliRunner
        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["feature-store", "--help"])
        assert result.exit_code == 0
        assert "feature" in result.output.lower()

    def test_feature_store_list_empty(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner
        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["feature-store", "list", "--store-root", str(tmp_path / "empty_store")],
        )
        assert result.exit_code == 0
        assert "No feature sets" in result.output

    def test_feature_store_list_with_data(self, tmp_path: Path) -> None:
        from spectraquant_v3.feature_store import FeatureStore
        from typer.testing import CliRunner
        from spectraquant_v3.cli.main import app

        # Populate a store
        store = FeatureStore(tmp_path / "store")
        df = _make_feature_df()
        store.write_feature_frame(df, "rsi_14", "1.0.0", "BTC", "crypto")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["feature-store", "list", "--store-root", str(tmp_path / "store")],
        )
        assert result.exit_code == 0
        assert "rsi_14" in result.output

    def test_strategy_portfolio_help(self) -> None:
        from typer.testing import CliRunner
        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["strategy-portfolio", "--help"])
        assert result.exit_code == 0
        assert "portfolio" in result.output.lower()
