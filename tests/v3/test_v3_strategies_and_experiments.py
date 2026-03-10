"""Tests for the Strategy Registry, Experiment Tracking, and CLI commands.

Tests are self-contained – no network calls, no permanent file system writes
(tmp_path fixtures only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ===========================================================================
# Helpers
# ===========================================================================


def _minimal_crypto_cfg(symbols=None) -> dict:
    if symbols is None:
        symbols = ["BTC", "ETH"]
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


def _minimal_equity_cfg(tickers=None) -> dict:
    if tickers is None:
        tickers = ["AAPL", "MSFT"]
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
            "target_vol": 0.15,
            "allocator": "equal_weight",
        },
        "equities": {
            "universe": {"tickers": tickers, "mode": "static"},
            "primary_ohlcv_provider": "yfinance",
            "quality_gate": {
                "min_market_cap_usd": 0,
                "min_30d_volume_usd": 0,
                "require_tradable_mapping": True,
            },
            "signals": {"momentum_lookback": 20, "rsi_period": 14},
        },
    }


# ===========================================================================
# StrategyDefinition tests
# ===========================================================================


class TestStrategyDefinition:
    def test_round_trip_serialisation(self):
        from spectraquant_v3.strategies.strategy_definition import (
            RiskConfig,
            StrategyDefinition,
        )

        defn = StrategyDefinition(
            strategy_id="test_strat_v1",
            asset_class="crypto",
            agents=["crypto_momentum_v1"],
            policy="confidence_filter_v1",
            allocator="rank_vol_target_allocator",
            rebalance_freq="W",
            tags=["test"],
            risk_config=RiskConfig(max_weight=0.15),
        )
        d = defn.to_dict()
        restored = StrategyDefinition.from_dict(d)
        assert restored.strategy_id == defn.strategy_id
        assert restored.asset_class == defn.asset_class
        assert restored.agents == defn.agents
        assert restored.risk_config.max_weight == defn.risk_config.max_weight
        assert restored.tags == defn.tags

    def test_validate_ok(self):
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        defn = StrategyDefinition(
            strategy_id="ok_strat",
            asset_class="equity",
            agents=["equity_momentum_v1"],
            policy="confidence_filter_v1",
            allocator="equal_weight",
        )
        defn.validate()  # should not raise

    def test_validate_bad_asset_class(self):
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        defn = StrategyDefinition(
            strategy_id="bad_strat",
            asset_class="futures",
            agents=["momentum"],
            policy="p",
            allocator="a",
        )
        with pytest.raises(ValueError, match="asset_class"):
            defn.validate()

    def test_validate_no_agents(self):
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        defn = StrategyDefinition(
            strategy_id="no_agents",
            asset_class="crypto",
            agents=[],
            policy="p",
            allocator="a",
        )
        with pytest.raises(ValueError, match="at least one agent"):
            defn.validate()

    def test_risk_config_round_trip(self):
        from spectraquant_v3.strategies.strategy_definition import RiskConfig

        rc = RiskConfig(max_weight=0.10, target_vol=0.20)
        assert RiskConfig.from_dict(rc.to_dict()).max_weight == 0.10
        assert RiskConfig.from_dict(rc.to_dict()).target_vol == 0.20


# ===========================================================================
# StrategyRegistry tests
# ===========================================================================


class TestStrategyRegistry:
    def test_builtin_strategies_registered(self):
        from spectraquant_v3.strategies.registry import StrategyRegistry

        ids = StrategyRegistry.list()
        assert "crypto_momentum_v1" in ids
        assert "equity_momentum_v1" in ids

    def test_get_existing(self):
        from spectraquant_v3.strategies.registry import StrategyRegistry

        defn = StrategyRegistry.get("crypto_momentum_v1")
        assert defn.asset_class == "crypto"

    def test_get_missing_raises(self):
        from spectraquant_v3.strategies.registry import StrategyRegistry

        with pytest.raises(KeyError, match="not registered"):
            StrategyRegistry.get("nonexistent_strategy")

    def test_register_and_unregister(self):
        from spectraquant_v3.strategies.registry import StrategyRegistry
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        defn = StrategyDefinition(
            strategy_id="temp_strat",
            asset_class="crypto",
            agents=["crypto_momentum_v1"],
            policy="confidence_filter_v1",
            allocator="equal_weight",
        )
        StrategyRegistry.register(defn)
        assert "temp_strat" in StrategyRegistry.list()

        StrategyRegistry.unregister("temp_strat")
        assert "temp_strat" not in StrategyRegistry.list()

    def test_list_enabled(self):
        from spectraquant_v3.strategies.registry import StrategyRegistry
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        disabled = StrategyDefinition(
            strategy_id="disabled_strat",
            asset_class="crypto",
            agents=["crypto_momentum_v1"],
            policy="confidence_filter_v1",
            allocator="equal_weight",
            enabled=False,
        )
        StrategyRegistry.register(disabled)
        enabled = StrategyRegistry.list_enabled()
        assert all(s.enabled for s in enabled)
        assert "disabled_strat" not in [s.strategy_id for s in enabled]
        # cleanup
        StrategyRegistry.unregister("disabled_strat")


# ===========================================================================
# Sub-registry tests
# ===========================================================================


class TestAgentRegistry:
    def test_builtin_agents_registered(self):
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        agents = AgentRegistry.list()
        assert "crypto_momentum_v1" in agents
        assert "equity_momentum_v1" in agents

    def test_get_returns_class(self):
        from spectraquant_v3.strategies.agents.registry import AgentRegistry
        from spectraquant_v3.crypto.signals.momentum import CryptoMomentumAgent

        cls = AgentRegistry.get("crypto_momentum_v1")
        assert cls is CryptoMomentumAgent

    def test_get_missing_raises(self):
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        with pytest.raises(KeyError):
            AgentRegistry.get("nonexistent_agent")

    def test_build(self):
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        agent = AgentRegistry.build("crypto_momentum_v1", run_id="test")
        assert hasattr(agent, "evaluate")


class TestPolicyRegistry:
    def test_builtin_policy_registered(self):
        from spectraquant_v3.strategies.policies.registry import PolicyRegistry

        assert "confidence_filter_v1" in PolicyRegistry.list()

    def test_get_returns_class(self):
        from spectraquant_v3.strategies.policies.registry import PolicyRegistry
        from spectraquant_v3.pipeline.meta_policy import MetaPolicy

        cls = PolicyRegistry.get("confidence_filter_v1")
        assert cls is MetaPolicy

    def test_get_missing_raises(self):
        from spectraquant_v3.strategies.policies.registry import PolicyRegistry

        with pytest.raises(KeyError):
            PolicyRegistry.get("nonexistent_policy")


class TestAllocatorRegistry:
    def test_builtin_allocators_registered(self):
        from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry

        allocs = AllocatorRegistry.list()
        assert "equal_weight" in allocs
        assert "vol_target_v1" in allocs
        assert "rank_vol_target_allocator" in allocs

    def test_get_returns_class(self):
        from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry
        from spectraquant_v3.pipeline.allocator import Allocator

        cls = AllocatorRegistry.get("equal_weight")
        assert cls is Allocator

    def test_get_missing_raises(self):
        from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry

        with pytest.raises(KeyError):
            AllocatorRegistry.get("nonexistent_allocator")


# ===========================================================================
# StrategyLoader tests
# ===========================================================================


class TestStrategyLoader:
    def test_load_crypto_momentum(self):
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load("crypto_momentum_v1")
        assert defn.strategy_id == "crypto_momentum_v1"
        assert defn.asset_class == "crypto"

    def test_load_equity_momentum(self):
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load("equity_momentum_v1")
        assert defn.asset_class == "equity"

    def test_load_missing_raises(self):
        from spectraquant_v3.strategies.loader import StrategyLoader

        with pytest.raises(KeyError):
            StrategyLoader.load("does_not_exist")

    def test_load_disabled_raises(self):
        from spectraquant_v3.strategies.loader import StrategyLoader
        from spectraquant_v3.strategies.registry import StrategyRegistry
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        disabled = StrategyDefinition(
            strategy_id="load_disabled_test",
            asset_class="crypto",
            agents=["crypto_momentum_v1"],
            policy="confidence_filter_v1",
            allocator="equal_weight",
            enabled=False,
        )
        StrategyRegistry.register(disabled)
        try:
            with pytest.raises(ValueError, match="disabled"):
                StrategyLoader.load("load_disabled_test")
        finally:
            StrategyRegistry.unregister("load_disabled_test")

    def test_build_pipeline_config_merges_risk(self):
        from spectraquant_v3.strategies.loader import StrategyLoader

        cfg = _minimal_crypto_cfg()
        merged = StrategyLoader.build_pipeline_config("crypto_momentum_v1", cfg)
        portfolio = merged["portfolio"]
        # Loaded from strategy's RiskConfig defaults
        assert "max_weight" in portfolio
        assert "_strategy_id" in merged
        assert merged["_strategy_id"] == "crypto_momentum_v1"

    def test_build_pipeline_config_vol_target_allocator(self):
        from spectraquant_v3.strategies.loader import StrategyLoader

        # crypto_momentum_v1 uses rank_vol_target_allocator -> should map to "vol_target"
        cfg = _minimal_crypto_cfg()
        merged = StrategyLoader.build_pipeline_config("crypto_momentum_v1", cfg)
        assert merged["portfolio"]["allocator"] == "vol_target"


# ===========================================================================
# run_strategy pipeline integration tests
# ===========================================================================


class TestRunStrategy:
    def test_run_crypto_strategy_dry_run(self, tmp_path):
        from spectraquant_v3.pipeline import run_strategy

        cfg = _minimal_crypto_cfg()
        result = run_strategy(
            strategy_id="crypto_momentum_v1",
            cfg=cfg,
            dry_run=True,
            project_root=str(tmp_path),
        )
        assert result["status"] == "success"
        assert result["strategy_id"] == "crypto_momentum_v1"
        assert isinstance(result["universe"], list)

    def test_run_equity_strategy_dry_run(self, tmp_path):
        from spectraquant_v3.pipeline import run_strategy

        cfg = _minimal_equity_cfg()
        result = run_strategy(
            strategy_id="equity_momentum_v1",
            cfg=cfg,
            dry_run=True,
            project_root=str(tmp_path),
        )
        assert result["status"] == "success"
        assert result["strategy_id"] == "equity_momentum_v1"

    def test_wrong_asset_class_raises(self, tmp_path):
        from spectraquant_v3.core.errors import MixedAssetClassRunError
        from spectraquant_v3.pipeline import run_strategy

        # Pass equity config for a crypto strategy
        equity_cfg = _minimal_equity_cfg()
        with pytest.raises(MixedAssetClassRunError):
            run_strategy(
                strategy_id="crypto_momentum_v1",
                cfg=equity_cfg,
                dry_run=True,
                project_root=str(tmp_path),
            )

    def test_unknown_strategy_raises(self, tmp_path):
        from spectraquant_v3.pipeline import run_strategy

        cfg = _minimal_crypto_cfg()
        with pytest.raises(KeyError):
            run_strategy(
                strategy_id="not_a_real_strategy",
                cfg=cfg,
                dry_run=True,
                project_root=str(tmp_path),
            )


# ===========================================================================
# ResultStore tests
# ===========================================================================


class TestResultStore:
    def test_write_and_read_config(self, tmp_path):
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        doc = {"experiment_id": "exp_001", "strategy_id": "s1"}
        store.write_config("exp_001", doc)
        loaded = store.read_config("exp_001")
        assert loaded["strategy_id"] == "s1"

    def test_write_and_read_metrics(self, tmp_path):
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        metrics = {"sharpe": 1.2, "cagr": 0.15, "max_drawdown": -0.10}
        store.write_metrics("exp_001", metrics)
        loaded = store.read_metrics("exp_001")
        assert loaded["sharpe"] == pytest.approx(1.2)

    def test_list_experiments(self, tmp_path):
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        for eid in ("exp_001", "exp_002", "exp_003"):
            store.write_config(eid, {"experiment_id": eid})

        ids = store.list_experiments()
        assert ids == ["exp_001", "exp_002", "exp_003"]

    def test_list_experiments_empty(self, tmp_path):
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "empty_dir")
        assert store.list_experiments() == []

    def test_read_missing_raises(self, tmp_path):
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        with pytest.raises(FileNotFoundError):
            store.read_config("nonexistent_exp")

    def test_write_dataset_manifest(self, tmp_path):
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        manifest = {"symbols": ["BTC", "ETH"], "dataset_version": "v1"}
        store.write_dataset_manifest("exp_001", manifest)
        loaded = store.read_dataset_manifest("exp_001")
        assert "BTC" in loaded["symbols"]

    def test_write_backtest_summary(self, tmp_path):
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        summary = {"total_steps": 52, "final_nav": 1.35}
        store.write_backtest_summary("exp_001", summary)
        loaded = store.read_backtest_summary("exp_001")
        assert loaded["total_steps"] == 52


# ===========================================================================
# RunTracker tests
# ===========================================================================


class TestRunTracker:
    def test_record_and_save(self, tmp_path):
        from spectraquant_v3.experiments.result_store import ResultStore
        from spectraquant_v3.experiments.run_tracker import RunTracker

        tracker = RunTracker(
            experiment_id="exp_001",
            strategy_id="crypto_momentum_v1",
            dataset_version="v1",
            config={"key": "value"},
        )
        tracker.record_metrics({"sharpe": 1.5, "cagr": 0.20})
        tracker.record_artefact("signals", "/tmp/signals.json")

        store = ResultStore(tmp_path / "experiments")
        paths = tracker.save(store)

        assert "config" in paths
        assert "metrics" in paths

        config_doc = store.read_config("exp_001")
        assert config_doc["strategy_id"] == "crypto_momentum_v1"
        assert config_doc["dataset_version"] == "v1"
        assert len(config_doc["config_hash"]) == 16
        assert config_doc["metrics_payload"]["sharpe"] == pytest.approx(1.5)

        metrics = store.read_metrics("exp_001")
        assert metrics["sharpe"] == pytest.approx(1.5)

    def test_to_dict(self):
        from spectraquant_v3.experiments.run_tracker import RunTracker

        tracker = RunTracker("e1", "s1")
        tracker.record_metrics({"cagr": 0.10})
        d = tracker.to_dict()
        assert d["experiment_id"] == "e1"
        assert d["metrics"]["cagr"] == pytest.approx(0.10)

    def test_config_hash_deterministic(self):
        from spectraquant_v3.experiments.run_tracker import RunTracker

        cfg_a = {"b": [1, {"x": 2}], "a": {"z": 9, "y": 8}}
        cfg_b = {"a": {"y": 8, "z": 9}, "b": [1, {"x": 2}]}
        h1 = RunTracker._hash_config(cfg_a)
        h2 = RunTracker._hash_config(cfg_b)
        assert h1 == h2
        assert len(h1) == 16


# ===========================================================================
# ExperimentManager tests
# ===========================================================================


class TestExperimentManager:
    def test_run_experiment_dry_run(self, tmp_path):
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager

        manager = ExperimentManager(tmp_path / "experiments")
        cfg = _minimal_crypto_cfg()

        result = manager.run_experiment(
            experiment_id="exp_001",
            strategy_id="crypto_momentum_v1",
            cfg=cfg,
            dry_run=True,
            project_root=str(tmp_path),
        )
        assert result["experiment_id"] == "exp_001"
        assert result["strategy_id"] == "crypto_momentum_v1"
        assert result["status"] == "success"

    def test_run_experiment_persists_results(self, tmp_path):
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager

        manager = ExperimentManager(tmp_path / "experiments")
        cfg = _minimal_crypto_cfg()

        manager.run_experiment(
            experiment_id="exp_persist",
            strategy_id="crypto_momentum_v1",
            cfg=cfg,
            dry_run=False,
            project_root=str(tmp_path),
        )
        config_path = tmp_path / "experiments" / "exp_persist" / "config.json"
        assert config_path.exists()

    def test_compare_experiments(self, tmp_path):
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        for eid, sharpe in [("exp_A", 1.2), ("exp_B", 0.8)]:
            store.write_config(
                eid,
                {
                    "experiment_id": eid,
                    "strategy_id": "s1",
                    "config_hash": "abc123",
                    "dataset_version": "ds_v1",
                    "metrics_payload": {
                        "sharpe": sharpe,
                        "cagr": 0.10,
                        "max_drawdown": -0.2,
                        "volatility": 0.3,
                        "win_rate": 0.55,
                        "turnover": 0.9,
                    },
                },
            )
            store.write_metrics(eid, {"sharpe": sharpe, "cagr": 0.10})

        manager = ExperimentManager(tmp_path / "experiments")
        rows = manager.compare_experiments(["exp_A", "exp_B"])

        assert len(rows) == 2
        sharpe_map = {r["experiment_id"]: r["sharpe"] for r in rows}
        assert sharpe_map["exp_A"] == pytest.approx(1.2)
        assert sharpe_map["exp_B"] == pytest.approx(0.8)
        assert rows[0]["turnover"] == pytest.approx(0.9)
        assert rows[0]["config_hash"] == "abc123"


    def test_compare_experiments_legacy_missing_metrics_file(self, tmp_path):
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        store.write_config(
            "exp_legacy",
            {
                "experiment_id": "exp_legacy",
                "strategy_id": "legacy_s",
                "metrics_payload": {"sharpe": 1.1, "turnover": 0.5},
            },
        )

        manager = ExperimentManager(tmp_path / "experiments")
        rows = manager.compare_experiments(["exp_legacy"])
        assert rows[0]["sharpe"] == pytest.approx(1.1)
        assert rows[0]["turnover"] == pytest.approx(0.5)

    def test_compare_missing_experiment(self, tmp_path):
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager

        manager = ExperimentManager(tmp_path / "experiments")
        rows = manager.compare_experiments(["does_not_exist"])
        assert rows[0].get("error") == "not_found"


# ===========================================================================
# CLI command tests (Typer CliRunner)
# ===========================================================================


class TestStrategyCLI:
    def test_strategy_list(self):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["strategy", "list"])
        assert result.exit_code == 0
        assert "crypto_momentum_v1" in result.output
        assert "equity_momentum_v1" in result.output

    def test_strategy_list_filter_crypto(self):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["strategy", "list", "--asset-class", "crypto"])
        assert result.exit_code == 0
        assert "crypto_momentum_v1" in result.output

    def test_strategy_show(self):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["strategy", "show", "crypto_momentum_v1"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["strategy_id"] == "crypto_momentum_v1"
        assert data["asset_class"] == "crypto"

    def test_strategy_show_unknown(self):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["strategy", "show", "unknown_strategy"])
        assert result.exit_code == 1


class TestExperimentCLI:
    def test_experiment_list_empty(self, tmp_path):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["experiment", "list", "--base-dir", str(tmp_path / "empty")],
        )
        assert result.exit_code == 0
        assert "No experiments" in result.output

    def test_experiment_list_with_data(self, tmp_path):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        store.write_config("exp_001", {"strategy_id": "s1", "run_timestamp": "2024-01-01T00:00:00+00:00"})
        store.write_metrics("exp_001", {"sharpe": 1.5})

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["experiment", "list", "--base-dir", str(tmp_path / "experiments")],
        )
        assert result.exit_code == 0
        assert "exp_001" in result.output

    def test_experiment_show(self, tmp_path):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        store.write_config("exp_001", {"strategy_id": "s1", "run_timestamp": "2024-01-01"})
        store.write_metrics("exp_001", {"sharpe": 1.1})

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["experiment", "show", "exp_001", "--base-dir", str(tmp_path / "experiments")],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["experiment_id"] == "exp_001"
        assert data["metrics"]["sharpe"] == pytest.approx(1.1)

    def test_experiment_compare(self, tmp_path):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        for eid in ("exp_001", "exp_002"):
            store.write_config(eid, {"strategy_id": "s1", "run_timestamp": "2024-01-01"})
            store.write_metrics(eid, {"sharpe": 1.0, "turnover": 0.25})

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "experiment",
                "compare",
                "exp_001,exp_002",
                "--base-dir",
                str(tmp_path / "experiments"),
            ],
        )
        assert result.exit_code == 0
        assert "exp_001" in result.output
        assert "exp_002" in result.output
        assert "TURNOVER" in result.output

    def test_experiment_compare_json_format(self, tmp_path):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app
        from spectraquant_v3.experiments.result_store import ResultStore

        store = ResultStore(tmp_path / "experiments")
        store.write_config("exp_001", {"strategy_id": "s1", "run_timestamp": "2024-01-01", "config_hash": "h1", "dataset_version": "d1"})
        store.write_metrics("exp_001", {"sharpe": 1.3, "turnover": 0.4})

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "experiment",
                "compare",
                "exp_001",
                "--base-dir",
                str(tmp_path / "experiments"),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        rows = json.loads(result.output)
        assert rows[0]["experiment_id"] == "exp_001"
        assert rows[0]["turnover"] == pytest.approx(0.4)
        assert rows[0]["config_hash"] == "h1"
