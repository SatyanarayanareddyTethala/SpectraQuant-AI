"""Regression tests for V3 strategy-definition and config-shape consistency.

Validates that:
- All registered strategies load successfully through StrategyLoader
- build_pipeline_config produces all expected keys (_strategy_id, _asset_class,
  _agents, _rebalance_freq) and correct allocator mode
- Pipeline agent resolution uses _agents from build_pipeline_config (registry-
  validated) rather than silently depending on cfg["strategies"][...]["agents"]
- strategies.yaml field names and values match StrategyDefinition schema and
  AgentRegistry / AllocatorRegistry contents
- CLI-to-pipeline handoff works for both crypto and equity strategies
- Invalid or incomplete strategy/config combinations fail with clear errors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_crypto_cfg(symbols: list[str] | None = None) -> dict[str, Any]:
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
            "symbols": symbols or ["BTC", "ETH"],
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


def _minimal_equity_cfg(tickers: list[str] | None = None) -> dict[str, Any]:
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
            "universe": {"tickers": tickers or ["AAPL", "MSFT"], "mode": "static"},
            "primary_ohlcv_provider": "yfinance",
            "quality_gate": {
                "min_market_cap_usd": 0,
                "min_30d_volume_usd": 0,
                "require_tradable_mapping": True,
            },
            "signals": {"momentum_lookback": 20, "rsi_period": 14},
        },
    }


# ---------------------------------------------------------------------------
# 1. Strategy loading – all registered strategies must load successfully
# ---------------------------------------------------------------------------


class TestStrategyLoaderAllStrategies:
    """StrategyLoader.load() must succeed for every registered, enabled strategy."""

    def test_all_enabled_strategies_load(self):
        from spectraquant_v3.strategies.loader import StrategyLoader
        from spectraquant_v3.strategies.registry import StrategyRegistry

        failed = []
        for sid in StrategyRegistry.list():
            defn = StrategyRegistry.get(sid)
            if not defn.enabled:
                continue
            try:
                StrategyLoader.load(sid)
            except (KeyError, ValueError) as exc:
                failed.append(f"{sid}: {exc}")

        assert not failed, (
            "The following registered strategies failed StrategyLoader.load():\n"
            + "\n".join(failed)
        )

    def test_all_agents_are_registered(self):
        """Every agent referenced by a registered strategy must be in AgentRegistry."""
        from spectraquant_v3.strategies.agents.registry import AgentRegistry
        from spectraquant_v3.strategies.registry import StrategyRegistry

        registered_agents = set(AgentRegistry.list())
        missing = {}
        for sid in StrategyRegistry.list():
            defn = StrategyRegistry.get(sid)
            bad = [a for a in defn.agents if a not in registered_agents]
            if bad:
                missing[sid] = bad

        assert not missing, (
            "Strategies reference unregistered agents:\n"
            + "\n".join(f"  {sid}: {bad}" for sid, bad in missing.items())
        )

    def test_all_allocators_are_registered(self):
        """Every allocator referenced by a registered strategy must be in AllocatorRegistry."""
        from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry
        from spectraquant_v3.strategies.registry import StrategyRegistry

        registered_allocs = set(AllocatorRegistry.list())
        missing = {}
        for sid in StrategyRegistry.list():
            defn = StrategyRegistry.get(sid)
            if defn.allocator not in registered_allocs:
                missing[sid] = defn.allocator

        assert not missing, (
            "Strategies reference unregistered allocators:\n"
            + "\n".join(f"  {sid}: {alloc!r}" for sid, alloc in missing.items())
        )

    def test_all_policies_are_registered(self):
        """Every policy referenced by a registered strategy must be in PolicyRegistry."""
        from spectraquant_v3.strategies.policies.registry import PolicyRegistry
        from spectraquant_v3.strategies.registry import StrategyRegistry

        registered_policies = set(PolicyRegistry.list())
        missing = {}
        for sid in StrategyRegistry.list():
            defn = StrategyRegistry.get(sid)
            if defn.policy not in registered_policies:
                missing[sid] = defn.policy

        assert not missing, (
            "Strategies reference unregistered policies:\n"
            + "\n".join(f"  {sid}: {pol!r}" for sid, pol in missing.items())
        )


# ---------------------------------------------------------------------------
# 2. build_pipeline_config – key shape validation
# ---------------------------------------------------------------------------


class TestBuildPipelineConfigShape:
    """StrategyLoader.build_pipeline_config must produce all expected private keys."""

    _REQUIRED_META_KEYS = ("_strategy_id", "_asset_class", "_agents", "_rebalance_freq")

    def _check_meta_keys(self, merged: dict[str, Any], strategy_id: str) -> None:
        for key in self._REQUIRED_META_KEYS:
            assert key in merged, (
                f"build_pipeline_config for {strategy_id!r} did not produce {key!r}"
            )

    def test_crypto_momentum_v1_meta_keys(self):
        from spectraquant_v3.strategies.loader import StrategyLoader

        merged = StrategyLoader.build_pipeline_config(
            "crypto_momentum_v1", _minimal_crypto_cfg()
        )
        self._check_meta_keys(merged, "crypto_momentum_v1")
        assert merged["_strategy_id"] == "crypto_momentum_v1"
        assert merged["_asset_class"] == "crypto"
        assert isinstance(merged["_agents"], list)
        assert len(merged["_agents"]) > 0

    def test_equity_momentum_v1_meta_keys(self):
        from spectraquant_v3.strategies.loader import StrategyLoader

        merged = StrategyLoader.build_pipeline_config(
            "equity_momentum_v1", _minimal_equity_cfg()
        )
        self._check_meta_keys(merged, "equity_momentum_v1")
        assert merged["_strategy_id"] == "equity_momentum_v1"
        assert merged["_asset_class"] == "equity"

    def test_all_registered_strategies_produce_meta_keys(self):
        from spectraquant_v3.strategies.loader import StrategyLoader
        from spectraquant_v3.strategies.registry import StrategyRegistry

        for sid in StrategyRegistry.list():
            defn = StrategyRegistry.get(sid)
            if not defn.enabled:
                continue
            base = _minimal_crypto_cfg() if defn.asset_class == "crypto" else _minimal_equity_cfg()
            merged = StrategyLoader.build_pipeline_config(sid, base)
            for key in self._REQUIRED_META_KEYS:
                assert key in merged, (
                    f"build_pipeline_config({sid!r}) missing {key!r}"
                )
            # _agents must match the registry definition
            assert merged["_agents"] == list(defn.agents), (
                f"_agents mismatch for {sid!r}: got {merged['_agents']}, "
                f"expected {list(defn.agents)}"
            )
            assert merged["_asset_class"] == defn.asset_class

    def test_agents_match_registry_definition(self):
        """_agents in merged config must equal agents in the StrategyDefinition."""
        from spectraquant_v3.strategies.loader import StrategyLoader
        from spectraquant_v3.strategies.registry import StrategyRegistry

        defn = StrategyRegistry.get("crypto_momentum_v1")
        merged = StrategyLoader.build_pipeline_config(
            "crypto_momentum_v1", _minimal_crypto_cfg()
        )
        assert merged["_agents"] == list(defn.agents)

    def test_allocator_mapping_vol_target_v1(self):
        """vol_target_v1 allocator maps to 'vol_target' pipeline mode."""
        from spectraquant_v3.strategies.loader import StrategyLoader

        # crypto_momentum_v2 uses vol_target_v1
        merged = StrategyLoader.build_pipeline_config(
            "crypto_momentum_v2", _minimal_crypto_cfg()
        )
        assert merged["portfolio"]["allocator"] == "vol_target"

    def test_allocator_mapping_rank_vol_target(self):
        """rank_vol_target_allocator maps to 'vol_target' pipeline mode."""
        from spectraquant_v3.strategies.loader import StrategyLoader

        # crypto_momentum_v1 uses rank_vol_target_allocator
        merged = StrategyLoader.build_pipeline_config(
            "crypto_momentum_v1", _minimal_crypto_cfg()
        )
        assert merged["portfolio"]["allocator"] == "vol_target"

    def test_allocator_mapping_equal_weight(self):
        """equal_weight allocator passes through as 'equal_weight'."""
        from spectraquant_v3.strategies.loader import StrategyLoader

        # equity_momentum_v1 uses equal_weight
        merged = StrategyLoader.build_pipeline_config(
            "equity_momentum_v1", _minimal_equity_cfg()
        )
        assert merged["portfolio"]["allocator"] == "equal_weight"

    def test_allocator_mapping_equal_weight_v1(self):
        """equal_weight_v1 allocator must map to 'equal_weight' pipeline mode."""
        from spectraquant_v3.strategies.loader import StrategyLoader
        from spectraquant_v3.strategies.registry import StrategyRegistry
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        # Register a temporary strategy using the equal_weight_v1 alias
        tmp = StrategyDefinition(
            strategy_id="_test_ew_v1_alias",
            asset_class="crypto",
            agents=["crypto_momentum_v1"],
            policy="confidence_filter_v1",
            allocator="equal_weight_v1",
        )
        StrategyRegistry.register(tmp)
        try:
            merged = StrategyLoader.build_pipeline_config(
                "_test_ew_v1_alias", _minimal_crypto_cfg()
            )
            assert merged["portfolio"]["allocator"] == "equal_weight"
        finally:
            StrategyRegistry.unregister("_test_ew_v1_alias")

    def test_base_cfg_not_mutated(self):
        """build_pipeline_config must not modify the original base_cfg."""
        from spectraquant_v3.strategies.loader import StrategyLoader

        original = _minimal_crypto_cfg()
        original_portfolio_copy = dict(original["portfolio"])
        StrategyLoader.build_pipeline_config("crypto_momentum_v1", original)
        # original portfolio must be unchanged
        assert original["portfolio"] == original_portfolio_copy
        assert "_strategy_id" not in original


# ---------------------------------------------------------------------------
# 3. Pipeline agent resolution uses _agents from config
# ---------------------------------------------------------------------------


class TestPipelineAgentResolution:
    """Pipelines must prefer cfg['_agents'] over cfg['strategies'][...]['agents']."""

    def test_crypto_pipeline_uses_agents_from_config(self, tmp_path):
        """When _agents is set in cfg, crypto_pipeline must use it directly."""
        from spectraquant_v3.pipeline import run_strategy

        cfg = _minimal_crypto_cfg()
        # Run via run_strategy so build_pipeline_config sets _agents
        result = run_strategy(
            "crypto_momentum_v1", cfg=cfg, dry_run=True, project_root=str(tmp_path)
        )
        assert result["status"] == "success"

    def test_equity_pipeline_uses_agents_from_config(self, tmp_path):
        """When _agents is set in cfg, equity_pipeline must use it directly."""
        from spectraquant_v3.pipeline import run_strategy

        cfg = _minimal_equity_cfg()
        result = run_strategy(
            "equity_momentum_v1", cfg=cfg, dry_run=True, project_root=str(tmp_path)
        )
        assert result["status"] == "success"

    def test_pipeline_ignores_stale_yaml_agents(self, tmp_path):
        """Pipeline must NOT fail when cfg['strategies'][id]['agents'] contains
        a non-registered agent name, as long as _agents is present with valid names."""
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline
        from spectraquant_v3.strategies.loader import StrategyLoader

        base = _minimal_crypto_cfg()
        # Inject stale YAML-style agent name that is NOT in AgentRegistry
        base["strategies"] = {
            "crypto_momentum_v1": {"agents": ["momentum_agent_v1"]}  # stale / wrong name
        }
        merged = StrategyLoader.build_pipeline_config("crypto_momentum_v1", base)
        # _agents from registry should override the stale YAML entry
        assert merged["_agents"] == ["crypto_momentum_v1"]

        # Running the pipeline should succeed because it reads _agents, not
        # cfg["strategies"][...]["agents"]
        result = run_crypto_pipeline(
            cfg=merged, dry_run=True, project_root=str(tmp_path)
        )
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# 4. strategies.yaml alignment checks
# ---------------------------------------------------------------------------


class TestStrategiesYamlAlignment:
    """The strategies.yaml config file must use field names and values that
    are consistent with the StrategyDefinition schema and the sub-registries."""

    @pytest.fixture
    def yaml_strategies(self) -> dict[str, Any]:
        """Load the strategies section from config/v3/strategies.yaml."""
        import yaml

        yaml_path = Path(__file__).parents[2] / "config" / "v3" / "strategies.yaml"
        if not yaml_path.exists():
            pytest.skip("config/v3/strategies.yaml not found")
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("strategies", {})

    def test_yaml_agent_names_are_registered(self, yaml_strategies):
        """Every agent listed in strategies.yaml must exist in AgentRegistry."""
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        registered = set(AgentRegistry.list())
        missing = {}
        for sid, definition in yaml_strategies.items():
            bad = [a for a in definition.get("agents", []) if a not in registered]
            if bad:
                missing[sid] = bad

        assert not missing, (
            "strategies.yaml references unregistered agents:\n"
            + "\n".join(f"  {sid}: {bad}" for sid, bad in missing.items())
        )

    def test_yaml_allocator_names_are_registered(self, yaml_strategies):
        """Every allocator listed in strategies.yaml must exist in AllocatorRegistry."""
        from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry

        registered = set(AllocatorRegistry.list())
        missing = {}
        for sid, definition in yaml_strategies.items():
            alloc = definition.get("allocator")
            if alloc and alloc not in registered:
                missing[sid] = alloc

        assert not missing, (
            "strategies.yaml references unregistered allocators:\n"
            + "\n".join(f"  {sid}: {alloc!r}" for sid, alloc in missing.items())
        )

    def test_yaml_uses_universe_profile_not_universe_mode(self, yaml_strategies):
        """strategies.yaml must use 'universe_profile' (not 'universe_mode')."""
        wrong = [
            sid for sid, defn in yaml_strategies.items() if "universe_mode" in defn
        ]
        assert not wrong, (
            "strategies.yaml uses 'universe_mode' (wrong field name) for: "
            + str(wrong)
            + ". Use 'universe_profile' to match StrategyDefinition."
        )

    def test_yaml_uses_rebalance_freq_not_rebalance_frequency(self, yaml_strategies):
        """strategies.yaml must use 'rebalance_freq' (not 'rebalance_frequency')."""
        wrong = [
            sid for sid, defn in yaml_strategies.items() if "rebalance_frequency" in defn
        ]
        assert not wrong, (
            "strategies.yaml uses 'rebalance_frequency' (wrong field name) for: "
            + str(wrong)
            + ". Use 'rebalance_freq' to match StrategyDefinition."
        )

    def test_yaml_uses_risk_config_not_risk_limits(self, yaml_strategies):
        """strategies.yaml must use 'risk_config' (not 'risk_limits')."""
        wrong = [
            sid for sid, defn in yaml_strategies.items() if "risk_limits" in defn
        ]
        assert not wrong, (
            "strategies.yaml uses 'risk_limits' (wrong field name) for: "
            + str(wrong)
            + ". Use 'risk_config' to match StrategyDefinition."
        )

    def test_yaml_uses_feature_profile_not_feature_engine(self, yaml_strategies):
        """strategies.yaml must use 'feature_profile' (not 'feature_engine')."""
        wrong = [
            sid for sid, defn in yaml_strategies.items() if "feature_engine" in defn
        ]
        assert not wrong, (
            "strategies.yaml uses 'feature_engine' (wrong field name) for: "
            + str(wrong)
            + ". Use 'feature_profile' to match StrategyDefinition."
        )


# ---------------------------------------------------------------------------
# 5. CLI-to-pipeline handoff
# ---------------------------------------------------------------------------


class TestCLIToPipelineHandoff:
    """The strategy CLI command must correctly resolve strategy → pipeline."""

    def test_strategy_run_crypto_dry_run(self, tmp_path):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        # Find the repo-level config/v3 directory relative to this test file.
        config_dir = str(Path(__file__).parents[2] / "config" / "v3")

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "strategy",
                "run",
                "crypto_momentum_v1",
                "--dry-run",
                "--mode",
                "normal",
                "--config-dir",
                config_dir,
            ],
        )
        # Success or a config-not-found exit (code 1) are both acceptable since
        # optional YAML overlays may reference external data paths in CI.
        assert result.exit_code in (0, 1), (
            f"Unexpected exit code {result.exit_code}. Output:\n{result.output}"
        )
        # The specific unknown-strategy error must NOT appear
        assert "not registered" not in result.output

    def test_strategy_run_unknown_exits_nonzero(self):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["strategy", "run", "definitely_not_a_real_strategy_xyz"],
        )
        assert result.exit_code != 0

    def test_strategy_run_invalid_mode_exits_nonzero(self):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "strategy",
                "run",
                "crypto_momentum_v1",
                "--mode",
                "invalid_mode_xyz",
            ],
        )
        assert result.exit_code != 0

    def test_strategy_list_shows_crypto_momentum_v2(self):
        """crypto_momentum_v2 must appear in strategy list output."""
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["strategy", "list"])
        assert result.exit_code == 0
        assert "crypto_momentum_v2" in result.output

    def test_strategy_show_returns_json(self):
        """strategy show must return valid JSON with all expected fields."""
        import json

        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["strategy", "show", "crypto_momentum_v1"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["strategy_id"] == "crypto_momentum_v1"
        assert data["asset_class"] == "crypto"
        assert "agents" in data
        assert "allocator" in data
        assert "risk_config" in data

    def test_strategy_show_unknown_exits_nonzero(self):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["strategy", "show", "not_a_strategy_xyz"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 6. Invalid / incomplete strategy and config combinations
# ---------------------------------------------------------------------------


class TestInvalidCombinations:
    """Incomplete or invalid strategy/config combinations must fail with clear errors."""

    def test_unregistered_agent_raises_on_load(self):
        from spectraquant_v3.strategies.loader import StrategyLoader
        from spectraquant_v3.strategies.registry import StrategyRegistry
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        bad = StrategyDefinition(
            strategy_id="_test_bad_agent",
            asset_class="crypto",
            agents=["agent_that_does_not_exist_xyz"],
            policy="confidence_filter_v1",
            allocator="equal_weight",
        )
        StrategyRegistry.register(bad)
        try:
            with pytest.raises(ValueError, match="unregistered agent"):
                StrategyLoader.load("_test_bad_agent")
        finally:
            StrategyRegistry.unregister("_test_bad_agent")

    def test_unregistered_allocator_raises_on_load(self):
        from spectraquant_v3.strategies.loader import StrategyLoader
        from spectraquant_v3.strategies.registry import StrategyRegistry
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        bad = StrategyDefinition(
            strategy_id="_test_bad_alloc",
            asset_class="crypto",
            agents=["crypto_momentum_v1"],
            policy="confidence_filter_v1",
            allocator="allocator_that_does_not_exist_xyz",
        )
        StrategyRegistry.register(bad)
        try:
            with pytest.raises(ValueError, match="unregistered allocator"):
                StrategyLoader.load("_test_bad_alloc")
        finally:
            StrategyRegistry.unregister("_test_bad_alloc")

    def test_unregistered_policy_raises_on_load(self):
        from spectraquant_v3.strategies.loader import StrategyLoader
        from spectraquant_v3.strategies.registry import StrategyRegistry
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        bad = StrategyDefinition(
            strategy_id="_test_bad_policy",
            asset_class="crypto",
            agents=["crypto_momentum_v1"],
            policy="policy_that_does_not_exist_xyz",
            allocator="equal_weight",
        )
        StrategyRegistry.register(bad)
        try:
            with pytest.raises(ValueError, match="unregistered policy"):
                StrategyLoader.load("_test_bad_policy")
        finally:
            StrategyRegistry.unregister("_test_bad_policy")

    def test_mixed_asset_class_raises(self, tmp_path):
        from spectraquant_v3.core.errors import MixedAssetClassRunError
        from spectraquant_v3.pipeline import run_strategy

        equity_cfg = _minimal_equity_cfg()
        with pytest.raises(MixedAssetClassRunError):
            run_strategy(
                "crypto_momentum_v1",
                cfg=equity_cfg,
                dry_run=True,
                project_root=str(tmp_path),
            )

    def test_crypto_strategy_with_equity_config_raises(self, tmp_path):
        """Crypto strategy must not run against equity config."""
        from spectraquant_v3.core.errors import MixedAssetClassRunError
        from spectraquant_v3.pipeline import run_strategy

        with pytest.raises(MixedAssetClassRunError):
            run_strategy(
                "crypto_momentum_v1",
                cfg=_minimal_equity_cfg(),
                dry_run=True,
                project_root=str(tmp_path),
            )

    def test_equity_strategy_with_crypto_config_raises(self, tmp_path):
        """Equity strategy must not run against crypto config."""
        from spectraquant_v3.core.errors import MixedAssetClassRunError
        from spectraquant_v3.pipeline import run_strategy

        with pytest.raises(MixedAssetClassRunError):
            run_strategy(
                "equity_momentum_v1",
                cfg=_minimal_crypto_cfg(),
                dry_run=True,
                project_root=str(tmp_path),
            )

    def test_missing_crypto_section_raises_on_validate(self):
        from spectraquant_v3.core.config import validate_crypto_config
        from spectraquant_v3.core.errors import ConfigValidationError

        cfg = _minimal_equity_cfg()  # no "crypto" key
        with pytest.raises(ConfigValidationError):
            validate_crypto_config(cfg)

    def test_missing_equities_section_raises_on_validate(self):
        from spectraquant_v3.core.config import validate_equity_config
        from spectraquant_v3.core.errors import ConfigValidationError

        cfg = _minimal_crypto_cfg()  # no "equities" key
        with pytest.raises(ConfigValidationError):
            validate_equity_config(cfg)

    def test_disabled_strategy_raises_on_load(self):
        from spectraquant_v3.strategies.loader import StrategyLoader
        from spectraquant_v3.strategies.registry import StrategyRegistry
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        disabled = StrategyDefinition(
            strategy_id="_test_disabled_combo",
            asset_class="crypto",
            agents=["crypto_momentum_v1"],
            policy="confidence_filter_v1",
            allocator="equal_weight",
            enabled=False,
        )
        StrategyRegistry.register(disabled)
        try:
            with pytest.raises(ValueError, match="disabled"):
                StrategyLoader.load("_test_disabled_combo")
        finally:
            StrategyRegistry.unregister("_test_disabled_combo")

    def test_empty_agents_list_fails_definition_validate(self):
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        defn = StrategyDefinition(
            strategy_id="empty_agents_test",
            asset_class="crypto",
            agents=[],
            policy="confidence_filter_v1",
            allocator="equal_weight",
        )
        with pytest.raises(ValueError, match="at least one agent"):
            defn.validate()

    def test_build_pipeline_config_for_unknown_raises(self):
        from spectraquant_v3.strategies.loader import StrategyLoader

        with pytest.raises(KeyError):
            StrategyLoader.build_pipeline_config(
                "totally_unknown_strategy_xyz", _minimal_crypto_cfg()
            )


# ---------------------------------------------------------------------------
# 7. crypto_momentum_v2 registration
# ---------------------------------------------------------------------------


class TestCryptoMomentumV2:
    """crypto_momentum_v2 must be registered and loadable."""

    def test_crypto_momentum_v2_registered(self):
        from spectraquant_v3.strategies.registry import StrategyRegistry

        assert "crypto_momentum_v2" in StrategyRegistry.list()

    def test_crypto_momentum_v2_loads(self):
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load("crypto_momentum_v2")
        assert defn.asset_class == "crypto"
        assert defn.allocator in {"vol_target_v1", "rank_vol_target_allocator"}

    def test_crypto_momentum_v2_pipeline_config(self):
        from spectraquant_v3.strategies.loader import StrategyLoader

        merged = StrategyLoader.build_pipeline_config(
            "crypto_momentum_v2", _minimal_crypto_cfg()
        )
        assert merged["_strategy_id"] == "crypto_momentum_v2"
        assert merged["_asset_class"] == "crypto"
        assert merged["_agents"] == ["crypto_momentum_v1"]
        # vol_target_v1 must map to "vol_target" pipeline mode
        assert merged["portfolio"]["allocator"] == "vol_target"
