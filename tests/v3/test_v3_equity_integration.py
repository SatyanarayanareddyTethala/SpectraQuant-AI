"""V3 equity integration audit tests.

Verifies that the 4 newly-ported equity signal agents are fully integrated
into the V3 pipeline flow — not just registered, but actually reachable and
correctly exercised through:

  StrategyRegistry → StrategyLoader.load() → run_strategy() → run_equity_pipeline()
  → AgentRegistry lookup → agent.evaluate_many() → SignalRow output

All tests are self-contained (no network calls, no file-system side-effects
beyond tmp_path fixtures).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100.0 + np.cumsum(rng.standard_normal(n))
    c = np.maximum(c, 1.0)
    return pd.DataFrame(
        {
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": np.ones(n) * 1e6,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
    )


def _equity_cfg(tickers: list[str] | None = None) -> dict:
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
            "universe": {
                "tickers": tickers or ["INFY.NS", "TCS.NS"],
                "exclude": [],
            },
            "quality_gate": {
                "min_price": 0,
                "min_avg_volume": 0,
                "min_history_days": 0,
            },
            "signals": {"momentum_lookback": 20, "rsi_period": 14},
        },
    }


_NEW_EQUITY_STRATEGIES = [
    "equity_breakout_v1",
    "equity_mean_reversion_v1",
    "equity_volatility_v1",
    "equity_quality_v1",
]

_NEW_EQUITY_AGENTS = [
    ("equity_breakout_v1", "EquityBreakoutAgent"),
    ("equity_mean_reversion_v1", "EquityMeanReversionAgent"),
    ("equity_volatility_v1", "EquityVolatilityAgent"),
    ("equity_quality_v1", "EquityQualityAgent"),
]


# ===========================================================================
# 1. StrategyLoader.load() validates all 4 new strategies
# ===========================================================================


class TestStrategyLoaderNewEquityStrategies:
    """StrategyLoader.load() must succeed for every newly-registered strategy
    and confirm all referenced components (agents, policy, allocator) exist."""

    @pytest.mark.parametrize("strategy_id", _NEW_EQUITY_STRATEGIES)
    def test_load_succeeds(self, strategy_id: str) -> None:
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load(strategy_id)
        assert defn.strategy_id == strategy_id

    @pytest.mark.parametrize("strategy_id", _NEW_EQUITY_STRATEGIES)
    def test_asset_class_is_equity(self, strategy_id: str) -> None:
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load(strategy_id)
        assert defn.asset_class == "equity"

    @pytest.mark.parametrize("strategy_id", _NEW_EQUITY_STRATEGIES)
    def test_agent_is_registered(self, strategy_id: str) -> None:
        """Every agent referenced by the strategy must be in AgentRegistry."""
        from spectraquant_v3.strategies.agents.registry import AgentRegistry
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load(strategy_id)
        for agent_name in defn.agents:
            # AgentRegistry.get raises KeyError if missing
            cls = AgentRegistry.get(agent_name)
            assert cls is not None

    @pytest.mark.parametrize("strategy_id", _NEW_EQUITY_STRATEGIES)
    def test_policy_is_registered(self, strategy_id: str) -> None:
        from spectraquant_v3.strategies.loader import StrategyLoader
        from spectraquant_v3.strategies.policies.registry import PolicyRegistry

        defn = StrategyLoader.load(strategy_id)
        assert defn.policy in PolicyRegistry.list()

    @pytest.mark.parametrize("strategy_id", _NEW_EQUITY_STRATEGIES)
    def test_allocator_is_registered(self, strategy_id: str) -> None:
        from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load(strategy_id)
        assert defn.allocator in AllocatorRegistry.list()

    @pytest.mark.parametrize("strategy_id", _NEW_EQUITY_STRATEGIES)
    def test_build_pipeline_config_sets_strategy_id(self, strategy_id: str) -> None:
        """build_pipeline_config must stamp _strategy_id into the merged config."""
        from spectraquant_v3.strategies.loader import StrategyLoader

        merged = StrategyLoader.build_pipeline_config(strategy_id, _equity_cfg())
        assert merged.get("_strategy_id") == strategy_id


# ===========================================================================
# 2. AgentRegistry.build() produces the right class for each new strategy agent
# ===========================================================================


class TestAgentRegistryLookup:
    """AgentRegistry must return the correct class for each new equity agent."""

    @pytest.mark.parametrize("agent_id,class_name", _NEW_EQUITY_AGENTS)
    def test_registry_returns_correct_class(
        self, agent_id: str, class_name: str
    ) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        cls = AgentRegistry.get(agent_id)
        assert cls.__name__ == class_name

    @pytest.mark.parametrize("agent_id,_", _NEW_EQUITY_AGENTS)
    def test_registry_build_instantiates(self, agent_id: str, _: str) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        agent = AgentRegistry.build(agent_id, run_id="audit_run")
        assert agent.run_id == "audit_run"


# ===========================================================================
# 3. equity_pipeline.py dispatches to the correct agent via _strategy_id
# ===========================================================================


class TestEquityPipelineAgentDispatch:
    """run_equity_pipeline must route signals through the correct agent
    class when _strategy_id is set in the config."""

    @pytest.mark.parametrize("strategy_id,agent_id", [
        ("equity_breakout_v1", "equity_breakout_v1"),
        ("equity_mean_reversion_v1", "equity_mean_reversion_v1"),
        ("equity_volatility_v1", "equity_volatility_v1"),
        ("equity_quality_v1", "equity_quality_v1"),
    ])
    def test_signals_carry_correct_agent_id(
        self, strategy_id: str, agent_id: str, tmp_path: Path
    ) -> None:
        """When _strategy_id is injected into cfg, signal rows must carry
        the matching agent_id."""
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline
        from spectraquant_v3.strategies.loader import StrategyLoader

        base_cfg = _equity_cfg(["INFY.NS", "TCS.NS"])
        merged_cfg = StrategyLoader.build_pipeline_config(strategy_id, base_cfg)

        price_data = {
            "INFY.NS": _ohlcv(60, seed=1),
            "TCS.NS": _ohlcv(60, seed=2),
        }
        result = run_equity_pipeline(
            merged_cfg,
            dry_run=True,
            price_data=price_data,
            run_id=f"audit_{strategy_id}",
            project_root=str(tmp_path),
        )

        assert result["status"] == "success"
        signal_agent_ids = {s.agent_id for s in result["signals"]}
        assert agent_id in signal_agent_ids, (
            f"Expected agent_id={agent_id!r} in signals; got {signal_agent_ids}"
        )

    def test_default_strategy_id_uses_equity_momentum(self, tmp_path: Path) -> None:
        """When _strategy_id is absent the pipeline defaults to equity_momentum_v1."""
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        cfg = _equity_cfg(["INFY.NS"])
        price_data = {"INFY.NS": _ohlcv(60)}
        result = run_equity_pipeline(
            cfg,
            dry_run=True,
            price_data=price_data,
            project_root=str(tmp_path),
        )
        assert result["status"] == "success"
        agent_ids = {s.agent_id for s in result["signals"]}
        assert "equity_momentum_v1" in agent_ids

    def test_pipeline_emits_one_signal_per_symbol(self, tmp_path: Path) -> None:
        """Each new equity strategy must emit exactly one signal per universe symbol."""
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline
        from spectraquant_v3.strategies.loader import StrategyLoader

        tickers = ["INFY.NS", "TCS.NS", "WIPRO.NS"]
        base_cfg = _equity_cfg(tickers)
        price_data = {t: _ohlcv(60, seed=i) for i, t in enumerate(tickers)}

        for strategy_id in _NEW_EQUITY_STRATEGIES:
            merged_cfg = StrategyLoader.build_pipeline_config(strategy_id, base_cfg)
            result = run_equity_pipeline(
                merged_cfg,
                dry_run=True,
                price_data=price_data,
                run_id=f"count_{strategy_id}",
                project_root=str(tmp_path),
            )
            assert result["status"] == "success"
            assert len(result["signals"]) == len(tickers), (
                f"{strategy_id}: expected {len(tickers)} signals, "
                f"got {len(result['signals'])}"
            )


# ===========================================================================
# 4. run_strategy() full end-to-end path for all 4 new agents
# ===========================================================================


class TestRunStrategyNewEquityAgents:
    """run_strategy() must succeed for all 4 new equity strategies and
    produce signals whose agent_id matches the strategy's registered agent."""

    @pytest.mark.parametrize("strategy_id", _NEW_EQUITY_STRATEGIES)
    def test_run_strategy_succeeds(self, strategy_id: str, tmp_path: Path) -> None:
        from spectraquant_v3.pipeline import run_strategy

        cfg = _equity_cfg(["INFY.NS", "TCS.NS"])
        price_data = {
            "INFY.NS": _ohlcv(60, seed=10),
            "TCS.NS": _ohlcv(60, seed=11),
        }
        result = run_strategy(
            strategy_id=strategy_id,
            cfg=cfg,
            dry_run=True,
            price_data=price_data,
            project_root=str(tmp_path),
        )
        assert result["status"] == "success"
        assert result["strategy_id"] == strategy_id

    @pytest.mark.parametrize("strategy_id", _NEW_EQUITY_STRATEGIES)
    def test_run_strategy_signal_agent_id_matches(
        self, strategy_id: str, tmp_path: Path
    ) -> None:
        """Signals emitted by run_strategy must carry the correct agent_id."""
        from spectraquant_v3.pipeline import run_strategy

        cfg = _equity_cfg(["INFY.NS"])
        price_data = {"INFY.NS": _ohlcv(60)}
        result = run_strategy(
            strategy_id=strategy_id,
            cfg=cfg,
            dry_run=True,
            price_data=price_data,
            project_root=str(tmp_path),
        )
        agent_ids = {s.agent_id for s in result["signals"]}
        assert strategy_id in agent_ids, (
            f"Expected agent_id={strategy_id!r} in signal rows; got {agent_ids}"
        )

    @pytest.mark.parametrize("strategy_id", _NEW_EQUITY_STRATEGIES)
    def test_run_strategy_signal_scores_valid(
        self, strategy_id: str, tmp_path: Path
    ) -> None:
        """Signal scores must be within [-1, +1] and confidence within [0, 1]."""
        from spectraquant_v3.pipeline import run_strategy

        cfg = _equity_cfg(["INFY.NS", "TCS.NS"])
        price_data = {
            "INFY.NS": _ohlcv(60, seed=20),
            "TCS.NS": _ohlcv(60, seed=21),
        }
        result = run_strategy(
            strategy_id=strategy_id,
            cfg=cfg,
            dry_run=True,
            price_data=price_data,
            project_root=str(tmp_path),
        )
        for sig in result["signals"]:
            assert -1.0 <= sig.signal_score <= 1.0, (
                f"{strategy_id}: signal_score={sig.signal_score} out of range"
            )
            assert 0.0 <= sig.confidence <= 1.0, (
                f"{strategy_id}: confidence={sig.confidence} out of range"
            )

    @pytest.mark.parametrize("strategy_id", _NEW_EQUITY_STRATEGIES)
    def test_run_strategy_no_price_data_still_succeeds(
        self, strategy_id: str, tmp_path: Path
    ) -> None:
        """run_strategy must succeed (with NO_SIGNAL rows) even when no price
        data is provided, matching the behavior of the momentum pipeline."""
        from spectraquant_v3.pipeline import run_strategy

        cfg = _equity_cfg(["INFY.NS"])
        result = run_strategy(
            strategy_id=strategy_id,
            cfg=cfg,
            dry_run=True,
            project_root=str(tmp_path),
        )
        assert result["status"] == "success"
        # All signals should be NO_SIGNAL when there is no price data
        assert all(s.status == "NO_SIGNAL" for s in result["signals"])

    def test_run_strategy_wrong_asset_class_raises(self, tmp_path: Path) -> None:
        """Equity strategies must be rejected when a crypto config is passed."""
        from spectraquant_v3.core.errors import MixedAssetClassRunError
        from spectraquant_v3.pipeline import run_strategy

        crypto_cfg = {
            "run": {"mode": "normal"},
            "cache": {"root": "data/cache"},
            "qa": {"min_ohlcv_coverage": 1.0},
            "execution": {"mode": "paper"},
            "portfolio": {
                "max_weight": 0.25, "max_gross_leverage": 1.0,
                "min_confidence": 0.10, "min_signal_threshold": 0.05,
                "target_vol": 0.15, "allocator": "equal_weight",
            },
            "crypto": {
                "symbols": ["BTC"],
                "primary_ohlcv_provider": "ccxt",
                "universe_mode": "static",
                "quality_gate": {
                    "min_market_cap_usd": 0, "min_24h_volume_usd": 0,
                    "min_age_days": 0, "require_tradable_mapping": True,
                },
                "signals": {"momentum_lookback": 20},
            },
        }
        with pytest.raises(MixedAssetClassRunError):
            run_strategy(
                strategy_id="equity_breakout_v1",
                cfg=crypto_cfg,
                dry_run=True,
                project_root=str(tmp_path),
            )


# ===========================================================================
# 5. equity_pipeline.py does not import from spectraquant_v3.crypto
#    (regression guard for the new imports added to equity_pipeline.py)
# ===========================================================================


class TestEquityPipelineImportIntegrity:
    def test_equity_pipeline_does_not_import_crypto_after_fix(self) -> None:
        """Confirm that adding AgentRegistry/run_signal_agent imports to
        equity_pipeline.py did NOT introduce any crypto imports."""
        import ast
        import spectraquant_v3.pipeline.equity_pipeline as ep

        src_file = ep.__file__
        text = open(src_file).read()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("spectraquant_v3.crypto"), (
                    f"equity_pipeline.py must not import from spectraquant_v3.crypto; "
                    f"found: from {module} import ..."
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("spectraquant_v3.crypto"), (
                        f"equity_pipeline.py must not import spectraquant_v3.crypto; "
                        f"found: import {alias.name}"
                    )

    def test_equity_pipeline_imports_agent_registry(self) -> None:
        """AgentRegistry must now be imported by equity_pipeline.py."""
        import spectraquant_v3.pipeline.equity_pipeline as ep

        src_file = ep.__file__
        text = open(src_file).read()
        assert "AgentRegistry" in text, (
            "equity_pipeline.py must import AgentRegistry to support dynamic agent dispatch"
        )

    def test_equity_pipeline_imports_run_signal_agent(self) -> None:
        """run_signal_agent must now be imported by equity_pipeline.py."""
        import spectraquant_v3.pipeline.equity_pipeline as ep

        src_file = ep.__file__
        text = open(src_file).read()
        assert "run_signal_agent" in text, (
            "equity_pipeline.py must import run_signal_agent"
        )
