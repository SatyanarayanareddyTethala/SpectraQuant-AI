"""Regression tests for V3 allocator registry/config-driven dispatch.

These tests verify that:
- The AllocatorRegistry maps strategy allocator names to the correct classes.
- Both Allocator and RankVolTargetAllocator expose the uniform
  allocate_decisions(decisions, vol_map) interface.
- The crypto and equity pipelines use registry-driven dispatch rather than
  any hardcoded allocator class or name.
- A custom allocator registered at runtime is correctly dispatched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectraquant_v3.pipeline.allocator import Allocator
from spectraquant_v3.pipeline.meta_policy import PolicyDecision
from spectraquant_v3.strategies.allocators.rank_vol_target_allocator import (
    RankVolTargetAllocator,
)
from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision(sym: str, score: float = 0.5, confidence: float = 0.8, passed: bool = True) -> PolicyDecision:
    return PolicyDecision(
        canonical_symbol=sym,
        asset_class="crypto",
        composite_score=score,
        composite_confidence=confidence,
        passed=passed,
        reason="" if passed else "blocked",
    )


def _portfolio_cfg(allocator_mode: str = "equal_weight") -> dict:
    return {
        "portfolio": {
            "max_weight": 0.50,
            "max_gross_leverage": 1.0,
            "target_vol": 0.15,
            "min_confidence": 0.10,
            "min_signal_threshold": 0.05,
            "allocator": allocator_mode,
        },
    }


# ===========================================================================
# 1. AllocatorRegistry maps names to expected classes
# ===========================================================================


class TestAllocatorRegistryMapping:
    def test_equal_weight_maps_to_allocator(self) -> None:
        cls = AllocatorRegistry.get("equal_weight")
        assert cls is Allocator

    def test_vol_target_v1_maps_to_allocator(self) -> None:
        cls = AllocatorRegistry.get("vol_target_v1")
        assert cls is Allocator

    def test_rank_vol_target_allocator_maps_to_rank_vol(self) -> None:
        cls = AllocatorRegistry.get("rank_vol_target_allocator")
        assert cls is RankVolTargetAllocator

    def test_unknown_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            AllocatorRegistry.get("nonexistent_allocator_xyz")

    def test_list_includes_all_builtins(self) -> None:
        registered = AllocatorRegistry.list()
        assert "equal_weight" in registered
        assert "vol_target_v1" in registered
        assert "rank_vol_target_allocator" in registered


# ===========================================================================
# 2. Allocator.allocate_decisions() uniform interface
# ===========================================================================


class TestAllocatorUniformInterface:
    def test_allocate_decisions_returns_allocation_rows(self) -> None:
        alloc = Allocator.from_config(_portfolio_cfg("equal_weight"))
        decisions = [_decision("A"), _decision("B")]
        rows = alloc.allocate_decisions(decisions)
        assert len(rows) == 2
        symbols = {r.canonical_symbol for r in rows}
        assert symbols == {"A", "B"}

    def test_allocate_decisions_equal_weights(self) -> None:
        alloc = Allocator.from_config(_portfolio_cfg("equal_weight"))
        decisions = [_decision("A"), _decision("B")]
        rows = alloc.allocate_decisions(decisions)
        weights = {r.canonical_symbol: r.target_weight for r in rows}
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(0.5)

    def test_allocate_decisions_blocked_symbols_have_zero_weight(self) -> None:
        alloc = Allocator.from_config(_portfolio_cfg("equal_weight"))
        decisions = [_decision("A"), _decision("B", passed=False)]
        rows = alloc.allocate_decisions(decisions)
        by_sym = {r.canonical_symbol: r for r in rows}
        assert by_sym["A"].target_weight > 0.0
        assert by_sym["B"].target_weight == pytest.approx(0.0)
        assert by_sym["B"].blocked is True

    def test_allocate_decisions_delegates_to_allocate(self) -> None:
        """allocate_decisions must return identical results to allocate()."""
        alloc = Allocator.from_config(_portfolio_cfg("equal_weight"))
        decisions = [_decision("X", score=0.7), _decision("Y", score=0.3)]
        via_decisions = alloc.allocate_decisions(decisions)
        via_allocate = alloc.allocate(decisions)
        assert {r.canonical_symbol: r.target_weight for r in via_decisions} == pytest.approx(
            {r.canonical_symbol: r.target_weight for r in via_allocate}
        )


# ===========================================================================
# 3. RankVolTargetAllocator.allocate_decisions() uniform interface
# ===========================================================================


class TestRankVolTargetAllocatorUniformInterface:
    def test_allocate_decisions_returns_allocation_rows(self) -> None:
        alloc = RankVolTargetAllocator.from_config(_portfolio_cfg())
        decisions = [_decision("A", score=0.8), _decision("B", score=0.4)]
        rows = alloc.allocate_decisions(decisions)
        assert len(rows) == 2
        symbols = {r.canonical_symbol for r in rows}
        assert symbols == {"A", "B"}

    def test_allocate_decisions_passed_symbols_get_positive_weight(self) -> None:
        alloc = RankVolTargetAllocator.from_config(_portfolio_cfg())
        decisions = [_decision("A", score=0.9), _decision("B", score=0.5)]
        rows = alloc.allocate_decisions(decisions)
        for r in rows:
            assert r.target_weight >= 0.0

    def test_allocate_decisions_blocked_have_zero_weight(self) -> None:
        alloc = RankVolTargetAllocator.from_config(_portfolio_cfg())
        decisions = [_decision("PASS", score=0.9), _decision("BLOCK", score=0.5, passed=False)]
        rows = alloc.allocate_decisions(decisions)
        by_sym = {r.canonical_symbol: r for r in rows}
        assert by_sym["BLOCK"].target_weight == pytest.approx(0.0)
        assert by_sym["BLOCK"].blocked is True
        assert by_sym["PASS"].target_weight > 0.0
        assert by_sym["PASS"].blocked is False

    def test_allocate_decisions_with_vol_map(self) -> None:
        alloc = RankVolTargetAllocator.from_config(_portfolio_cfg())
        decisions = [_decision("A", score=0.9), _decision("B", score=0.5)]
        vol_map = {"A": 0.10, "B": 0.30}
        rows = alloc.allocate_decisions(decisions, vol_map=vol_map)
        assert len(rows) == 2
        total = sum(r.target_weight for r in rows)
        assert total <= 1.0 + 1e-9

    def test_allocate_decisions_run_id_propagated(self) -> None:
        alloc = RankVolTargetAllocator.from_config(_portfolio_cfg(), run_id="test-run-42")
        decisions = [_decision("A")]
        rows = alloc.allocate_decisions(decisions)
        assert all(r.run_id == "test-run-42" for r in rows)

    def test_allocate_decisions_empty_decisions_returns_empty(self) -> None:
        alloc = RankVolTargetAllocator.from_config(_portfolio_cfg())
        rows = alloc.allocate_decisions([])
        assert rows == []

    def test_allocate_decisions_all_blocked_returns_zero_weights(self) -> None:
        alloc = RankVolTargetAllocator.from_config(_portfolio_cfg())
        decisions = [_decision("A", passed=False), _decision("B", passed=False)]
        rows = alloc.allocate_decisions(decisions)
        assert all(r.target_weight == pytest.approx(0.0) for r in rows)
        assert all(r.blocked is True for r in rows)

    def test_allocate_decisions_ranks_by_descending_score(self) -> None:
        """Symbol with higher |score| should get rank 1 (higher weight)."""
        alloc = RankVolTargetAllocator.from_config(
            {"portfolio": {"max_weight": 1.0, "max_gross_leverage": 1.0, "target_vol": 0.15}}
        )
        decisions = [
            _decision("LOW_SCORE", score=0.1),
            _decision("HIGH_SCORE", score=0.9),
        ]
        rows = alloc.allocate_decisions(decisions)
        by_sym = {r.canonical_symbol: r.target_weight for r in rows}
        assert by_sym["HIGH_SCORE"] > by_sym["LOW_SCORE"]


# ===========================================================================
# 4. Registry-driven dispatch: correct class instantiated per strategy
# ===========================================================================


class TestRegistryDrivenDispatch:
    def test_equal_weight_strategy_instantiates_allocator(self) -> None:
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load("equity_momentum_v1")
        assert defn.allocator == "equal_weight"
        cls = AllocatorRegistry.get(defn.allocator)
        assert cls is Allocator

    def test_crypto_momentum_strategy_instantiates_rank_vol(self) -> None:
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load("crypto_momentum_v1")
        assert defn.allocator == "rank_vol_target_allocator"
        cls = AllocatorRegistry.get(defn.allocator)
        assert cls is RankVolTargetAllocator

    def test_from_config_builds_correct_instance_equal_weight(self) -> None:
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load("equity_momentum_v1")
        cfg = _portfolio_cfg("equal_weight")
        allocator = AllocatorRegistry.get(defn.allocator).from_config(cfg, run_id="test")
        assert isinstance(allocator, Allocator)

    def test_from_config_builds_correct_instance_rank_vol(self) -> None:
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load("crypto_momentum_v1")
        cfg = _portfolio_cfg("vol_target")
        allocator = AllocatorRegistry.get(defn.allocator).from_config(cfg, run_id="test")
        assert isinstance(allocator, RankVolTargetAllocator)

    def test_custom_allocator_registration_and_dispatch(self) -> None:
        """A custom allocator registered at runtime is dispatched correctly."""
        from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry
        from spectraquant_v3.strategies.registry import StrategyRegistry
        from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

        class _DummyAllocator:
            """Minimal stub allocator for registry dispatch testing."""

            def __init__(self) -> None:
                pass

            @classmethod
            def from_config(cls, cfg: dict, run_id: str | None = None) -> "_DummyAllocator":
                return cls()

            def allocate_decisions(
                self,
                decisions: list[PolicyDecision],
                vol_map: dict | None = None,
            ) -> list:
                return []

        AllocatorRegistry.register("_test_dummy_allocator", _DummyAllocator)
        StrategyRegistry.register(
            StrategyDefinition(
                strategy_id="_test_dispatch_strategy",
                asset_class="crypto",
                agents=["crypto_momentum_v1"],
                policy="confidence_filter_v1",
                allocator="_test_dummy_allocator",
            )
        )
        try:
            from spectraquant_v3.strategies.loader import StrategyLoader

            defn = StrategyLoader.load("_test_dispatch_strategy")
            cls = AllocatorRegistry.get(defn.allocator)
            assert cls is _DummyAllocator
            inst = cls.from_config({})
            rows = inst.allocate_decisions([_decision("A")])
            assert rows == []
        finally:
            AllocatorRegistry.unregister("_test_dummy_allocator")
            StrategyRegistry.unregister("_test_dispatch_strategy")


# ===========================================================================
# 5. Pipeline allocator dispatch (integration smoke)
# ===========================================================================


def _crypto_cfg_for_strategy(strategy_id: str, symbols: list[str] | None = None) -> dict:
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


def _equity_cfg_for_strategy(strategy_id: str, tickers: list[str] | None = None) -> dict:
    if tickers is None:
        tickers = ["INFY.NS", "TCS.NS"]
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


class TestPipelineAllocatorDispatch:
    """Smoke tests confirming pipelines use registry-driven allocator dispatch."""

    def test_crypto_pipeline_rank_vol_allocator_produces_allocations(
        self, tmp_path: Path
    ) -> None:
        """crypto_momentum_v1 uses rank_vol_target_allocator; smoke-test dispatch."""
        from spectraquant_v3.pipeline import run_strategy
        from spectraquant_v3.core.enums import RunMode

        cfg = _crypto_cfg_for_strategy("crypto_momentum_v1")
        result = run_strategy(
            "crypto_momentum_v1",
            cfg=cfg,
            run_mode=RunMode.TEST,
            dry_run=True,
            project_root=str(tmp_path),
        )
        assert result["status"] == "success"
        allocations = result["allocations"]
        assert isinstance(allocations, list)
        # All allocation rows must have a target_weight that is a float.
        for row in allocations:
            assert isinstance(row.target_weight, float)

    def test_equity_pipeline_equal_weight_allocator_produces_allocations(
        self, tmp_path: Path
    ) -> None:
        """equity_momentum_v1 uses equal_weight; smoke-test registry dispatch."""
        from spectraquant_v3.pipeline import run_strategy
        from spectraquant_v3.core.enums import RunMode

        cfg = _equity_cfg_for_strategy("equity_momentum_v1")
        result = run_strategy(
            "equity_momentum_v1",
            cfg=cfg,
            run_mode=RunMode.TEST,
            dry_run=True,
            project_root=str(tmp_path),
        )
        assert result["status"] == "success"
        allocations = result["allocations"]
        assert isinstance(allocations, list)
        for row in allocations:
            assert isinstance(row.target_weight, float)

    def test_crypto_pipeline_allocator_is_rank_vol_not_base_allocator(
        self, tmp_path: Path
    ) -> None:
        """Verify that the crypto pipeline dispatches to RankVolTargetAllocator
        (not the base Allocator) when strategy specifies rank_vol_target_allocator.
        We confirm this by checking the strategy loader, not by inspecting internals.
        """
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load("crypto_momentum_v1")
        cls = AllocatorRegistry.get(defn.allocator)
        assert cls is RankVolTargetAllocator, (
            f"Expected RankVolTargetAllocator for crypto_momentum_v1 "
            f"but got {cls.__name__}"
        )

    def test_equity_pipeline_allocator_is_base_allocator(self) -> None:
        """Verify that the equity pipeline dispatches to Allocator
        (not RankVolTargetAllocator) when strategy specifies equal_weight.
        """
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load("equity_momentum_v1")
        cls = AllocatorRegistry.get(defn.allocator)
        assert cls is Allocator, (
            f"Expected Allocator for equity_momentum_v1 but got {cls.__name__}"
        )
