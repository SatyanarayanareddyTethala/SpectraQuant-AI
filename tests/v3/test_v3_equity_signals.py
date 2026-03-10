"""Regression tests for Stage-2 V3 equity signal agents.

Tests cover the four newly-ported agents:
- EquityBreakoutAgent       (equity_breakout_v1)
- EquityMeanReversionAgent  (equity_mean_reversion_v1)
- EquityVolatilityAgent     (equity_volatility_v1)
- EquityQualityAgent        (equity_quality_v1)

And their registration in the AgentRegistry / StrategyRegistry.

All tests are self-contained (no network calls, no file-system side-effects).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 60, seed: int = 42, *, add_trend: float = 0.0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with *n* rows."""
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.standard_normal(n)) + np.linspace(0, add_trend * n, n)
    close = np.maximum(base, 1.0)
    high = close * (1.0 + rng.uniform(0.0, 0.02, n))
    low = close * (1.0 - rng.uniform(0.0, 0.02, n))
    open_ = close * (1.0 + rng.uniform(-0.01, 0.01, n))
    volume = rng.uniform(1_000_000, 5_000_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _tiny_df(n: int = 5) -> pd.DataFrame:
    """Too-small OHLCV DataFrame for testing insufficient-rows paths."""
    return _ohlcv(n=n)


# ===========================================================================
# EquityBreakoutAgent
# ===========================================================================


class TestEquityBreakoutAgent:
    """Tests for spectraquant_v3.equities.signals.breakout.EquityBreakoutAgent."""

    def _agent(self, **kw):
        from spectraquant_v3.equities.signals.breakout import EquityBreakoutAgent
        return EquityBreakoutAgent(run_id="test_run", **kw)

    # --- basic construction -----------------------------------------------

    def test_default_construction(self) -> None:
        agent = self._agent()
        assert agent.run_id == "test_run"
        assert agent.window == 52
        assert agent.min_rows == 20

    def test_from_config(self) -> None:
        from spectraquant_v3.equities.signals.breakout import EquityBreakoutAgent
        cfg = {"equities": {"signals": {"breakout_window": 30}}}
        agent = EquityBreakoutAgent.from_config(cfg, run_id="r1")
        assert agent.window == 30

    # --- evaluate output schema -------------------------------------------

    def test_evaluate_returns_signal_row(self) -> None:
        from spectraquant_v3.core.schema import SignalRow
        agent = self._agent()
        row = agent.evaluate("INFY.NS", _ohlcv(60))
        assert isinstance(row, SignalRow)

    def test_evaluate_fields_populated(self) -> None:
        agent = self._agent()
        row = agent.evaluate("TCS.NS", _ohlcv(60))
        assert row.canonical_symbol == "TCS.NS"
        assert row.asset_class == "equity"
        assert row.agent_id == "equity_breakout_v1"
        assert row.run_id == "test_run"

    def test_evaluate_ok_status_on_sufficient_data(self) -> None:
        agent = self._agent(window=20, min_rows=20)
        row = agent.evaluate("X", _ohlcv(60))
        assert row.status == "OK"

    def test_signal_score_in_range(self) -> None:
        agent = self._agent()
        row = agent.evaluate("X", _ohlcv(60))
        assert -1.0 <= row.signal_score <= 1.0

    def test_confidence_in_range(self) -> None:
        agent = self._agent()
        row = agent.evaluate("X", _ohlcv(60))
        assert 0.0 <= row.confidence <= 1.0

    # --- insufficient data paths ------------------------------------------

    def test_no_signal_on_too_few_rows(self) -> None:
        agent = self._agent(min_rows=20)
        row = agent.evaluate("X", _tiny_df(5))
        assert row.status == "NO_SIGNAL"
        assert row.no_signal_reason == "insufficient_rows"

    def test_no_signal_on_empty_df(self) -> None:
        agent = self._agent()
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        row = agent.evaluate("X", empty)
        assert row.status == "NO_SIGNAL"

    # --- evaluate_many ---------------------------------------------------

    def test_evaluate_many_returns_one_row_per_symbol(self) -> None:
        agent = self._agent()
        fmap = {"A": _ohlcv(60), "B": _ohlcv(60, seed=7), "C": _ohlcv(60, seed=13)}
        rows = agent.evaluate_many(fmap)
        assert len(rows) == 3
        symbols = {r.canonical_symbol for r in rows}
        assert symbols == {"A", "B", "C"}

    def test_evaluate_many_empty_map(self) -> None:
        agent = self._agent()
        rows = agent.evaluate_many({})
        assert rows == []

    # --- breakout-specific logic ------------------------------------------

    def test_positive_score_near_high(self) -> None:
        """A price very close to its rolling high should yield a positive score."""
        agent = self._agent(window=10, min_rows=10)
        n = 20
        rng = np.random.default_rng(0)
        close = np.ones(n) * 100.0
        close[-1] = 105.0  # push price above the rolling high
        high = close * 1.01
        low = close * 0.99
        df = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": np.ones(n) * 1e6},
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        row = agent.evaluate("X", df)
        assert row.signal_score > 0.0, f"expected positive score, got {row.signal_score}"

    def test_error_wrapping(self) -> None:
        """Agent must return ERROR row, not raise, when df is pathological."""
        agent = self._agent()
        # DataFrame with columns but all-NaN values
        bad_df = pd.DataFrame(
            {"open": [np.nan]*30, "high": [np.nan]*30, "low": [np.nan]*30,
             "close": [np.nan]*30, "volume": [np.nan]*30}
        )
        row = agent.evaluate("BAD", bad_df)
        assert row.status in {"NO_SIGNAL", "ERROR"}


# ===========================================================================
# EquityMeanReversionAgent
# ===========================================================================


class TestEquityMeanReversionAgent:
    """Tests for spectraquant_v3.equities.signals.mean_reversion.EquityMeanReversionAgent."""

    def _agent(self, **kw):
        from spectraquant_v3.equities.signals.mean_reversion import EquityMeanReversionAgent
        return EquityMeanReversionAgent(run_id="test_run", **kw)

    # --- basic construction -----------------------------------------------

    def test_default_construction(self) -> None:
        agent = self._agent()
        assert agent.window == 20
        assert agent.z_threshold == 1.0

    def test_from_config(self) -> None:
        from spectraquant_v3.equities.signals.mean_reversion import EquityMeanReversionAgent
        cfg = {"equities": {"signals": {"mean_reversion_window": 30, "mean_reversion_z_threshold": 2.0}}}
        agent = EquityMeanReversionAgent.from_config(cfg, run_id="r1")
        assert agent.window == 30
        assert agent.z_threshold == 2.0

    # --- evaluate output schema -------------------------------------------

    def test_evaluate_returns_signal_row(self) -> None:
        from spectraquant_v3.core.schema import SignalRow
        agent = self._agent()
        row = agent.evaluate("INFY.NS", _ohlcv(60))
        assert isinstance(row, SignalRow)

    def test_evaluate_fields(self) -> None:
        agent = self._agent()
        row = agent.evaluate("HDFCBANK.NS", _ohlcv(60))
        assert row.agent_id == "equity_mean_reversion_v1"
        assert row.asset_class == "equity"

    def test_ok_status_on_sufficient_data(self) -> None:
        agent = self._agent()
        row = agent.evaluate("X", _ohlcv(60))
        assert row.status == "OK"

    def test_score_in_range(self) -> None:
        agent = self._agent()
        row = agent.evaluate("X", _ohlcv(60))
        assert -1.0 <= row.signal_score <= 1.0

    def test_confidence_in_range(self) -> None:
        agent = self._agent()
        row = agent.evaluate("X", _ohlcv(60))
        assert 0.0 <= row.confidence <= 1.0

    # --- direction of signal ----------------------------------------------

    def test_positive_score_below_mean(self) -> None:
        """Price 2 std below mean → positive (buy) signal."""
        n = 40
        close = np.ones(n) * 100.0
        # Drop price at the end so it's well below the rolling mean
        close[-1] = 80.0
        df = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.ones(n) * 1e6},
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        agent = self._agent(window=20)
        row = agent.evaluate("X", df)
        assert row.signal_score > 0.0, f"expected positive score, got {row.signal_score}"

    def test_negative_score_above_mean(self) -> None:
        """Price 2 std above mean → negative (sell) signal."""
        n = 40
        close = np.ones(n) * 100.0
        close[-1] = 120.0
        df = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.ones(n) * 1e6},
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        agent = self._agent(window=20)
        row = agent.evaluate("X", df)
        assert row.signal_score < 0.0, f"expected negative score, got {row.signal_score}"

    # --- insufficient data ------------------------------------------------

    def test_no_signal_on_too_few_rows(self) -> None:
        agent = self._agent(min_rows=20)
        row = agent.evaluate("X", _tiny_df(5))
        assert row.status == "NO_SIGNAL"
        assert row.no_signal_reason == "insufficient_rows"

    def test_no_signal_on_missing_close(self) -> None:
        df = pd.DataFrame(
            {"open": [100.0]*25, "high": [101.0]*25, "low": [99.0]*25, "volume": [1e6]*25}
        )
        agent = self._agent()
        row = agent.evaluate("X", df)
        assert row.status == "NO_SIGNAL"

    # --- evaluate_many ---------------------------------------------------

    def test_evaluate_many_count(self) -> None:
        agent = self._agent()
        fmap = {"A": _ohlcv(60), "B": _ohlcv(60, seed=7)}
        rows = agent.evaluate_many(fmap)
        assert len(rows) == 2


# ===========================================================================
# EquityVolatilityAgent
# ===========================================================================


class TestEquityVolatilityAgent:
    """Tests for spectraquant_v3.equities.signals.volatility.EquityVolatilityAgent."""

    def _agent(self, **kw):
        from spectraquant_v3.equities.signals.volatility import EquityVolatilityAgent
        return EquityVolatilityAgent(run_id="test_run", **kw)

    # --- basic construction -----------------------------------------------

    def test_default_construction(self) -> None:
        agent = self._agent()
        assert agent.window == 20
        assert agent.neutral_vol == 0.15

    def test_from_config(self) -> None:
        from spectraquant_v3.equities.signals.volatility import EquityVolatilityAgent
        cfg = {"equities": {"signals": {"volatility_window": 30, "volatility_neutral_vol": 0.20}}}
        agent = EquityVolatilityAgent.from_config(cfg, run_id="r1")
        assert agent.window == 30
        assert agent.neutral_vol == pytest.approx(0.20)

    # --- evaluate output schema -------------------------------------------

    def test_evaluate_returns_signal_row(self) -> None:
        from spectraquant_v3.core.schema import SignalRow
        agent = self._agent()
        row = agent.evaluate("WIPRO.NS", _ohlcv(60))
        assert isinstance(row, SignalRow)

    def test_evaluate_fields(self) -> None:
        agent = self._agent()
        row = agent.evaluate("X", _ohlcv(60))
        assert row.agent_id == "equity_volatility_v1"
        assert row.asset_class == "equity"

    def test_score_in_range(self) -> None:
        agent = self._agent()
        row = agent.evaluate("X", _ohlcv(60))
        assert -1.0 <= row.signal_score <= 1.0

    def test_confidence_is_fixed(self) -> None:
        """Confidence is structurally fixed at 0.7."""
        agent = self._agent()
        row = agent.evaluate("X", _ohlcv(60))
        # confidence is 0.7 for OK signals
        if row.status == "OK":
            assert row.confidence == pytest.approx(0.7)

    # --- volatility logic -------------------------------------------------

    def test_neutral_score_for_low_vol(self) -> None:
        """Very smooth price series → low ann_vol → score = 0.0."""
        n = 40
        # Constant price → zero returns → zero vol
        close = np.ones(n) * 100.0
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.ones(n) * 1e6},
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        agent = self._agent(window=20, neutral_vol=0.15)
        row = agent.evaluate("X", df)
        assert row.status in {"OK", "NO_SIGNAL"}  # zero variance → NaN std possible
        if row.status == "OK":
            assert row.signal_score == pytest.approx(0.0, abs=1e-6)

    def test_negative_score_for_high_vol(self) -> None:
        """Highly volatile price series → large ann_vol → negative score."""
        n = 50
        rng = np.random.default_rng(123)
        # Large random steps to push ann_vol well above 0.15
        close = 100.0 + np.cumsum(rng.standard_normal(n) * 3.0)
        close = np.maximum(close, 1.0)
        df = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.ones(n) * 1e6},
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        agent = self._agent(window=20, neutral_vol=0.05)  # low threshold
        row = agent.evaluate("X", df)
        assert row.status == "OK"
        assert row.signal_score <= 0.0, f"expected non-positive score, got {row.signal_score}"

    # --- insufficient data ------------------------------------------------

    def test_no_signal_on_too_few_rows(self) -> None:
        agent = self._agent(min_rows=21)
        row = agent.evaluate("X", _tiny_df(5))
        assert row.status == "NO_SIGNAL"
        assert row.no_signal_reason == "insufficient_rows"

    # --- evaluate_many ---------------------------------------------------

    def test_evaluate_many_count(self) -> None:
        agent = self._agent()
        fmap = {"A": _ohlcv(60), "B": _ohlcv(60, seed=7), "C": _ohlcv(60, seed=99)}
        rows = agent.evaluate_many(fmap)
        assert len(rows) == 3


# ===========================================================================
# EquityQualityAgent
# ===========================================================================


class TestEquityQualityAgent:
    """Tests for spectraquant_v3.equities.signals.quality.EquityQualityAgent."""

    def _agent(self, **kw):
        from spectraquant_v3.equities.signals.quality import EquityQualityAgent
        return EquityQualityAgent(run_id="test_run", **kw)

    # --- basic construction -----------------------------------------------

    def test_default_construction(self) -> None:
        agent = self._agent()
        assert agent.min_rows == 60
        assert agent.max_zero_return_fraction == pytest.approx(0.10)

    def test_from_config(self) -> None:
        from spectraquant_v3.equities.signals.quality import EquityQualityAgent
        cfg = {
            "equities": {
                "quality_gate": {"min_history_days": 120},
                "signals": {"quality_max_zero_fraction": 0.05},
            }
        }
        agent = EquityQualityAgent.from_config(cfg, run_id="r1")
        assert agent.min_rows == 120
        assert agent.max_zero_return_fraction == pytest.approx(0.05)

    # --- evaluate output schema -------------------------------------------

    def test_evaluate_returns_signal_row(self) -> None:
        from spectraquant_v3.core.schema import SignalRow
        agent = self._agent(min_rows=60)
        row = agent.evaluate("INFY.NS", _ohlcv(100))
        assert isinstance(row, SignalRow)

    def test_evaluate_fields(self) -> None:
        agent = self._agent(min_rows=60)
        row = agent.evaluate("TCS.NS", _ohlcv(100))
        assert row.agent_id == "equity_quality_v1"
        assert row.asset_class == "equity"

    # --- pass case --------------------------------------------------------

    def test_ok_on_good_data(self) -> None:
        agent = self._agent(min_rows=60)
        row = agent.evaluate("X", _ohlcv(120))
        assert row.status == "OK"
        assert row.signal_score == pytest.approx(0.0)
        assert row.confidence == pytest.approx(0.9)

    # --- fail cases -------------------------------------------------------

    def test_no_signal_on_insufficient_rows(self) -> None:
        agent = self._agent(min_rows=60)
        row = agent.evaluate("X", _ohlcv(30))
        assert row.status == "NO_SIGNAL"
        assert row.no_signal_reason == "insufficient_rows"

    def test_no_signal_on_too_many_zero_returns(self) -> None:
        """Stale prices with many zero returns should fail the quality gate."""
        n = 80
        # Constant price → many zero returns
        close = np.ones(n) * 100.0
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.ones(n) * 1e6},
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        agent = self._agent(min_rows=60, max_zero_return_fraction=0.10)
        row = agent.evaluate("X", df)
        assert row.status == "NO_SIGNAL"

    def test_no_signal_on_missing_close_column(self) -> None:
        df = pd.DataFrame(
            {"open": [100.0]*80, "high": [101.0]*80, "low": [99.0]*80,
             "volume": [1e6]*80}
        )
        agent = self._agent()
        row = agent.evaluate("X", df)
        assert row.status == "NO_SIGNAL"
        assert row.no_signal_reason == "missing_inputs"

    # --- evaluate_many ---------------------------------------------------

    def test_evaluate_many_returns_one_row_per_symbol(self) -> None:
        agent = self._agent(min_rows=60)
        fmap = {"A": _ohlcv(120), "B": _ohlcv(120, seed=7)}
        rows = agent.evaluate_many(fmap)
        assert len(rows) == 2
        assert all(r.status == "OK" for r in rows)

    def test_evaluate_many_with_failing_symbol(self) -> None:
        agent = self._agent(min_rows=60)
        fmap = {"good": _ohlcv(120), "bad": _ohlcv(10)}  # bad has too few rows
        rows = agent.evaluate_many(fmap)
        statuses = {r.canonical_symbol: r.status for r in rows}
        assert statuses["good"] == "OK"
        assert statuses["bad"] == "NO_SIGNAL"


# ===========================================================================
# AgentRegistry integration
# ===========================================================================


class TestAgentRegistryIntegration:
    """New agents are accessible via AgentRegistry after module import."""

    def test_all_new_agents_registered(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry
        expected = {
            "equity_breakout_v1",
            "equity_mean_reversion_v1",
            "equity_volatility_v1",
            "equity_quality_v1",
        }
        registered = set(AgentRegistry.list())
        missing = expected - registered
        assert not missing, f"Agents not registered: {missing}"

    def test_breakout_agent_build(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry
        agent = AgentRegistry.build("equity_breakout_v1", run_id="r1")
        assert agent.__class__.__name__ == "EquityBreakoutAgent"

    def test_mean_reversion_agent_build(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry
        agent = AgentRegistry.build("equity_mean_reversion_v1", run_id="r1")
        assert agent.__class__.__name__ == "EquityMeanReversionAgent"

    def test_volatility_agent_build(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry
        agent = AgentRegistry.build("equity_volatility_v1", run_id="r1")
        assert agent.__class__.__name__ == "EquityVolatilityAgent"

    def test_quality_agent_build(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry
        agent = AgentRegistry.build("equity_quality_v1", run_id="r1")
        assert agent.__class__.__name__ == "EquityQualityAgent"


# ===========================================================================
# StrategyRegistry integration
# ===========================================================================


class TestStrategyRegistryIntegration:
    """New strategies are accessible via StrategyRegistry after module import."""

    def test_all_new_strategies_registered(self) -> None:
        from spectraquant_v3.strategies.registry import StrategyRegistry
        expected = {
            "equity_breakout_v1",
            "equity_mean_reversion_v1",
            "equity_volatility_v1",
            "equity_quality_v1",
        }
        registered = set(StrategyRegistry.list())
        missing = expected - registered
        assert not missing, f"Strategies not registered: {missing}"

    def test_breakout_strategy_asset_class(self) -> None:
        from spectraquant_v3.strategies.registry import StrategyRegistry
        defn = StrategyRegistry.get("equity_breakout_v1")
        assert defn.asset_class == "equity"

    def test_mean_reversion_strategy_tags(self) -> None:
        from spectraquant_v3.strategies.registry import StrategyRegistry
        defn = StrategyRegistry.get("equity_mean_reversion_v1")
        assert "mean_reversion" in defn.tags

    def test_volatility_strategy_tags(self) -> None:
        from spectraquant_v3.strategies.registry import StrategyRegistry
        defn = StrategyRegistry.get("equity_volatility_v1")
        assert "risk_off" in defn.tags

    def test_quality_strategy_agents(self) -> None:
        from spectraquant_v3.strategies.registry import StrategyRegistry
        defn = StrategyRegistry.get("equity_quality_v1")
        assert "equity_quality_v1" in defn.agents


# ===========================================================================
# V3 contract compliance: SignalRow clamping and field rules
# ===========================================================================


class TestSignalRowContractCompliance:
    """All new agents must emit SignalRow instances compliant with V3 contracts."""

    @pytest.mark.parametrize("agent_name,module_path", [
        ("EquityBreakoutAgent", "spectraquant_v3.equities.signals.breakout"),
        ("EquityMeanReversionAgent", "spectraquant_v3.equities.signals.mean_reversion"),
        ("EquityVolatilityAgent", "spectraquant_v3.equities.signals.volatility"),
        ("EquityQualityAgent", "spectraquant_v3.equities.signals.quality"),
    ])
    def test_score_always_clamped(self, agent_name: str, module_path: str) -> None:
        """score must always be in [-1, 1] after agent evaluation."""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, agent_name)
        agent = cls(run_id="r1")
        row = agent.evaluate("SYM", _ohlcv(120))
        assert -1.0 <= row.signal_score <= 1.0

    @pytest.mark.parametrize("agent_name,module_path", [
        ("EquityBreakoutAgent", "spectraquant_v3.equities.signals.breakout"),
        ("EquityMeanReversionAgent", "spectraquant_v3.equities.signals.mean_reversion"),
        ("EquityVolatilityAgent", "spectraquant_v3.equities.signals.volatility"),
        ("EquityQualityAgent", "spectraquant_v3.equities.signals.quality"),
    ])
    def test_confidence_always_clamped(self, agent_name: str, module_path: str) -> None:
        """confidence must always be in [0, 1]."""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, agent_name)
        agent = cls(run_id="r1")
        row = agent.evaluate("SYM", _ohlcv(120))
        assert 0.0 <= row.confidence <= 1.0

    @pytest.mark.parametrize("agent_name,module_path", [
        ("EquityBreakoutAgent", "spectraquant_v3.equities.signals.breakout"),
        ("EquityMeanReversionAgent", "spectraquant_v3.equities.signals.mean_reversion"),
        ("EquityVolatilityAgent", "spectraquant_v3.equities.signals.volatility"),
        ("EquityQualityAgent", "spectraquant_v3.equities.signals.quality"),
    ])
    def test_asset_class_is_equity(self, agent_name: str, module_path: str) -> None:
        """All new agents must tag their output as EQUITY."""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, agent_name)
        agent = cls(run_id="r1")
        row = agent.evaluate("SYM", _ohlcv(120))
        assert row.asset_class == "equity"

    @pytest.mark.parametrize("agent_name,module_path", [
        ("EquityBreakoutAgent", "spectraquant_v3.equities.signals.breakout"),
        ("EquityMeanReversionAgent", "spectraquant_v3.equities.signals.mean_reversion"),
        ("EquityVolatilityAgent", "spectraquant_v3.equities.signals.volatility"),
        ("EquityQualityAgent", "spectraquant_v3.equities.signals.quality"),
    ])
    def test_no_crypto_import(self, agent_name: str, module_path: str) -> None:
        """Agent modules must not import from spectraquant_v3.crypto."""
        import importlib
        import ast
        mod = importlib.import_module(module_path)
        source_file = getattr(mod, "__file__", None)
        if source_file:
            text = open(source_file).read()
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # ast.ImportFrom has a .module attribute
                    module_name = getattr(node, "module", "") or ""
                    for alias in getattr(node, "names", []):
                        full = f"{module_name}.{alias.name}" if module_name else alias.name
                        assert not full.startswith("spectraquant_v3.crypto"), (
                            f"{module_path} must not import from spectraquant_v3.crypto; "
                            f"found import of '{full}'"
                        )
                    if module_name.startswith("spectraquant_v3.crypto"):
                        raise AssertionError(
                            f"{module_path} must not import from spectraquant_v3.crypto; "
                            f"found 'from {module_name} import ...'"
                        )

    @pytest.mark.parametrize("agent_name,module_path", [
        ("EquityBreakoutAgent", "spectraquant_v3.equities.signals.breakout"),
        ("EquityMeanReversionAgent", "spectraquant_v3.equities.signals.mean_reversion"),
        ("EquityVolatilityAgent", "spectraquant_v3.equities.signals.volatility"),
        ("EquityQualityAgent", "spectraquant_v3.equities.signals.quality"),
    ])
    def test_error_wrapping_on_pathological_input(
        self, agent_name: str, module_path: str
    ) -> None:
        """Agents must not raise; they must return a NO_SIGNAL or ERROR SignalRow."""
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, agent_name)
        agent = cls(run_id="r1")
        # Pass a completely empty DataFrame
        empty = pd.DataFrame()
        row = agent.evaluate("BAD", empty)
        assert row.status in {"NO_SIGNAL", "ERROR"}
