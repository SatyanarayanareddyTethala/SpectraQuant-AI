"""Tests for the V3 equity momentum + news + volatility-gate hybrid strategy.

Tests covering:
- Strategy and agent registration
- Hybrid score composition math (blend, no-news fallback, vol gate)
- EquityNewsSentimentAgent degradation paths (no-news, NaN, below threshold)
- EquityMomentumNewsHybridAgent degradation paths
- Pipeline reaches the correct agent via run_strategy
- Baseline equity_momentum_v1 still works (regression guard)
- No crypto imports in equity hybrid modules

All tests are self-contained (no network calls, no file-system side-effects).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _feature_df(
    n: int = 40,
    seed: int = 7,
    *,
    mom_val: float = 0.05,
    rsi_val: float = 55.0,
    news_score: float | None = 0.3,
    vol_realised: float | None = 0.18,
) -> pd.DataFrame:
    """Minimal feature DataFrame with columns expected by the hybrid agent.

    Momentum column ``ret_20d`` and ``rsi`` are synthetic constants so tests
    are deterministic regardless of OHLCV randomness.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    data: dict[str, list[float]] = {
        "ret_20d": [mom_val] * n,
        "rsi": [rsi_val] * n,
    }
    if news_score is not None:
        data["news_sentiment_score"] = [news_score] * n
    if vol_realised is not None:
        data["vol_realised"] = [vol_realised] * n
    return pd.DataFrame(data, index=idx)


def _ohlcv(n: int = 40, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.standard_normal(n))
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


# ===========================================================================
# Registry tests
# ===========================================================================


class TestRegistration:
    """Strategy and agent registration sanity checks."""

    def test_hybrid_strategy_in_strategy_registry(self) -> None:
        from spectraquant_v3.strategies.registry import StrategyRegistry

        assert "equity_momentum_news_hybrid_v1" in StrategyRegistry.list()

    def test_hybrid_agent_in_agent_registry(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        assert "equity_momentum_news_hybrid_v1" in AgentRegistry.list()

    def test_news_sentiment_agent_in_agent_registry(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        assert "equity_news_sentiment_v1" in AgentRegistry.list()

    def test_hybrid_strategy_definition_fields(self) -> None:
        from spectraquant_v3.strategies.registry import StrategyRegistry

        defn = StrategyRegistry.get("equity_momentum_news_hybrid_v1")
        assert defn.asset_class == "equity"
        assert "equity_momentum_news_hybrid_v1" in defn.agents
        assert "equity" in defn.tags
        assert "news" in defn.tags
        assert defn.enabled is True

    def test_hybrid_strategy_loadable_by_loader(self) -> None:
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load("equity_momentum_news_hybrid_v1")
        assert defn.strategy_id == "equity_momentum_news_hybrid_v1"

    def test_no_news_data_enum_value_exists(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason

        assert hasattr(NoSignalReason, "NO_NEWS_DATA")
        assert NoSignalReason.NO_NEWS_DATA.value == "no_news_data"


# ===========================================================================
# compose_equity_hybrid_score
# ===========================================================================


class TestComposeEquityHybridScore:
    """Unit tests for the pure blending function."""

    def _score(self, **kw):
        from spectraquant_v3.equities.signals.hybrid import compose_equity_hybrid_score

        return compose_equity_hybrid_score(**kw)

    def test_blend_math_with_news(self) -> None:
        # 0.7 * 0.5 + 0.3 * (-0.2) = 0.35 - 0.06 = 0.29
        score = self._score(
            momentum_score=0.5,
            news_score=-0.2,
            momentum_weight=0.7,
            news_weight=0.3,
        )
        assert score == pytest.approx(0.29, abs=1e-9)

    def test_fallback_to_pure_momentum_when_news_none(self) -> None:
        score = self._score(momentum_score=0.42, news_score=None)
        assert score == pytest.approx(0.42, abs=1e-9)

    def test_fallback_to_pure_momentum_when_news_nan(self) -> None:
        score = self._score(momentum_score=0.35, news_score=float("nan"))
        assert score == pytest.approx(0.35, abs=1e-9)

    def test_vol_gate_no_dampening_below_threshold(self) -> None:
        # vol_realised 0.18 < gate 0.25 → no dampening
        score = self._score(momentum_score=0.4, news_score=None, vol_realised=0.18)
        assert score == pytest.approx(0.4, abs=1e-9)

    def test_vol_gate_dampens_above_threshold(self) -> None:
        # vol_realised=0.375 (1.5 × gate), gate=0.25
        # excess = 0.125, dampening = max(0, 1 - 0.125/0.25) = 0.5
        # blended = 0.4 * 0.5 = 0.2
        score = self._score(
            momentum_score=0.4,
            news_score=None,
            vol_realised=0.375,
            vol_gate_threshold=0.25,
        )
        assert score == pytest.approx(0.2, abs=1e-9)

    def test_vol_gate_zeroes_at_double_threshold(self) -> None:
        # vol_realised = 2 × gate → dampening = 0
        score = self._score(
            momentum_score=0.8,
            news_score=None,
            vol_realised=0.5,
            vol_gate_threshold=0.25,
        )
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_score_clamped_to_minus_one_plus_one(self) -> None:
        score = self._score(momentum_score=1.0, news_score=1.0)
        assert -1.0 <= score <= 1.0

    def test_negative_news_reduces_positive_momentum(self) -> None:
        score = self._score(
            momentum_score=0.6,
            news_score=-0.8,
            momentum_weight=0.7,
            news_weight=0.3,
        )
        assert score < 0.6


# ===========================================================================
# EquityNewsSentimentAgent
# ===========================================================================


class TestEquityNewsSentimentAgent:
    """Tests for the ported V3 news-catalyst stand-alone agent."""

    def _agent(self, **kw):
        from spectraquant_v3.equities.signals.news_sentiment import (
            EquityNewsSentimentAgent,
        )

        return EquityNewsSentimentAgent(run_id="test_run", **kw)

    def _df_with_news(self, score: float, n: int = 30) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({"news_sentiment_score": [score] * n}, index=idx)

    def _df_without_news(self, n: int = 30) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({"close": [100.0] * n}, index=idx)

    # --- construction ---

    def test_default_construction(self) -> None:
        agent = self._agent()
        assert agent.run_id == "test_run"
        assert agent.min_confidence == pytest.approx(0.1)

    def test_from_config(self) -> None:
        from spectraquant_v3.equities.signals.news_sentiment import (
            EquityNewsSentimentAgent,
        )

        cfg = {"equities": {"signals": {"news_min_confidence": 0.2}}}
        agent = EquityNewsSentimentAgent.from_config(cfg, run_id="r1")
        assert agent.min_confidence == pytest.approx(0.2)

    # --- happy path ---

    def test_positive_news_returns_ok_signal(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        agent = self._agent()
        row = agent.evaluate("INFY.NS", self._df_with_news(0.5))
        assert row.status == SignalStatus.OK.value
        assert row.signal_score == pytest.approx(0.5, abs=1e-9)
        assert row.confidence == pytest.approx(0.5, abs=1e-9)

    def test_negative_news_returns_ok_signal(self) -> None:
        agent = self._agent()
        row = agent.evaluate("TCS.NS", self._df_with_news(-0.6))
        assert row.signal_score == pytest.approx(-0.6, abs=1e-9)

    def test_asset_class_is_equity(self) -> None:
        agent = self._agent()
        row = agent.evaluate("WIPRO.NS", self._df_with_news(0.4))
        assert row.asset_class == "equity"

    def test_agent_id_is_correct(self) -> None:
        agent = self._agent()
        row = agent.evaluate("WIPRO.NS", self._df_with_news(0.4))
        assert row.agent_id == "equity_news_sentiment_v1"

    def test_score_clamped_to_minus_one_plus_one(self) -> None:
        agent = self._agent()
        row = agent.evaluate("X", self._df_with_news(2.5))
        assert -1.0 <= row.signal_score <= 1.0

    # --- no-news / degradation paths ---

    def test_no_news_column_returns_no_signal(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason, SignalStatus

        agent = self._agent()
        row = agent.evaluate("RELIANCE.NS", self._df_without_news())
        assert row.status == SignalStatus.NO_SIGNAL.value
        assert row.no_signal_reason == NoSignalReason.NO_NEWS_DATA.value
        assert row.signal_score == 0.0
        assert row.confidence == 0.0

    def test_nan_news_value_returns_no_signal(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason, SignalStatus

        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({"news_sentiment_score": [float("nan")] * 10}, index=idx)
        agent = self._agent()
        row = agent.evaluate("HDFC.NS", df)
        assert row.status == SignalStatus.NO_SIGNAL.value
        assert row.no_signal_reason == NoSignalReason.NO_NEWS_DATA.value

    def test_score_below_min_confidence_returns_no_signal(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason, SignalStatus

        agent = self._agent(min_confidence=0.5)
        row = agent.evaluate("SBI.NS", self._df_with_news(0.1))
        assert row.status == SignalStatus.NO_SIGNAL.value
        assert row.no_signal_reason == NoSignalReason.BELOW_THRESHOLD.value

    def test_empty_df_returns_error_or_no_signal(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        agent = self._agent()
        empty = pd.DataFrame()
        row = agent.evaluate("X", empty)
        assert row.status in (SignalStatus.NO_SIGNAL.value, SignalStatus.ERROR.value)

    def test_evaluate_many_processes_all_symbols(self) -> None:
        agent = self._agent()
        feature_map = {
            "INFY.NS": self._df_with_news(0.4),
            "TCS.NS": self._df_with_news(-0.3),
        }
        rows = agent.evaluate_many(feature_map)
        assert len(rows) == 2
        syms = {r.canonical_symbol for r in rows}
        assert syms == {"INFY.NS", "TCS.NS"}


# ===========================================================================
# EquityMomentumNewsHybridAgent
# ===========================================================================


class TestEquityMomentumNewsHybridAgent:
    """Tests for the hybrid equity signal agent."""

    def _agent(self, **kw):
        from spectraquant_v3.equities.signals.hybrid import (
            EquityMomentumNewsHybridAgent,
        )

        return EquityMomentumNewsHybridAgent(run_id="test_run", **kw)

    # --- construction ---

    def test_default_construction(self) -> None:
        agent = self._agent()
        assert agent.run_id == "test_run"
        assert agent.momentum_weight == pytest.approx(0.7)
        assert agent.news_weight == pytest.approx(0.3)
        assert agent.vol_gate_threshold == pytest.approx(0.25)

    def test_from_config_reads_strategy_overrides(self) -> None:
        from spectraquant_v3.equities.signals.hybrid import (
            EquityMomentumNewsHybridAgent,
        )

        cfg = {
            "equities": {"signals": {"momentum_lookback": 20}},
            "portfolio": {},
            "strategies": {
                "equity_momentum_news_hybrid_v1": {
                    "signal_blend": {"momentum_weight": 0.6, "news_weight": 0.4},
                    "vol_gate": {"threshold": 0.30},
                }
            },
        }
        agent = EquityMomentumNewsHybridAgent.from_config(cfg, run_id="r1")
        assert agent.momentum_weight == pytest.approx(0.6)
        assert agent.news_weight == pytest.approx(0.4)
        assert agent.vol_gate_threshold == pytest.approx(0.30)

    # --- happy path ---

    def test_evaluate_returns_signal_row(self) -> None:
        from spectraquant_v3.core.schema import SignalRow

        agent = self._agent()
        row = agent.evaluate("INFY.NS", _feature_df())
        assert isinstance(row, SignalRow)

    def test_evaluate_asset_class_is_equity(self) -> None:
        agent = self._agent()
        row = agent.evaluate("TCS.NS", _feature_df())
        assert row.asset_class == "equity"

    def test_evaluate_agent_id_correct(self) -> None:
        agent = self._agent()
        row = agent.evaluate("WIPRO.NS", _feature_df())
        assert row.agent_id == "equity_momentum_news_hybrid_v1"

    def test_positive_momentum_and_news_gives_ok_signal(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        agent = self._agent()
        df = _feature_df(mom_val=0.08, news_score=0.4)
        row = agent.evaluate("RELIANCE.NS", df)
        assert row.status == SignalStatus.OK.value
        assert row.signal_score > 0.0

    def test_score_clamped_to_valid_range(self) -> None:
        agent = self._agent()
        row = agent.evaluate("X", _feature_df(mom_val=0.9, news_score=0.9))
        assert -1.0 <= row.signal_score <= 1.0
        assert 0.0 <= row.confidence <= 1.0

    # --- degradation paths ---

    def test_no_news_column_falls_back_to_momentum(self) -> None:
        """Hybrid must not ERROR when news data is absent – fall back to momentum."""
        from spectraquant_v3.core.enums import SignalStatus

        agent = self._agent()
        df_no_news = _feature_df(news_score=None)
        row = agent.evaluate("HDFC.NS", df_no_news)
        # Without news, it should still produce a signal based on momentum alone
        assert row.status in (SignalStatus.OK.value, SignalStatus.NO_SIGNAL.value)
        assert row.status != "ERROR"

    def test_nan_news_falls_back_to_momentum(self) -> None:
        """NaN news column should not trigger an ERROR."""
        from spectraquant_v3.core.enums import SignalStatus

        agent = self._agent()
        idx = pd.date_range("2024-01-01", periods=40, freq="D")
        df = pd.DataFrame(
            {
                "ret_20d": [0.05] * 40,
                "rsi": [55.0] * 40,
                "news_sentiment_score": [float("nan")] * 40,
            },
            index=idx,
        )
        row = agent.evaluate("WIPRO.NS", df)
        assert row.status != "ERROR"

    def test_high_volatility_dampens_signal(self) -> None:
        """A vol_realised above the gate threshold must reduce the score."""
        agent = self._agent(vol_gate_threshold=0.20)

        df_low_vol = _feature_df(mom_val=0.08, news_score=0.3, vol_realised=0.10)
        df_high_vol = _feature_df(mom_val=0.08, news_score=0.3, vol_realised=0.50)

        row_low = agent.evaluate("A", df_low_vol)
        row_high = agent.evaluate("A", df_high_vol)

        # High vol should produce a lower (or equal) absolute score
        assert abs(row_high.signal_score) <= abs(row_low.signal_score)

    def test_insufficient_rows_returns_no_signal(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        agent = self._agent(min_rows=20)
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        tiny = pd.DataFrame(
            {"ret_20d": [0.05] * 5, "rsi": [55.0] * 5},
            index=idx,
        )
        row = agent.evaluate("SBIN.NS", tiny)
        assert row.status == SignalStatus.NO_SIGNAL.value

    def test_bad_data_does_not_raise_exception(self) -> None:
        """Any input that causes a numeric error must return ERROR, never raise."""
        agent = self._agent()
        df_bad = pd.DataFrame({"ret_20d": ["garbage"] * 40, "rsi": ["x"] * 40})
        # Should not raise
        row = agent.evaluate("BAD", df_bad)
        assert row.status in ("ERROR", "NO_SIGNAL")

    def test_evaluate_many_returns_one_row_per_symbol(self) -> None:
        agent = self._agent()
        feature_map = {
            "INFY.NS": _feature_df(mom_val=0.06),
            "TCS.NS": _feature_df(mom_val=-0.04, news_score=-0.2),
        }
        rows = agent.evaluate_many(feature_map)
        assert len(rows) == 2
        syms = {r.canonical_symbol for r in rows}
        assert syms == {"INFY.NS", "TCS.NS"}


# ===========================================================================
# Pipeline integration: run_strategy dispatches to hybrid agent
# ===========================================================================


class TestHybridStrategyPipelineIntegration:
    """End-to-end test using run_strategy with dataset mode."""

    def _make_equity_cfg(self) -> dict:
        from spectraquant_v3.core.config import get_equity_config

        cfg = get_equity_config()
        cfg["equities"]["universe"]["require_exchange_coverage"] = False
        return cfg

    def test_run_strategy_reaches_hybrid_agent(self) -> None:
        from spectraquant_v3.pipeline import run_strategy

        cfg = self._make_equity_cfg()
        idx = pd.date_range("2025-01-01", periods=40, freq="D")
        ds = pd.DataFrame(
            {
                "ret_20d": [0.05] * 40,
                "rsi": [55.0] * 40,
                "news_sentiment_score": [0.3] * 40,
                "vol_realised": [0.18] * 40,
            },
            index=idx,
        )

        result = run_strategy(
            "equity_momentum_news_hybrid_v1",
            cfg=cfg,
            dry_run=True,
            dataset={"INFY.NS": ds},
        )

        assert result["status"] == "success"
        assert result["strategy_id"] == "equity_momentum_news_hybrid_v1"
        infy_rows = [s for s in result["signals"] if s.canonical_symbol == "INFY.NS"]
        assert infy_rows, "No signal row produced for INFY.NS"
        assert infy_rows[0].agent_id == "equity_momentum_news_hybrid_v1"

    def test_run_strategy_hybrid_positive_signal_when_data_good(self) -> None:
        from spectraquant_v3.pipeline import run_strategy

        cfg = self._make_equity_cfg()
        idx = pd.date_range("2025-01-01", periods=40, freq="D")
        ds = pd.DataFrame(
            {
                "ret_20d": [0.08] * 40,
                "rsi": [55.0] * 40,
                "news_sentiment_score": [0.5] * 40,
                "vol_realised": [0.12] * 40,
            },
            index=idx,
        )

        result = run_strategy(
            "equity_momentum_news_hybrid_v1",
            cfg=cfg,
            dry_run=True,
            dataset={"INFY.NS": ds},
        )

        infy_rows = [s for s in result["signals"] if s.canonical_symbol == "INFY.NS"]
        assert infy_rows[0].signal_score > 0.0

    def test_run_strategy_no_news_column_degrades_gracefully(self) -> None:
        """Strategy must complete successfully even with no news data in dataset."""
        from spectraquant_v3.pipeline import run_strategy

        cfg = self._make_equity_cfg()
        idx = pd.date_range("2025-01-01", periods=40, freq="D")
        # Dataset without news_sentiment_score
        ds = pd.DataFrame(
            {
                "ret_20d": [0.05] * 40,
                "rsi": [55.0] * 40,
                "vol_realised": [0.18] * 40,
            },
            index=idx,
        )

        result = run_strategy(
            "equity_momentum_news_hybrid_v1",
            cfg=cfg,
            dry_run=True,
            dataset={"INFY.NS": ds},
        )

        assert result["status"] == "success"
        # Signal may be OK or NO_SIGNAL but must not be ERROR
        infy_rows = [s for s in result["signals"] if s.canonical_symbol == "INFY.NS"]
        if infy_rows:
            assert infy_rows[0].status != "ERROR"

    def test_run_strategy_empty_dataset_produces_no_signal(self) -> None:
        """Completely empty feature maps must complete without raising."""
        from spectraquant_v3.pipeline import run_strategy

        cfg = self._make_equity_cfg()
        # dataset is None → pipeline evaluates against empty DataFrames
        result = run_strategy(
            "equity_momentum_news_hybrid_v1",
            cfg=cfg,
            dry_run=True,
            dataset={},
        )
        assert result["status"] == "success"


# ===========================================================================
# Baseline regression: equity_momentum_v1 still works
# ===========================================================================


class TestBaselineMomentumStillWorks:
    """Guard that porting the hybrid did not break the baseline momentum strategy."""

    def test_equity_momentum_v1_registered(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry
        from spectraquant_v3.strategies.registry import StrategyRegistry

        assert "equity_momentum_v1" in AgentRegistry.list()
        assert "equity_momentum_v1" in StrategyRegistry.list()

    def test_equity_momentum_v1_run_strategy_succeeds(self) -> None:
        from spectraquant_v3.core.config import get_equity_config
        from spectraquant_v3.pipeline import run_strategy

        cfg = get_equity_config()
        idx = pd.date_range("2025-01-01", periods=40, freq="D")
        ds = pd.DataFrame(
            {"ret_20d": [0.04] * 40, "rsi": [52.0] * 40},
            index=idx,
        )

        result = run_strategy(
            "equity_momentum_v1",
            cfg=cfg,
            dry_run=True,
            dataset={"INFY.NS": ds},
        )

        assert result["status"] == "success"
        infy = [s for s in result["signals"] if s.canonical_symbol == "INFY.NS"]
        assert infy
        assert infy[0].agent_id == "equity_momentum_v1"

    def test_momentum_agent_score_sign_correct(self) -> None:
        """Strong positive momentum must give a positive score."""
        from spectraquant_v3.equities.signals.momentum import EquityMomentumAgent

        agent = EquityMomentumAgent(run_id="r1")
        idx = pd.date_range("2024-01-01", periods=40, freq="D")
        df = pd.DataFrame(
            {"ret_20d": [0.15] * 40, "rsi": [55.0] * 40}, index=idx
        )
        row = agent.evaluate("INFY.NS", df)
        assert row.signal_score > 0.0


# ===========================================================================
# Asset-class segregation guard
# ===========================================================================


class TestAssetClassSegregation:
    """Ensure equity hybrid modules do not import from crypto."""

    def test_hybrid_module_has_no_crypto_import(self) -> None:
        import ast
        import importlib
        import importlib.util
        import pathlib

        spec = importlib.util.find_spec(
            "spectraquant_v3.equities.signals.hybrid"
        )
        assert spec is not None
        loader = spec.loader
        assert loader is not None

        src = pathlib.Path(spec.origin).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "crypto" not in node.module, (
                        f"equity hybrid module imports from crypto: {node.module}"
                    )

    def test_news_sentiment_module_has_no_crypto_import(self) -> None:
        import ast
        import importlib.util
        import pathlib

        spec = importlib.util.find_spec(
            "spectraquant_v3.equities.signals.news_sentiment"
        )
        assert spec is not None
        src = pathlib.Path(spec.origin).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "crypto" not in node.module
