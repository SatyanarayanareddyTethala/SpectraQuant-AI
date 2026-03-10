"""Unit tests for the intelligence layer modules.

Covers:
  - failure_memory
  - analog_memory (AnalogMarketMemory)
  - regime_engine
  - meta_learner
  - trade_planner
  - execution_intelligence
  - capital_intelligence
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_storage(tmp_path: Path) -> str:
    """Return a temporary directory string for storage."""
    return str(tmp_path / "intelligence_state")


# ===========================================================================
# A) failure_memory
# ===========================================================================

class TestFailureMemory:
    """Tests for src/spectraquant/intelligence/failure_memory.py"""

    def test_record_trade_intent_returns_id(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.failure_memory import record_trade_intent

        trade_id = record_trade_intent(
            run_id="run_001",
            ticker="RELIANCE.NS",
            side="long",
            confidence=0.72,
            features={"rsi": 55.0},
            news_context={"headline": "Q3 results beat"},
            regime="TRENDING",
            storage_dir=tmp_storage,
        )
        assert isinstance(trade_id, str)
        assert len(trade_id) > 0

    def test_record_trade_outcome_updates_record(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.failure_memory import (
            record_trade_intent,
            record_trade_outcome,
            update_failure_stats,
        )

        tid = record_trade_intent(
            run_id="run_002",
            ticker="TCS.NS",
            side="long",
            confidence=0.80,
            features={},
            news_context={},
            regime="RISK_ON",
            storage_dir=tmp_storage,
        )
        record_trade_outcome(
            trade_id=tid,
            realized_return=-0.03,
            mae=-0.04,
            mfe=0.01,
            slippage_bps=5.0,
            exit_reason="stop_hit",
            storage_dir=tmp_storage,
        )
        stats = update_failure_stats(storage_dir=tmp_storage)
        assert stats["total_trades"] >= 1
        assert stats["total_failures"] >= 1

    def test_label_failure_none_for_winner(self) -> None:
        from spectraquant.intelligence.failure_memory import TradeRecord, label_failure

        trade = TradeRecord(
            ticker="INFY.NS",
            realized_return=0.05,
            confidence=0.7,
            mae=-0.01,
            mfe=0.06,
            slippage_bps=3.0,
        )
        assert label_failure(trade) is None

    def test_label_failure_model_error(self) -> None:
        from spectraquant.intelligence.failure_memory import TradeRecord, label_failure

        trade = TradeRecord(
            ticker="WIPRO.NS",
            realized_return=-0.04,
            confidence=0.30,   # low confidence, not overconfident
            mae=-0.04,
            mfe=0.005,
            slippage_bps=4.0,  # under threshold (20 bps)
            news_context={},   # no news shock
            regime="TRENDING",
        )
        result = label_failure(trade)
        assert result == "MODEL_ERROR"

    def test_label_failure_overconfidence(self) -> None:
        from spectraquant.intelligence.failure_memory import TradeRecord, label_failure

        trade = TradeRecord(
            ticker="HDFC.NS",
            realized_return=-0.02,
            confidence=0.95,   # very high confidence, bad outcome
            mae=-0.02,
            mfe=0.005,
            slippage_bps=3.0,
            news_context={},
        )
        result = label_failure(trade)
        assert result == "OVERCONFIDENCE"

    def test_label_failure_news_shock(self) -> None:
        from spectraquant.intelligence.failure_memory import TradeRecord, label_failure

        trade = TradeRecord(
            ticker="ONGC.NS",
            realized_return=-0.05,
            confidence=0.60,
            mae=-0.05,
            mfe=0.01,
            slippage_bps=2.0,
            news_context={"risk_score": 0.9},
        )
        result = label_failure(trade)
        assert result == "NEWS_SHOCK"

    def test_label_failure_liquidity_slippage(self) -> None:
        from spectraquant.intelligence.failure_memory import TradeRecord, label_failure

        trade = TradeRecord(
            ticker="SMALLCAP.NS",
            realized_return=-0.02,
            confidence=0.50,
            mae=-0.02,
            mfe=0.01,
            slippage_bps=30.0,   # high slippage
            news_context={},
        )
        result = label_failure(trade)
        assert result == "LIQUIDITY_SLIPPAGE"

    def test_failure_types_constant(self) -> None:
        from spectraquant.intelligence.failure_memory import FAILURE_TYPES

        assert "MODEL_ERROR" in FAILURE_TYPES
        assert "NEWS_SHOCK" in FAILURE_TYPES
        assert len(FAILURE_TYPES) == 7

    def test_export_failure_report(self, tmp_storage: str, tmp_path: Path) -> None:
        from spectraquant.intelligence.failure_memory import (
            export_failure_report,
            record_trade_intent,
            record_trade_outcome,
        )

        tid = record_trade_intent(
            run_id="run_003",
            ticker="SAIL.NS",
            side="short",
            confidence=0.60,
            features={},
            news_context={},
            regime="CHOPPY",
            storage_dir=tmp_storage,
        )
        record_trade_outcome(
            trade_id=tid,
            realized_return=-0.02,
            mae=-0.03,
            mfe=0.005,
            slippage_bps=5.0,
            exit_reason="stop_hit",
            storage_dir=tmp_storage,
        )
        report_path = str(tmp_path / "failure_report.json")
        export_failure_report(report_path, storage_dir=tmp_storage)

        assert Path(report_path).exists()
        with open(report_path) as fh:
            data = json.load(fh)
        assert "total_trades" in data
        assert "generated_at" in data


# ===========================================================================
# B) analog_memory (AnalogMarketMemory)
# ===========================================================================

class TestAnalogMarketMemory:
    """Tests for src/spectraquant/intelligence/analog_memory.py"""

    def test_encode_state_shape(self) -> None:
        from spectraquant.intelligence.analog_memory import encode_state, _FEATURE_KEYS

        state = {"rsi_14": 55.0, "atr_pct": 0.02, "regime": "TRENDING"}
        emb = encode_state(state)
        assert emb.shape == (len(_FEATURE_KEYS),)
        assert emb.dtype == np.float32

    def test_encode_state_missing_keys_zero(self) -> None:
        from spectraquant.intelligence.analog_memory import encode_state

        emb = encode_state({})
        assert np.all(np.isfinite(emb))

    def test_encode_state_regime_encoding(self) -> None:
        from spectraquant.intelligence.analog_memory import encode_state, _REGIME_ENCODING

        for regime, expected in _REGIME_ENCODING.items():
            state = {"regime": regime}
            emb = encode_state(state)
            # regime_numeric is at position of "regime_numeric" in _FEATURE_KEYS
            from spectraquant.intelligence.analog_memory import _FEATURE_KEYS
            idx = list(_FEATURE_KEYS).index("regime_numeric")
            assert emb[idx] == pytest.approx(expected)

    def test_add_and_len(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.analog_memory import AnalogMarketMemory

        mem = AnalogMarketMemory(storage_dir=tmp_storage)
        assert len(mem) == 0
        mem.add_state({"rsi_14": 50.0}, outcome={"return_5d": 0.02})
        assert len(mem) == 1
        mem.add_state({"rsi_14": 60.0, "regime": "RISK_ON"}, outcome={"return_5d": -0.01})
        assert len(mem) == 2

    def test_query_similar_returns_sorted(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.analog_memory import AnalogMarketMemory

        mem = AnalogMarketMemory(storage_dir=tmp_storage)
        # Add several states
        for i in range(5):
            mem.add_state(
                {"rsi_14": float(40 + i * 5), "regime": "CHOPPY"},
                outcome={"return_5d": float(i) * 0.01},
            )
        query = {"rsi_14": 55.0, "regime": "CHOPPY"}
        neighbors = mem.query_similar(query, k=3)
        assert len(neighbors) <= 3
        # Should be sorted by distance ascending
        distances = [n.distance for n in neighbors]
        assert distances == sorted(distances)

    def test_query_similar_empty(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.analog_memory import AnalogMarketMemory

        mem = AnalogMarketMemory(storage_dir=tmp_storage)
        result = mem.query_similar({"rsi_14": 50.0}, k=5)
        assert result == []

    def test_calibrate_confidence_no_neighbors(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.analog_memory import AnalogMarketMemory

        mem = AnalogMarketMemory(storage_dir=tmp_storage)
        result = mem.calibrate_confidence(0.65, neighbors=[], min_neighbors=3)
        assert result == pytest.approx(0.65)

    def test_calibrate_confidence_adjusts(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.analog_memory import AnalogMarketMemory

        mem = AnalogMarketMemory(storage_dir=tmp_storage)
        for i in range(10):
            mem.add_state(
                {"rsi_14": float(50 + i)},
                outcome={"return_5d": 0.03},   # all positive
            )
        neighbors = mem.query_similar({"rsi_14": 55.0}, k=10)
        adj = mem.calibrate_confidence(0.50, neighbors, blend_weight=0.5, min_neighbors=3)
        # All positive outcomes → analog ratio = 1.0 → should push above 0.50
        assert adj > 0.50

    def test_calibrate_confidence_clamped(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.analog_memory import AnalogMarketMemory

        mem = AnalogMarketMemory(storage_dir=tmp_storage)
        for _ in range(5):
            mem.add_state({"rsi_14": 80.0}, outcome={"return_5d": 0.10})
        neighbors = mem.query_similar({"rsi_14": 80.0}, k=5)
        adj = mem.calibrate_confidence(1.0, neighbors, blend_weight=1.0, min_neighbors=1)
        assert 0.0 <= adj <= 1.0

    def test_deterministic_encoding(self) -> None:
        from spectraquant.intelligence.analog_memory import encode_state

        state = {"rsi_14": 45.0, "vol_20": 0.03, "regime": "RISK_OFF"}
        emb1 = encode_state(state)
        emb2 = encode_state(state)
        np.testing.assert_array_equal(emb1, emb2)


# ===========================================================================
# C) regime_engine
# ===========================================================================

class TestRegimeEngine:
    """Tests for src/spectraquant/intelligence/regime_engine.py"""

    def _make_df(self, n: int = 60, slope: float = 1.0) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = [100.0 + i * slope for i in range(n)]
        return pd.DataFrame({"close": prices}, index=dates)

    def test_compute_regime_features_trending(self) -> None:
        from spectraquant.intelligence.regime_engine import compute_regime_features

        df = self._make_df(slope=1.5)
        feats = compute_regime_features(df)
        assert "daily_vol" in feats
        assert "slope_pct" in feats
        assert "breadth" in feats
        assert feats["trend_strength"] > 0.5

    def test_compute_regime_features_empty_raises(self) -> None:
        from spectraquant.intelligence.regime_engine import compute_regime_features

        with pytest.raises(ValueError):
            compute_regime_features(pd.DataFrame())

    def test_classify_regime_trending(self) -> None:
        from spectraquant.intelligence.regime_engine import classify_regime

        features = {
            "daily_vol": 0.008,
            "vol_ratio": 0.8,
            "slope_pct": 0.003,
            "close_vs_sma": 0.02,
            "breadth": 0.65,
            "vix_proxy": 0.8,
            "trend_strength": 0.70,
        }
        label, conf = classify_regime(features)
        assert label in ("TRENDING", "RISK_ON", "CHOPPY")
        assert 0.0 <= conf <= 1.0

    def test_classify_regime_panic(self) -> None:
        from spectraquant.intelligence.regime_engine import classify_regime

        features = {
            "daily_vol": 0.06,    # > 4 % threshold
            "vol_ratio": 3.0,
            "slope_pct": -0.005,
            "close_vs_sma": -0.05,
            "breadth": 0.15,      # < 0.3 → panic
            "vix_proxy": 6.0,
            "trend_strength": 0.20,
        }
        label, conf = classify_regime(features)
        assert label == "PANIC"

    def test_classify_regime_choppy(self) -> None:
        from spectraquant.intelligence.regime_engine import classify_regime

        features = {
            "daily_vol": 0.008,
            "vol_ratio": 0.9,
            "slope_pct": 0.0,      # flat slope → choppy
            "close_vs_sma": 0.0,
            "breadth": 0.52,
            "vix_proxy": 0.8,
            "trend_strength": 0.50,
        }
        label, conf = classify_regime(features)
        assert label == "CHOPPY"
        assert 0.0 < conf <= 1.0

    def test_get_current_regime_returns_dict(self) -> None:
        from spectraquant.intelligence.regime_engine import get_current_regime, REGIME_LABELS

        result = get_current_regime()
        assert "label" in result
        assert "confidence" in result
        assert "features" in result
        assert "as_of" in result
        assert result["label"] in REGIME_LABELS

    def test_get_current_regime_with_df(self) -> None:
        from spectraquant.intelligence.regime_engine import get_current_regime, REGIME_LABELS

        df = self._make_df(slope=2.0)
        result = get_current_regime(df=df)
        assert result["label"] in REGIME_LABELS
        assert 0.0 <= result["confidence"] <= 1.0

    def test_regime_labels_stable(self) -> None:
        from spectraquant.intelligence.regime_engine import REGIME_LABELS

        assert "TRENDING" in REGIME_LABELS
        assert "PANIC" in REGIME_LABELS
        assert len(REGIME_LABELS) == 6


# ===========================================================================
# D) meta_learner
# ===========================================================================

class TestMetaLearner:
    """Tests for src/spectraquant/intelligence/meta_learner.py"""

    def _base_policy(self) -> Dict[str, Any]:
        return {
            "version": "abc12345",
            "thresholds": {"buy": 0.55, "sell": 0.45, "alpha": 0.0},
            "expert_weights": {},
            "max_candidates": 50,
            "cooldown_minutes": 60,
        }

    def test_validate_policy_valid(self) -> None:
        from spectraquant.intelligence.meta_learner import validate_policy_update

        ok, errors = validate_policy_update(self._base_policy())
        assert ok is True
        assert errors == []

    def test_validate_policy_buy_too_high(self) -> None:
        from spectraquant.intelligence.meta_learner import validate_policy_update

        policy = self._base_policy()
        policy["thresholds"]["buy"] = 0.99  # > 0.80 bound
        ok, errors = validate_policy_update(policy)
        assert ok is False
        assert any("buy" in e for e in errors)

    def test_validate_policy_buy_le_sell(self) -> None:
        from spectraquant.intelligence.meta_learner import validate_policy_update

        policy = self._base_policy()
        policy["thresholds"]["buy"] = 0.40
        policy["thresholds"]["sell"] = 0.45  # buy <= sell
        ok, errors = validate_policy_update(policy)
        assert ok is False

    def test_propose_policy_insufficient_trades(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.meta_learner import propose_policy_update

        metrics = {"total_trades": 2, "total_failures": 1}
        _, reasons = propose_policy_update(metrics, state_dir=tmp_storage)
        assert any("Insufficient" in r for r in reasons)

    def test_propose_policy_raises_buy_on_high_failure(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.meta_learner import propose_policy_update

        metrics = {
            "total_trades": 100,
            "total_failures": 60,   # 60 % failure rate
            "by_regime": {},
        }
        policy = self._base_policy()
        new_policy, reasons = propose_policy_update(metrics, current_policy=policy, state_dir=tmp_storage)
        assert new_policy["thresholds"]["buy"] > policy["thresholds"]["buy"]

    def test_propose_policy_lowers_buy_on_low_failure(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.meta_learner import propose_policy_update

        metrics = {
            "total_trades": 100,
            "total_failures": 15,   # 15 % failure
            "recent_win_rate": 0.70,
            "by_regime": {},
        }
        policy = self._base_policy()
        new_policy, reasons = propose_policy_update(metrics, current_policy=policy, state_dir=tmp_storage)
        assert new_policy["thresholds"]["buy"] <= policy["thresholds"]["buy"]

    def test_apply_policy_update_modifies_config(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.meta_learner import apply_policy_update

        config = {
            "intraday": {"signal_thresholds": {"buy": 0.55, "sell": 0.45}},
            "portfolio": {"alpha_threshold": 0.0},
            "news_universe": {"max_candidates": 50},
        }
        policy = {
            **self._base_policy(),
            "thresholds": {"buy": 0.60, "sell": 0.40, "alpha": 0.01},
            "max_candidates": 40,
        }
        updated = apply_policy_update(config, policy, state_dir=tmp_storage)
        assert updated["intraday"]["signal_thresholds"]["buy"] == pytest.approx(0.60)
        assert updated["news_universe"]["max_candidates"] == 40

    def test_rollback_policy_nonexistent(self, tmp_storage: str) -> None:
        from spectraquant.intelligence.meta_learner import rollback_policy

        result = rollback_policy("nonexistent_version", state_dir=tmp_storage)
        assert result is None


# ===========================================================================
# E) trade_planner
# ===========================================================================

class TestTradePlanner:
    """Tests for src/spectraquant/intelligence/trade_planner.py"""

    def _make_candidates(self, n: int = 5) -> list:
        return [
            {
                "ticker": f"STOCK{i}.NS",
                "score": 0.55 + i * 0.05,
                "confidence": 0.60 + i * 0.03,
                "side": "long",
                "entry": 100.0 + i * 10,
                "stop": 95.0 + i * 10,
                "target": 110.0 + i * 10,
                "sector": "IT" if i % 2 == 0 else "BANKING",
            }
            for i in range(n)
        ]

    def test_plan_returns_dict(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.trade_planner import generate_premarket_plan

        plan = generate_premarket_plan(
            candidates=self._make_candidates(),
            config={"output_dir": str(tmp_path / "plans")},
        )
        assert "trades" in plan
        assert "status" in plan
        assert plan["status"] == "generated"

    def test_plan_filters_below_threshold(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.trade_planner import generate_premarket_plan

        # All candidates have score below 0.70
        cands = self._make_candidates()
        plan = generate_premarket_plan(
            candidates=cands,
            config={
                "thresholds": {"buy": 0.90},   # very high threshold
                "output_dir": str(tmp_path / "plans"),
            },
        )
        assert len(plan["trades"]) == 0

    def test_plan_respects_max_positions(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.trade_planner import generate_premarket_plan

        cands = self._make_candidates(10)
        plan = generate_premarket_plan(
            candidates=cands,
            config={
                "max_positions": 3,
                "output_dir": str(tmp_path / "plans"),
            },
        )
        assert len(plan["trades"]) <= 3

    def test_plan_trades_sorted_by_rank(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.trade_planner import generate_premarket_plan

        plan = generate_premarket_plan(
            candidates=self._make_candidates(),
            config={"output_dir": str(tmp_path / "plans")},
        )
        ranks = [t["rank"] for t in plan["trades"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_plan_writes_json_file(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.trade_planner import generate_premarket_plan

        plan = generate_premarket_plan(
            candidates=self._make_candidates(),
            config={"output_dir": str(tmp_path / "plans")},
        )
        assert "output_path" in plan
        assert Path(plan["output_path"]).exists()

    def test_plan_suppresses_longs_in_panic(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.trade_planner import generate_premarket_plan

        plan = generate_premarket_plan(
            candidates=self._make_candidates(),
            regime_dict={"label": "PANIC", "confidence": 0.9},
            config={
                "thresholds": {"buy": 0.55},
                "output_dir": str(tmp_path / "plans"),
            },
        )
        # Candidates start at score 0.55, PANIC multiplies by 0.5 → 0.275 < threshold
        assert len(plan["trades"]) == 0

    def test_plan_sector_cap(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.trade_planner import generate_premarket_plan

        # All candidates in the same sector
        cands = [
            {
                "ticker": f"BANK{i}.NS",
                "score": 0.80,
                "confidence": 0.75,
                "side": "long",
                "entry": 200.0,
                "stop": 190.0,
                "target": 220.0,
                "sector": "BANKING",
            }
            for i in range(10)
        ]
        plan = generate_premarket_plan(
            candidates=cands,
            config={
                "max_positions": 10,
                "max_sector_exposure": 0.30,
                "output_dir": str(tmp_path / "plans"),
            },
        )
        # At most 30 % of max_positions (= 3) from BANKING
        assert len(plan["trades"]) <= 3

    def test_plan_position_sizing(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.trade_planner import generate_premarket_plan

        cands = [
            {
                "ticker": "RELIANCE.NS",
                "score": 0.70,
                "confidence": 0.70,
                "side": "long",
                "entry": 2000.0,
                "stop": 1900.0,   # risk = 100 per share
                "target": 2200.0,
                "sector": "ENERGY",
            }
        ]
        plan = generate_premarket_plan(
            candidates=cands,
            config={
                "equity_base": 1_000_000.0,
                "risk_per_trade_pct": 0.01,   # risk 10,000 INR
                "output_dir": str(tmp_path / "plans"),
            },
        )
        assert len(plan["trades"]) == 1
        # shares = 10,000 / 100 = 100
        assert plan["trades"][0]["shares"] == 100


# ===========================================================================
# F) execution_intelligence
# ===========================================================================

class TestExecutionIntelligence:
    """Tests for src/spectraquant/intelligence/execution_intelligence.py"""

    def _make_trade(self, ticker: str = "TCS.NS", entry: float = 3500.0) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "side": "long",
            "entry": entry,
            "stop": entry * 0.97,
            "target": entry * 1.05,
            "confidence": 0.72,
            "news_context": {},
        }

    def test_evaluate_trigger_execute(self) -> None:
        from spectraquant.intelligence.execution_intelligence import (
            evaluate_trigger, TradeState,
        )

        trade = self._make_trade(entry=3500.0)
        snapshot = {"TCS.NS": {"price": 3500.0, "spread_bps": 5.0}}
        result = evaluate_trigger(trade, snapshot)
        assert result["state"] == TradeState.EXECUTE

    def test_evaluate_trigger_wait(self) -> None:
        from spectraquant.intelligence.execution_intelligence import (
            evaluate_trigger, TradeState,
        )

        # Price is between stop and entry (not yet triggered, not blocked)
        trade = self._make_trade(entry=3500.0)
        trade["stop"] = 3000.0   # stop well below current price
        snapshot = {"TCS.NS": {"price": 3200.0}}   # below entry but above stop
        result = evaluate_trigger(trade, snapshot)
        assert result["state"] == TradeState.WAIT

    def test_evaluate_trigger_watch(self) -> None:
        from spectraquant.intelligence.execution_intelligence import (
            evaluate_trigger, TradeState,
        )

        trade = self._make_trade(entry=3500.0)
        snapshot = {"TCS.NS": {"price": 3496.0}}   # within 0.5 %
        result = evaluate_trigger(trade, snapshot)
        assert result["state"] == TradeState.WATCH

    def test_evaluate_trigger_blocked_spread(self) -> None:
        from spectraquant.intelligence.execution_intelligence import (
            evaluate_trigger, TradeState,
        )

        trade = self._make_trade(entry=3500.0)
        snapshot = {"TCS.NS": {"price": 3500.0, "spread_bps": 200.0}}
        result = evaluate_trigger(trade, snapshot, policy_config={"max_spread_bps": 50.0})
        assert result["state"] == TradeState.BLOCKED

    def test_evaluate_trigger_no_market_data(self) -> None:
        from spectraquant.intelligence.execution_intelligence import (
            evaluate_trigger, TradeState,
        )

        trade = self._make_trade()
        result = evaluate_trigger(trade, {})   # empty snapshot
        assert result["state"] == TradeState.WAIT

    def test_cooldown_manager(self) -> None:
        from spectraquant.intelligence.execution_intelligence import CooldownManager

        mgr = CooldownManager(cooldown_minutes=0.001)  # exactly 0.06 seconds
        assert not mgr.is_cooling("TCS.NS")
        mgr.record_execution("TCS.NS")
        assert mgr.is_cooling("TCS.NS")

    def test_monitor_plan(self) -> None:
        from spectraquant.intelligence.execution_intelligence import (
            monitor_plan, TradeState,
        )

        plan = {
            "trades": [
                {"ticker": "TCS.NS", "side": "long", "entry": 3500.0, "stop": 3395.0, "confidence": 0.72, "news_context": {}},
                {"ticker": "INFY.NS", "side": "long", "entry": 1500.0, "stop": 1455.0, "confidence": 0.68, "news_context": {}},
            ]
        }
        snapshot = {
            "TCS.NS": {"price": 3500.0, "spread_bps": 3.0},
            "INFY.NS": {"price": 1400.0, "spread_bps": 3.0},   # below entry
        }
        result = monitor_plan(plan, snapshot)
        assert "ready" in result
        assert "waiting" in result
        # TCS should be ready, INFY should be waiting or watching
        ready_tickers = [t["ticker"] for t in result["ready"]]
        assert "TCS.NS" in ready_tickers

    def test_save_execution_snapshot(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.execution_intelligence import save_execution_snapshot

        snapshot = {"as_of": "2024-01-15T09:00:00+00:00", "ready": []}
        path = save_execution_snapshot(snapshot, output_dir=str(tmp_path / "intelligence"))
        assert Path(path).exists()


# ===========================================================================
# G) capital_intelligence
# ===========================================================================

class TestCapitalIntelligence:
    """Tests for src/spectraquant/intelligence/capital_intelligence.py"""

    def _positions(self) -> list:
        return [
            {"ticker": "TCS.NS", "sector": "IT", "notional": 100_000.0, "side": "long"},
            {"ticker": "INFY.NS", "sector": "IT", "notional": 80_000.0, "side": "long"},
            {"ticker": "HDFC.NS", "sector": "BANKING", "notional": 120_000.0, "side": "long"},
        ]

    def test_compute_exposures_basic(self) -> None:
        from spectraquant.intelligence.capital_intelligence import compute_exposures

        positions = self._positions()
        exp = compute_exposures(positions, equity=1_000_000.0)
        assert exp["gross_exposure"] == pytest.approx(0.30)
        assert exp["position_count"] == 3
        assert "by_sector" in exp
        assert "by_name" in exp
        assert exp["by_sector"]["IT"] == pytest.approx(0.18)

    def test_compute_exposures_empty(self) -> None:
        from spectraquant.intelligence.capital_intelligence import compute_exposures

        exp = compute_exposures([], equity=1_000_000.0)
        assert exp["gross_exposure"] == pytest.approx(0.0)
        assert exp["position_count"] == 0

    def test_check_trade_allowed_basic(self) -> None:
        from spectraquant.intelligence.capital_intelligence import (
            check_trade_allowed, compute_exposures,
        )

        exp = compute_exposures(self._positions(), equity=1_000_000.0)
        trade = {
            "ticker": "WIPRO.NS",
            "sector": "IT",
            "notional": 50_000.0,
            "confidence": 0.72,
        }
        allowed, reasons = check_trade_allowed(
            trade,
            exp,
            daily_pnl=0.0,
            limits={"equity_base": 1_000_000.0, "max_positions": 10, "max_sector_exposure": 0.30},
        )
        # IT sector would go from 0.18 to 0.23, still under 0.30
        assert allowed is True

    def test_check_trade_daily_loss(self) -> None:
        from spectraquant.intelligence.capital_intelligence import (
            check_trade_allowed, compute_exposures,
        )

        exp = compute_exposures([], equity=1_000_000.0)
        trade = {"ticker": "TCS.NS", "sector": "IT", "notional": 50_000.0, "confidence": 0.72}
        allowed, reasons = check_trade_allowed(
            trade,
            exp,
            daily_pnl=-6000.0,   # breached 5000 limit
            limits={"equity_base": 1_000_000.0, "daily_loss_limit": 5000.0},
        )
        assert allowed is False
        assert any("daily_loss" in r for r in reasons)

    def test_check_trade_max_positions(self) -> None:
        from spectraquant.intelligence.capital_intelligence import (
            check_trade_allowed, compute_exposures,
        )

        # Fill positions to max
        positions = [
            {"ticker": f"X{i}.NS", "sector": "IT", "notional": 10_000.0, "side": "long"}
            for i in range(5)
        ]
        exp = compute_exposures(positions, equity=1_000_000.0)
        trade = {"ticker": "NEW.NS", "sector": "FMCG", "notional": 10_000.0, "confidence": 0.72}
        allowed, reasons = check_trade_allowed(
            trade,
            exp,
            daily_pnl=0.0,
            limits={"equity_base": 1_000_000.0, "max_positions": 5},
        )
        assert allowed is False
        assert any("max_positions" in r for r in reasons)

    def test_check_trade_low_confidence(self) -> None:
        from spectraquant.intelligence.capital_intelligence import (
            check_trade_allowed, compute_exposures,
        )

        exp = compute_exposures([], equity=1_000_000.0)
        trade = {"ticker": "TCS.NS", "sector": "IT", "notional": 10_000.0, "confidence": 0.30}
        allowed, reasons = check_trade_allowed(
            trade, exp, daily_pnl=0.0,
            limits={"equity_base": 1_000_000.0, "min_confidence": 0.50},
        )
        assert allowed is False
        assert any("confidence" in r for r in reasons)

    def test_generate_risk_report(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.capital_intelligence import generate_risk_report

        # Use smaller notionals so name exposure stays below 10 % of 1M equity
        positions = [
            {"ticker": "TCS.NS", "sector": "IT", "notional": 80_000.0, "side": "long"},
            {"ticker": "INFY.NS", "sector": "IT", "notional": 70_000.0, "side": "long"},
            {"ticker": "HDFC.NS", "sector": "BANKING", "notional": 90_000.0, "side": "long"},
        ]
        report = generate_risk_report(
            positions=positions,
            daily_pnl=1000.0,
            equity=1_000_000.0,
            output_dir=str(tmp_path / "intelligence"),
        )
        assert report["status"] == "ok"
        assert "output_path" in report
        assert Path(report["output_path"]).exists()

    def test_generate_risk_report_with_breach(self, tmp_path: Path) -> None:
        from spectraquant.intelligence.capital_intelligence import generate_risk_report

        report = generate_risk_report(
            positions=self._positions(),
            daily_pnl=-6000.0,   # breached daily_loss_limit=5000
            equity=1_000_000.0,
            limits={"daily_loss_limit": 5000.0},
            output_dir=str(tmp_path / "intelligence"),
        )
        assert report["status"] == "warning"
        assert "daily_loss_limit" in report["breaches"]
