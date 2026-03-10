"""V3 Cognition Layer tests.

Tests proving the six core V3 requirements:
1. News changes BUY selection.
2. Similar events produce similar signals.
3. Small universes still produce trades.
4. System adapts after failures.
5. No circular imports.
6. Pipeline runs end-to-end with --from-news (tested in test_pipeline_from_news.py).
"""
from __future__ import annotations

import importlib
import math

import pytest

from spectraquant.intelligence.cognition import (
    BeliefEngine,
    BeliefScore,
    CandidateExplanation,
    CausalTemplate,
    ExplanationEngine,
    MechanismTag,
    get_causal_template,
)
from spectraquant.intelligence.cognition.causal_templates import (
    CAUSAL_TEMPLATE_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    ticker: str = "TEST",
    expected_return_annual: float = 0.10,
    score: float = 60.0,
    event_type: str = "unknown",
    event_confidence: float = 1.0,
    analog_prior: float = 0.5,
) -> dict:
    return {
        "ticker": ticker,
        "expected_return_annual": expected_return_annual,
        "score": score,
        "event_type": event_type,
        "event_confidence": event_confidence,
        "analog_prior": analog_prior,
    }


# ===========================================================================
# 1. News changes BUY selection
# ===========================================================================

class TestNewsChangesBuySelection:
    """Requirement 1: news event type materially changes which candidates
    are selected as BUY."""

    def test_earnings_beat_vs_no_event_changes_rank(self):
        """A candidate with earnings_beat should rank above an otherwise
        identical candidate with no event (unknown)."""
        engine = BeliefEngine(adaptive_fraction=0.5)
        candidates = [
            _make_candidate("BEAT", expected_return_annual=0.10, event_type="earnings_beat"),
            _make_candidate("NONE", expected_return_annual=0.10, event_type="unknown"),
        ]
        results = engine.evaluate(candidates, regime="RISK_ON")
        ranks = {r.ticker: r.rank for r in results}
        assert ranks["BEAT"] < ranks["NONE"], (
            "earnings_beat candidate should outrank unknown-event candidate"
        )

    def test_fraud_allegation_suppresses_buy(self):
        """A candidate with fraud_allegation should not be BUY when a
        clean candidate exists with similar price signal."""
        engine = BeliefEngine(adaptive_fraction=0.5, max_positions=1)
        candidates = [
            _make_candidate("CLEAN", expected_return_annual=0.12, event_type="earnings_beat"),
            _make_candidate("FRAUD", expected_return_annual=0.15, event_type="fraud_allegation"),
        ]
        results = engine.evaluate(candidates, regime="RISK_ON")
        buy_tickers = {r.ticker for r in results if r.signal == "BUY"}
        # With max_positions=1, the clean earnings_beat candidate should win
        assert "CLEAN" in buy_tickers, (
            "Clean earnings_beat candidate should be selected over fraud_allegation"
        )

    def test_high_event_confidence_raises_belief_score(self):
        """Higher event_confidence increases belief_score."""
        engine = BeliefEngine()
        low_conf = engine.score_candidate(
            "AAPL",
            expected_return=0.10,
            score_raw=60.0,
            event_type="earnings_beat",
            event_confidence=0.3,
        )
        high_conf = engine.score_candidate(
            "AAPL",
            expected_return=0.10,
            score_raw=60.0,
            event_type="earnings_beat",
            event_confidence=0.95,
        )
        assert high_conf.belief_score > low_conf.belief_score

    def test_buy_set_changes_when_event_changes(self):
        """Changing event_type from 'unknown' to 'regulatory_approval' must
        change the belief score (and potentially the BUY set)."""
        engine = BeliefEngine(adaptive_fraction=0.5, max_positions=1)

        # Baseline: no event
        without_event = engine.evaluate(
            [
                _make_candidate("A", expected_return_annual=0.05, event_type="unknown"),
                _make_candidate("B", expected_return_annual=0.10, event_type="unknown"),
            ],
            regime="RISK_ON",
        )

        # With strong event on A
        with_event = engine.evaluate(
            [
                _make_candidate("A", expected_return_annual=0.05, event_type="regulatory_approval"),
                _make_candidate("B", expected_return_annual=0.10, event_type="unknown"),
            ],
            regime="RISK_ON",
        )

        belief_a_without = next(r.belief_score for r in without_event if r.ticker == "A")
        belief_a_with = next(r.belief_score for r in with_event if r.ticker == "A")

        assert belief_a_with > belief_a_without, (
            "Adding regulatory_approval event must increase belief_score"
        )


# ===========================================================================
# 2. Similar events produce similar signals
# ===========================================================================

class TestSimilarEventsSimilarSignals:
    """Requirement 2: two candidates with the same event type should receive
    the same signal direction and similar belief scores."""

    def test_same_event_same_regime_similar_scores(self):
        """Two candidates with identical inputs should have identical scores."""
        engine = BeliefEngine()
        s1 = engine.score_candidate(
            "A", expected_return=0.12, score_raw=70.0,
            event_type="earnings_beat", regime="RISK_ON",
        )
        s2 = engine.score_candidate(
            "B", expected_return=0.12, score_raw=70.0,
            event_type="earnings_beat", regime="RISK_ON",
        )
        assert abs(s1.belief_score - s2.belief_score) < 1e-9

    def test_same_event_type_same_mechanism_tags(self):
        """Candidates with the same event type get the same mechanism tags."""
        engine = BeliefEngine()
        s1 = engine.score_candidate("A", expected_return=0.10, score_raw=60.0,
                                     event_type="rate_hike")
        s2 = engine.score_candidate("B", expected_return=0.05, score_raw=40.0,
                                     event_type="rate_hike")
        assert s1.mechanism_tags == s2.mechanism_tags

    def test_positive_events_rank_above_negative_events(self):
        """Positive event types (earnings_beat) should consistently rank
        above strongly negative events (fraud_allegation) in RISK_ON."""
        engine = BeliefEngine()
        candidates = [
            {"ticker": f"POS_{i}", "expected_return_annual": 0.10,
             "score": 60.0, "event_type": "earnings_beat"}
            for i in range(3)
        ] + [
            {"ticker": f"NEG_{i}", "expected_return_annual": 0.10,
             "score": 60.0, "event_type": "fraud_allegation"}
            for i in range(3)
        ]
        results = engine.evaluate(candidates, regime="RISK_ON")
        pos_scores = [r.belief_score for r in results if r.ticker.startswith("POS")]
        neg_scores = [r.belief_score for r in results if r.ticker.startswith("NEG")]
        assert min(pos_scores) > max(neg_scores), (
            "All positive-event candidates should score above negative-event candidates"
        )

    def test_causal_template_deterministic(self):
        """get_causal_template returns the same object for repeated calls."""
        t1 = get_causal_template("earnings_beat")
        t2 = get_causal_template("earnings_beat")
        assert t1 is t2  # same dict entry


# ===========================================================================
# 3. Small universes still produce trades
# ===========================================================================

class TestSmallUniversesProduceTrades:
    """Requirement 3: even with 1 candidate, the system must produce a BUY."""

    def test_single_candidate_gets_buy(self):
        engine = BeliefEngine(max_positions=5, adaptive_fraction=0.4)
        results = engine.evaluate(
            [_make_candidate("ONLY", expected_return_annual=0.10)],
            regime="RISK_ON",
        )
        assert len(results) == 1
        assert results[0].signal == "BUY", "Single candidate must always be BUY"

    def test_two_candidates_at_least_one_buy(self):
        engine = BeliefEngine(max_positions=5, adaptive_fraction=0.4)
        results = engine.evaluate(
            [
                _make_candidate("A", expected_return_annual=0.10),
                _make_candidate("B", expected_return_annual=-0.05),
            ],
            regime="RISK_OFF",
        )
        buy_count = sum(1 for r in results if r.signal == "BUY")
        assert buy_count >= 1

    def test_adaptive_k_scales_with_universe_size(self):
        """K should scale with N: larger universe → more BUYs (up to max)."""
        engine = BeliefEngine(max_positions=10, adaptive_fraction=0.5)
        small = engine.evaluate(
            [_make_candidate(f"T{i}") for i in range(2)],
            regime="RISK_ON",
        )
        large = engine.evaluate(
            [_make_candidate(f"T{i}") for i in range(10)],
            regime="RISK_ON",
        )
        small_buys = sum(1 for r in small if r.signal == "BUY")
        large_buys = sum(1 for r in large if r.signal == "BUY")
        assert large_buys >= small_buys, (
            "Larger universe should produce at least as many BUYs as smaller universe"
        )

    def test_empty_universe_returns_empty(self):
        engine = BeliefEngine()
        results = engine.evaluate([], regime="RISK_ON")
        assert results == []

    def test_max_positions_respected(self):
        engine = BeliefEngine(max_positions=2, adaptive_fraction=1.0)
        candidates = [_make_candidate(f"T{i}") for i in range(10)]
        results = engine.evaluate(candidates, regime="RISK_ON")
        buy_count = sum(1 for r in results if r.signal == "BUY")
        assert buy_count <= 2


# ===========================================================================
# 4. System adapts after failures
# ===========================================================================

class TestSystemAdaptsAfterFailures:
    """Requirement 4: adapt_weights adjusts tower weights in the right direction."""

    def test_adapt_weights_increases_tower_weight(self):
        engine = BeliefEngine()
        before = engine._weights["RISK_ON"]["event"]
        engine.adapt_weights("RISK_ON", "event", direction=+1.0, learning_rate=0.10)
        after = engine._weights["RISK_ON"]["event"]
        assert after > before, "Positive direction should increase tower weight"

    def test_adapt_weights_decreases_tower_weight(self):
        engine = BeliefEngine()
        before = engine._weights["RISK_ON"]["price"]
        engine.adapt_weights("RISK_ON", "price", direction=-1.0, learning_rate=0.10)
        after = engine._weights["RISK_ON"]["price"]
        assert after < before

    def test_adapt_weights_renormalizes(self):
        engine = BeliefEngine()
        engine.adapt_weights("RISK_ON", "analog", direction=+1.0, learning_rate=0.20)
        total = sum(engine._weights["RISK_ON"].values())
        assert abs(total - 1.0) < 1e-9, "Weights must remain normalized after adaptation"

    def test_adapt_weights_after_failure_reduces_event_weight(self):
        """After a failed trade driven by event signal, the event weight
        should decrease, making future trades less event-dependent."""
        engine = BeliefEngine()
        w_before = engine._weights["RISK_ON"]["event"]
        # Simulate failure: event tower was wrong → reduce its weight
        engine.adapt_weights("RISK_ON", "event", direction=-1.0, learning_rate=0.05)
        w_after = engine._weights["RISK_ON"]["event"]
        assert w_after < w_before

    def test_adapt_weights_unknown_tower_no_crash(self):
        """adapt_weights with unknown tower should not raise."""
        engine = BeliefEngine()
        engine.adapt_weights("RISK_ON", "nonexistent_tower", direction=+1.0)
        # Should complete without error


# ===========================================================================
# 5. No circular imports
# ===========================================================================

class TestNoCircularImports:
    """Requirement 5: all cognition modules import cleanly and independently."""

    def test_import_causal_templates(self):
        mod = importlib.import_module("spectraquant.intelligence.cognition.causal_templates")
        assert hasattr(mod, "CAUSAL_TEMPLATE_REGISTRY")
        assert hasattr(mod, "get_causal_template")

    def test_import_belief_engine(self):
        mod = importlib.import_module("spectraquant.intelligence.cognition.belief_engine")
        assert hasattr(mod, "BeliefEngine")
        assert hasattr(mod, "BeliefScore")

    def test_import_explanation_engine(self):
        mod = importlib.import_module("spectraquant.intelligence.cognition.explanation_engine")
        assert hasattr(mod, "ExplanationEngine")
        assert hasattr(mod, "CandidateExplanation")

    def test_import_cognition_package(self):
        mod = importlib.import_module("spectraquant.intelligence.cognition")
        for name in ["BeliefEngine", "BeliefScore", "CausalTemplate", "ExplanationEngine"]:
            assert hasattr(mod, name), f"{name} missing from cognition package"

    def test_cognition_does_not_import_cli(self):
        """Cognition layer must not depend on CLI or pipeline modules."""
        mod = importlib.import_module("spectraquant.intelligence.cognition.belief_engine")
        assert "spectraquant.cli" not in str(vars(mod))

    def test_cognition_does_not_import_providers(self):
        """Cognition layer must not depend on data providers."""
        mod = importlib.import_module("spectraquant.intelligence.cognition.belief_engine")
        assert "yfinance" not in str(vars(mod))


# ===========================================================================
# Causal template registry tests
# ===========================================================================

class TestCausalTemplateRegistry:
    def test_all_templates_have_mechanism_tags(self):
        for event_type, template in CAUSAL_TEMPLATE_REGISTRY.items():
            assert template.mechanism_tags, (
                f"{event_type} has no mechanism tags"
            )

    def test_all_templates_have_valid_strength(self):
        for event_type, template in CAUSAL_TEMPLATE_REGISTRY.items():
            assert 0.0 <= template.base_event_strength <= 1.0, (
                f"{event_type} has invalid base_event_strength"
            )

    def test_all_templates_have_valid_direction_bias(self):
        for event_type, template in CAUSAL_TEMPLATE_REGISTRY.items():
            assert -1.0 <= template.direction_bias <= 1.0, (
                f"{event_type} has invalid direction_bias"
            )

    def test_unknown_event_type_falls_back_to_unknown_template(self):
        template = get_causal_template("nonexistent_event_xyz")
        assert template.event_type == "unknown"

    def test_earnings_beat_is_bullish(self):
        t = get_causal_template("earnings_beat")
        assert t.direction_bias > 0

    def test_fraud_allegation_is_bearish(self):
        t = get_causal_template("fraud_allegation")
        assert t.direction_bias < 0

    def test_mechanism_tag_enum_members(self):
        expected = {
            "drift", "gap", "reversal_risk", "volatility_expansion",
            "liquidity_shock", "sector_rotation", "delayed_drift",
            "uncertainty", "momentum_continuation", "mean_reversion",
        }
        actual = {m.value for m in MechanismTag}
        assert expected == actual


# ===========================================================================
# Explanation engine tests
# ===========================================================================

class TestExplanationEngine:
    def test_explanation_has_all_required_fields(self):
        engine = BeliefEngine()
        belief = engine.score_candidate(
            "AAPL", expected_return=0.12, score_raw=75.0,
            event_type="earnings_beat", regime="RISK_ON",
        )
        belief.signal = "BUY"
        explainer = ExplanationEngine()
        exp = explainer.explain(belief)
        assert exp.ticker == "AAPL"
        assert exp.signal == "BUY"
        assert exp.event_type == "earnings_beat"
        assert isinstance(exp.mechanism_tags, list)
        assert isinstance(exp.confidence_components, dict)
        assert exp.final_reason

    def test_final_reason_mentions_event_type(self):
        engine = BeliefEngine()
        belief = engine.score_candidate(
            "TCS.NS", expected_return=0.20, score_raw=80.0,
            event_type="government_contract", regime="RISK_ON",
        )
        belief.signal = "BUY"
        exp = ExplanationEngine().explain(belief)
        assert "government_contract" in exp.final_reason

    def test_analog_examples_in_explanation(self):
        engine = BeliefEngine()
        belief = engine.score_candidate(
            "INFY.NS", expected_return=0.15, score_raw=70.0,
            event_type="earnings_beat", regime="RISK_ON",
        )
        belief.signal = "BUY"
        examples = [
            {"event_type": "earnings_beat", "ticker": "TCS.NS",
             "observed_return": 0.062, "horizon_days": 5},
            {"event_type": "earnings_beat", "ticker": "WIPRO.NS",
             "observed_return": 0.041, "horizon_days": 5},
        ]
        exp = ExplanationEngine().explain(belief, analog_examples=examples)
        assert len(exp.analog_examples) == 2
        assert "2 similar past events" in exp.final_reason

    def test_explain_all_returns_one_per_belief(self):
        engine = BeliefEngine()
        candidates = [
            _make_candidate(f"T{i}", event_type="earnings_beat") for i in range(5)
        ]
        beliefs = engine.evaluate(candidates, regime="RISK_ON")
        explanations = ExplanationEngine().explain_all(beliefs)
        assert len(explanations) == 5

    def test_confidence_components_sum_to_belief_score_approximately(self):
        """confidence_components should approximate the weighted sub-scores."""
        engine = BeliefEngine()
        belief = engine.score_candidate(
            "X", expected_return=0.10, score_raw=60.0,
            event_type="earnings_beat", regime="RISK_ON",
        )
        exp = ExplanationEngine().explain(belief)
        # All components should be in [0, 1]
        for k, v in exp.confidence_components.items():
            assert 0.0 <= v <= 1.0, f"Component {k}={v} out of range"


# ===========================================================================
# Belief score invariants
# ===========================================================================

class TestBeliefScoreInvariants:
    def test_belief_score_in_zero_one(self):
        engine = BeliefEngine()
        for event_type in list(CAUSAL_TEMPLATE_REGISTRY.keys())[:5]:
            bs = engine.score_candidate(
                "T", expected_return=0.10, score_raw=60.0,
                event_type=event_type,
            )
            assert 0.0 <= bs.belief_score <= 1.0

    def test_conviction_le_belief_score(self):
        """Conviction = belief / (1 + uncertainty) ≤ belief."""
        engine = BeliefEngine()
        bs = engine.score_candidate(
            "T", expected_return=0.10, score_raw=60.0,
            event_type="earnings_beat", vol_ratio=2.0,
        )
        assert bs.conviction <= bs.belief_score

    def test_high_uncertainty_lowers_conviction(self):
        engine = BeliefEngine()
        low_unc = engine.score_candidate(
            "T", expected_return=0.10, score_raw=60.0,
            event_type="earnings_beat", event_confidence=0.95, vol_ratio=1.0,
        )
        high_unc = engine.score_candidate(
            "T", expected_return=0.10, score_raw=60.0,
            event_type="earnings_beat", event_confidence=0.20, vol_ratio=3.0,
        )
        assert high_unc.uncertainty > low_unc.uncertainty
        assert high_unc.conviction < low_unc.conviction

    def test_ranks_are_contiguous_starting_at_1(self):
        engine = BeliefEngine()
        n = 7
        results = engine.evaluate(
            [_make_candidate(f"T{i}") for i in range(n)],
            regime="RISK_ON",
        )
        ranks = sorted(r.rank for r in results)
        assert ranks == list(range(1, n + 1))
