"""Tests for the V3 news-first market selector.

Covers:
* Empty records → RUN_NONE
* Equity-dominant events → RUN_EQUITIES
* Crypto-dominant events → RUN_CRYPTO
* Balanced strong events → RUN_BOTH
* Weak events → RUN_NONE
* PANIC veto → RUN_NONE regardless of scores
* RISK_OFF penalty behaviour (scores halved)
* EVENT_DRIVEN boost behaviour (scores multiplied by 1.2)
* Determinism: same input always produces same route and scores
* MarketSelectorDecision.route is a valid MarketRoute value
* Event-type affinity correctness (earnings → equity, listing → crypto)
* No dependence on provider-specific fields
* JSON serialization compatibility (to_dict / from_dict)
* MarketSelectorConfig typed dataclass
* MarketSelectorInput typed dataclass
* Extended affinity table (guidance, dividend, analyst, protocol_upgrade, etc.)
* Sentiment factor contribution to scoring
* risk_off_penalty_applied flag
* score_input() convenience method

All tests are self-contained: no network calls, no file-system side-effects.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from spectraquant_v3.core.enums import MarketRoute
from spectraquant_v3.core.news_schema import NewsIntelligenceRecord
from spectraquant_v3.intelligence.market_selector import (
    EVENT_ASSET_AFFINITY,
    MarketSelector,
    MarketSelectorConfig,
    MarketSelectorDecision,
    MarketSelectorInput,
    ScoredRecord,
)


# ===========================================================================
# Helpers
# ===========================================================================

# Use a fixed far-future timestamp so recency decay is ~1.0 for all records.
# This isolates the scoring logic from wall-clock time.
_FRESH_TS = "2099-01-01T00:00:00+00:00"


def _rec(
    asset: str,
    event_type: str = "earnings",
    impact_score: float = 0.9,
    confidence: float = 0.9,
    sentiment_score: float = 0.8,
    timestamp: str = _FRESH_TS,
    **kwargs: Any,
) -> NewsIntelligenceRecord:
    """Build a minimal but valid NewsIntelligenceRecord."""
    return NewsIntelligenceRecord(
        canonical_symbol="TEST",
        asset=asset,
        timestamp=timestamp,
        event_type=event_type,
        sentiment_score=sentiment_score,
        impact_score=impact_score,
        confidence=confidence,
        **kwargs,
    )


def _equity(**kw: Any) -> NewsIntelligenceRecord:
    return _rec(asset="equity", event_type="earnings", **kw)


def _crypto(**kw: Any) -> NewsIntelligenceRecord:
    return _rec(asset="crypto", event_type="listing", **kw)


# ===========================================================================
# Basic contract tests
# ===========================================================================


class TestMarketSelectorContract:
    """Verify the output contract of MarketSelector.score()."""

    def test_returns_decision_type(self) -> None:
        sel = MarketSelector()
        decision = sel.score([_equity()])
        assert isinstance(decision, MarketSelectorDecision)

    def test_route_is_valid_market_route(self) -> None:
        sel = MarketSelector()
        decision = sel.score([_equity()])
        assert isinstance(decision.route, MarketRoute)
        assert decision.route in list(MarketRoute)

    def test_scores_in_unit_interval(self) -> None:
        sel = MarketSelector()
        decision = sel.score([_equity(), _crypto()])
        assert 0.0 <= decision.equity_score <= 1.0
        assert 0.0 <= decision.crypto_score <= 1.0

    def test_record_counts_match(self) -> None:
        sel = MarketSelector()
        records = [_equity(), _equity(), _crypto()]
        decision = sel.score(records)
        assert decision.equity_record_count == 2
        assert decision.crypto_record_count == 1
        assert decision.record_count == 3

    def test_scored_at_is_populated(self) -> None:
        sel = MarketSelector()
        decision = sel.score([])
        assert decision.scored_at != ""

    def test_rationale_is_populated(self) -> None:
        sel = MarketSelector()
        decision = sel.score([_equity()])
        assert decision.rationale != ""
        assert "equity_score" in decision.rationale
        assert "crypto_score" in decision.rationale

    def test_threshold_fields_match_config(self) -> None:
        cfg = {"min_score_to_run": 0.25, "both_threshold": 0.70}
        sel = MarketSelector(config=cfg)
        decision = sel.score([])
        assert decision.threshold_used == 0.25
        assert decision.both_threshold_used == 0.70


# ===========================================================================
# Core routing tests
# ===========================================================================


class TestRoutingDecisions:
    """Verify the routing logic under various news input scenarios."""

    def test_empty_records_returns_run_none(self) -> None:
        sel = MarketSelector()
        decision = sel.score([])
        assert decision.route == MarketRoute.RUN_NONE
        assert decision.equity_score == 0.0
        assert decision.crypto_score == 0.0

    def test_equity_dominant_returns_run_equities(self) -> None:
        """Strong equity signals, no crypto → RUN_EQUITIES."""
        sel = MarketSelector()
        records = [_equity(impact_score=0.9, confidence=0.9) for _ in range(5)]
        decision = sel.score(records)
        assert decision.route == MarketRoute.RUN_EQUITIES

    def test_crypto_dominant_returns_run_crypto(self) -> None:
        """Strong crypto signals, no equity → RUN_CRYPTO."""
        sel = MarketSelector()
        records = [_crypto(impact_score=0.9, confidence=0.9) for _ in range(5)]
        decision = sel.score(records)
        assert decision.route == MarketRoute.RUN_CRYPTO

    def test_balanced_strong_returns_run_both(self) -> None:
        """Strong signals on both sides → RUN_BOTH."""
        sel = MarketSelector()
        records = [
            _equity(impact_score=0.95, confidence=0.95) for _ in range(5)
        ] + [
            _crypto(impact_score=0.95, confidence=0.95) for _ in range(5)
        ]
        decision = sel.score(records)
        assert decision.route == MarketRoute.RUN_BOTH

    def test_weak_events_returns_run_none(self) -> None:
        """Below-threshold signals on both sides → RUN_NONE."""
        sel = MarketSelector()
        records = [
            _equity(impact_score=0.05, confidence=0.05),
            _crypto(impact_score=0.05, confidence=0.05),
        ]
        decision = sel.score(records)
        assert decision.route == MarketRoute.RUN_NONE

    def test_one_side_strong_one_weak(self) -> None:
        """One side clearly above threshold, other below → single-class route."""
        sel = MarketSelector()
        strong_equity = [_equity(impact_score=0.9, confidence=0.9) for _ in range(5)]
        weak_crypto = [_crypto(impact_score=0.02, confidence=0.02)]
        decision = sel.score(strong_equity + weak_crypto)
        assert decision.route == MarketRoute.RUN_EQUITIES


# ===========================================================================
# Regime handling tests
# ===========================================================================


class TestRegimeHandling:
    """Verify regime veto and multiplier behaviour."""

    def test_panic_veto_forces_run_none(self) -> None:
        """PANIC overrides any score, no matter how strong."""
        sel = MarketSelector()
        strong = [
            _equity(impact_score=1.0, confidence=1.0) for _ in range(10)
        ] + [
            _crypto(impact_score=1.0, confidence=1.0) for _ in range(10)
        ]
        decision = sel.score(strong, regime_label="PANIC")
        assert decision.route == MarketRoute.RUN_NONE
        assert decision.regime_vetoed is True

    def test_panic_veto_reported_in_rationale(self) -> None:
        sel = MarketSelector()
        decision = sel.score([_equity()], regime_label="PANIC")
        assert "PANIC" in decision.rationale or "vetoed" in decision.rationale.lower()

    def test_risk_off_reduces_scores(self) -> None:
        """RISK_OFF multiplies both scores by 0.5."""
        sel = MarketSelector()
        records = [_equity(impact_score=0.8, confidence=0.8) for _ in range(3)]
        normal = sel.score(records, regime_label="UNKNOWN")
        risk_off = sel.score(records, regime_label="RISK_OFF")
        # Scores under RISK_OFF must be strictly lower than neutral
        assert risk_off.equity_score < normal.equity_score
        # The ratio should be ~0.5 (may be clipped at 1.0)
        if normal.equity_score > 0:
            ratio = risk_off.equity_score / normal.equity_score
            assert abs(ratio - 0.5) < 1e-6

    def test_risk_off_can_push_score_below_threshold(self) -> None:
        """RISK_OFF may push a borderline score below threshold → RUN_NONE."""
        # Choose impact/confidence so that the raw score is just above threshold
        # but halved is below.
        # min_score default = 0.35; raw ≈ 0.4, risk_off ≈ 0.2 → RUN_NONE
        sel = MarketSelector()
        # listing affinity for equity is 0.10 — use earnings (1.0)
        # raw weight per record ≈ 1.0 * impact * affinity_equity * confidence
        # We need: impact * 1.0 * confidence ≈ 0.40
        # => impact=0.7, confidence=0.57 → 0.7*0.57 ≈ 0.40
        records = [_equity(impact_score=0.7, confidence=0.57)]
        neutral_decision = sel.score(records, regime_label="UNKNOWN")
        risk_off_decision = sel.score(records, regime_label="RISK_OFF")
        # Neutral should be above or near threshold
        assert neutral_decision.equity_score >= 0.35
        # RISK_OFF should push it below
        assert risk_off_decision.equity_score < 0.35

    def test_event_driven_boosts_scores(self) -> None:
        """EVENT_DRIVEN multiplies scores by 1.2."""
        sel = MarketSelector()
        records = [_equity(impact_score=0.5, confidence=0.5) for _ in range(3)]
        normal = sel.score(records, regime_label="UNKNOWN")
        event_driven = sel.score(records, regime_label="EVENT_DRIVEN")
        assert event_driven.equity_score >= normal.equity_score

    def test_event_driven_boost_ratio(self) -> None:
        """EVENT_DRIVEN score should be 1.2× neutral (when not clamped)."""
        sel = MarketSelector()
        # Use a score that stays < 1/1.2 ≈ 0.833 to avoid clamping
        records = [_equity(impact_score=0.5, confidence=0.5)]
        normal = sel.score(records, regime_label="UNKNOWN")
        event_driven = sel.score(records, regime_label="EVENT_DRIVEN")
        if normal.equity_score > 0:
            ratio = event_driven.equity_score / normal.equity_score
            assert abs(ratio - 1.2) < 1e-6

    def test_unknown_regime_is_neutral(self) -> None:
        """Unrecognised regime labels should not alter scores."""
        sel = MarketSelector()
        records = [_equity()]
        explicit_unknown = sel.score(records, regime_label="UNKNOWN")
        some_other = sel.score(records, regime_label="TRENDING")
        assert explicit_unknown.equity_score == pytest.approx(
            some_other.equity_score, rel=1e-9
        )


# ===========================================================================
# Determinism tests
# ===========================================================================


class TestDeterminism:
    """Verify that identical inputs always produce identical outputs."""

    def test_same_inputs_same_route(self) -> None:
        sel = MarketSelector()
        records = [_equity(), _crypto()]
        first = sel.score(records)
        second = sel.score(records)
        assert first.route == second.route

    def test_same_inputs_same_scores(self) -> None:
        sel = MarketSelector()
        records = [_equity(impact_score=0.7, confidence=0.8) for _ in range(3)]
        first = sel.score(records)
        second = sel.score(records)
        assert first.equity_score == second.equity_score
        assert first.crypto_score == second.crypto_score

    def test_multiple_calls_do_not_mutate_state(self) -> None:
        sel = MarketSelector()
        records = [_equity(), _crypto()]
        results = [sel.score(records) for _ in range(5)]
        routes = [r.route for r in results]
        assert len(set(routes)) == 1, "Route changed across repeated calls"

    def test_selector_instances_agree(self) -> None:
        """Two selectors with the same config produce the same decision."""
        cfg = {"min_score_to_run": 0.30, "both_threshold": 0.55}
        sel_a = MarketSelector(config=cfg)
        sel_b = MarketSelector(config=cfg)
        records = [_equity(impact_score=0.8, confidence=0.9) for _ in range(4)]
        assert sel_a.score(records).route == sel_b.score(records).route


# ===========================================================================
# Event-type affinity tests
# ===========================================================================


class TestEventTypeAffinity:
    """Verify that event_type correctly modulates per-asset-class scoring."""

    def test_earnings_boosts_equity_not_crypto(self) -> None:
        """earnings event should contribute much more to equity than crypto."""
        sel = MarketSelector()
        earnings_rec = _rec(
            asset="equity",
            event_type="earnings",
            impact_score=0.9,
            confidence=0.9,
        )
        decision = sel.score([earnings_rec])
        # equity_score should be high; crypto_score should be 0 (no crypto records)
        assert decision.equity_score > 0.5
        assert decision.crypto_score == 0.0

    def test_listing_boosts_crypto_not_equity(self) -> None:
        """listing event should contribute much more to crypto than equity."""
        sel = MarketSelector()
        listing_rec = _rec(
            asset="crypto",
            event_type="listing",
            impact_score=0.9,
            confidence=0.9,
        )
        decision = sel.score([listing_rec])
        assert decision.crypto_score > 0.5
        assert decision.equity_score == 0.0

    def test_affinity_table_equity_earnings_vs_listing(self) -> None:
        """Within the affinity table, earnings equity > listing equity."""
        assert EVENT_ASSET_AFFINITY["earnings"]["equity"] > EVENT_ASSET_AFFINITY["listing"]["equity"]

    def test_affinity_table_crypto_listing_vs_earnings(self) -> None:
        """Within the affinity table, listing crypto > earnings crypto."""
        assert EVENT_ASSET_AFFINITY["listing"]["crypto"] > EVENT_ASSET_AFFINITY["earnings"]["crypto"]

    def test_unknown_event_type_gets_default_affinity(self) -> None:
        """Records with unrecognised event_type use the 0.5/0.5 default."""
        sel = MarketSelector()
        equity_unknown = _rec(
            asset="equity",
            event_type="totally_new_event_type",
            impact_score=0.8,
            confidence=0.8,
        )
        decision = sel.score([equity_unknown])
        # Score should be non-zero (default affinity 0.5)
        assert decision.equity_score > 0.0

    def test_macro_contributes_to_both_asset_classes(self) -> None:
        """macro has meaningful affinity for both asset classes."""
        assert EVENT_ASSET_AFFINITY["macro"]["equity"] > 0.4
        assert EVENT_ASSET_AFFINITY["macro"]["crypto"] > 0.4

    def test_regulatory_is_crypto_dominant(self) -> None:
        """regulatory affinity is higher for crypto than equity."""
        assert EVENT_ASSET_AFFINITY["regulatory"]["crypto"] > EVENT_ASSET_AFFINITY["regulatory"]["equity"]


# ===========================================================================
# Provider-agnostic field usage
# ===========================================================================


class TestProviderAgnostic:
    """Verify no dependence on provider-specific fields."""

    def test_provider_field_does_not_affect_score(self) -> None:
        """The provider name must not influence the computed score."""
        sel = MarketSelector()
        rec_a = _equity(provider="perplexity")
        rec_b = _equity(provider="newsapi")
        rec_c = _equity(provider="")  # no provider
        assert sel.score([rec_a]).equity_score == pytest.approx(
            sel.score([rec_b]).equity_score, rel=1e-9
        )
        assert sel.score([rec_a]).equity_score == pytest.approx(
            sel.score([rec_c]).equity_score, rel=1e-9
        )

    def test_source_urls_do_not_affect_score(self) -> None:
        """source_urls must not influence the computed score."""
        sel = MarketSelector()
        rec_a = _equity(source_urls=["https://example.com/1"])
        rec_b = _equity(source_urls=[])
        rec_c = _equity(source_urls=["https://a.com", "https://b.com", "https://c.com"])
        scores = {sel.score([r]).equity_score for r in [rec_a, rec_b, rec_c]}
        assert len(scores) == 1, f"source_urls changed score: {scores}"

    def test_raw_response_does_not_affect_score(self) -> None:
        """raw_response must not influence the computed score."""
        sel = MarketSelector()
        rec_a = _equity(raw_response={})
        rec_b = _equity(raw_response={"extra": "data", "nested": {"key": 1}})
        assert sel.score([rec_a]).equity_score == pytest.approx(
            sel.score([rec_b]).equity_score, rel=1e-9
        )

    def test_rationale_field_does_not_affect_score(self) -> None:
        """The record's rationale string must not influence score."""
        sel = MarketSelector()
        rec_a = _equity(rationale="Strong Q4 results.")
        rec_b = _equity(rationale="")
        assert sel.score([rec_a]).equity_score == pytest.approx(
            sel.score([rec_b]).equity_score, rel=1e-9
        )


# ===========================================================================
# Top-N contributing records
# ===========================================================================


class TestTopRecords:
    """Verify the top contributing records in the decision output."""

    def test_top_equity_records_populated(self) -> None:
        sel = MarketSelector()
        records = [_equity() for _ in range(3)]
        decision = sel.score(records)
        assert len(decision.top_equity_records) <= 3
        assert all(isinstance(r, ScoredRecord) for r in decision.top_equity_records)

    def test_top_equity_records_sorted_by_weight_desc(self) -> None:
        sel = MarketSelector()
        records = [
            _equity(impact_score=0.9, confidence=0.9),
            _equity(impact_score=0.3, confidence=0.3),
            _equity(impact_score=0.6, confidence=0.6),
        ]
        decision = sel.score(records)
        weights = [r.raw_weight for r in decision.top_equity_records]
        assert weights == sorted(weights, reverse=True)

    def test_top_n_config_respected(self) -> None:
        sel = MarketSelector(config={"top_n": 2})
        records = [_equity() for _ in range(10)]
        decision = sel.score(records)
        assert len(decision.top_equity_records) <= 2

    def test_empty_records_empty_top(self) -> None:
        sel = MarketSelector()
        decision = sel.score([])
        assert decision.top_equity_records == []
        assert decision.top_crypto_records == []


# ===========================================================================
# Config / threshold edge cases
# ===========================================================================


class TestConfigThresholds:
    """Verify threshold parameters are respected."""

    def test_custom_min_score_to_run(self) -> None:
        """Setting a very high threshold should force RUN_NONE on moderate signals."""
        sel = MarketSelector(config={"min_score_to_run": 0.99})
        records = [_equity(impact_score=0.8, confidence=0.8)]
        decision = sel.score(records)
        assert decision.route == MarketRoute.RUN_NONE

    def test_zero_min_score_allows_any_score(self) -> None:
        """Setting threshold to 0 means any non-zero score qualifies."""
        sel = MarketSelector(config={"min_score_to_run": 0.0})
        records = [_equity(impact_score=0.01, confidence=0.01)]
        decision = sel.score(records)
        assert decision.route != MarketRoute.RUN_NONE

    def test_both_threshold_triggers_run_both(self) -> None:
        """When both scores exceed both_threshold, route should be RUN_BOTH."""
        # Use a low both_threshold so moderate signals trigger RUN_BOTH
        sel = MarketSelector(config={"both_threshold": 0.10, "min_score_to_run": 0.05})
        records = [
            _equity(impact_score=0.9, confidence=0.9),
            _crypto(impact_score=0.9, confidence=0.9),
        ]
        decision = sel.score(records)
        assert decision.route == MarketRoute.RUN_BOTH


# ===========================================================================
# MarketSelectorConfig typed dataclass tests
# ===========================================================================


class TestMarketSelectorConfig:
    """Verify the typed MarketSelectorConfig dataclass."""

    def test_default_values(self) -> None:
        cfg = MarketSelectorConfig()
        assert cfg.min_score_to_run == 0.35
        assert cfg.both_threshold == 0.60
        assert cfg.half_life_hours == 6.0
        assert cfg.top_n == 5

    def test_custom_values(self) -> None:
        cfg = MarketSelectorConfig(
            min_score_to_run=0.20,
            both_threshold=0.70,
            half_life_hours=12.0,
            top_n=3,
        )
        assert cfg.min_score_to_run == 0.20
        assert cfg.both_threshold == 0.70
        assert cfg.half_life_hours == 12.0
        assert cfg.top_n == 3

    def test_invalid_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life_hours"):
            MarketSelectorConfig(half_life_hours=0.0)

    def test_invalid_top_n_raises(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            MarketSelectorConfig(top_n=-1)

    def test_from_dict(self) -> None:
        cfg = MarketSelectorConfig.from_dict(
            {"min_score_to_run": 0.25, "both_threshold": 0.75}
        )
        assert cfg.min_score_to_run == 0.25
        assert cfg.both_threshold == 0.75
        # Unspecified fields default
        assert cfg.half_life_hours == 6.0
        assert cfg.top_n == 5

    def test_to_dict_roundtrip(self) -> None:
        cfg = MarketSelectorConfig(min_score_to_run=0.28, both_threshold=0.65)
        d = cfg.to_dict()
        assert d["min_score_to_run"] == 0.28
        assert d["both_threshold"] == 0.65
        restored = MarketSelectorConfig.from_dict(d)
        assert restored.min_score_to_run == cfg.min_score_to_run
        assert restored.both_threshold == cfg.both_threshold

    def test_selector_accepts_config_object(self) -> None:
        """MarketSelector accepts a MarketSelectorConfig instance."""
        cfg = MarketSelectorConfig(min_score_to_run=0.25, both_threshold=0.70)
        sel = MarketSelector(config=cfg)
        decision = sel.score([])
        assert decision.threshold_used == 0.25
        assert decision.both_threshold_used == 0.70

    def test_selector_config_matches_dict_config(self) -> None:
        """Typed config and equivalent dict config produce the same thresholds."""
        cfg = MarketSelectorConfig(min_score_to_run=0.30, both_threshold=0.65)
        sel_typed = MarketSelector(config=cfg)
        sel_dict = MarketSelector(config={"min_score_to_run": 0.30, "both_threshold": 0.65})
        records = [_equity(impact_score=0.8, confidence=0.8)]
        d_typed = sel_typed.score(records)
        d_dict = sel_dict.score(records)
        assert d_typed.route == d_dict.route
        assert d_typed.equity_score == pytest.approx(d_dict.equity_score, rel=1e-9)


# ===========================================================================
# MarketSelectorInput typed dataclass tests
# ===========================================================================


class TestMarketSelectorInput:
    """Verify the typed MarketSelectorInput dataclass."""

    def test_default_values(self) -> None:
        si = MarketSelectorInput()
        assert si.records == []
        assert si.regime_label == "UNKNOWN"
        assert si.as_of_utc == ""

    def test_custom_values(self) -> None:
        records = [_equity()]
        si = MarketSelectorInput(
            records=records,
            regime_label="RISK_OFF",
            as_of_utc="2026-03-11T09:00:00Z",
        )
        assert si.records == records
        assert si.regime_label == "RISK_OFF"
        assert si.as_of_utc == "2026-03-11T09:00:00Z"

    def test_to_dict_structure(self) -> None:
        si = MarketSelectorInput(
            records=[_equity()],
            regime_label="NORMAL",
            as_of_utc="2026-03-11T09:00:00Z",
        )
        d = si.to_dict()
        assert "records" in d
        assert isinstance(d["records"], list)
        assert len(d["records"]) == 1
        assert d["regime_label"] == "NORMAL"
        assert d["as_of_utc"] == "2026-03-11T09:00:00Z"

    def test_score_input_matches_score(self) -> None:
        """score_input() with MarketSelectorInput matches score() directly."""
        sel = MarketSelector()
        records = [_equity(impact_score=0.9, confidence=0.9) for _ in range(3)]
        direct = sel.score(records, regime_label="RISK_OFF")
        via_input = sel.score_input(
            MarketSelectorInput(records=records, regime_label="RISK_OFF")
        )
        assert direct.route == via_input.route
        assert direct.equity_score == pytest.approx(via_input.equity_score, rel=1e-9)

    def test_score_input_empty(self) -> None:
        sel = MarketSelector()
        decision = sel.score_input(MarketSelectorInput())
        assert decision.route == MarketRoute.RUN_NONE

    def test_score_input_panic(self) -> None:
        sel = MarketSelector()
        strong = [_equity(impact_score=1.0, confidence=1.0) for _ in range(5)]
        decision = sel.score_input(
            MarketSelectorInput(records=strong, regime_label="PANIC")
        )
        assert decision.route == MarketRoute.RUN_NONE
        assert decision.regime_vetoed is True


# ===========================================================================
# JSON serialization tests
# ===========================================================================


class TestJSONSerialization:
    """Verify JSON-compatible serialization of MarketSelectorDecision."""

    def test_to_dict_returns_dict(self) -> None:
        sel = MarketSelector()
        decision = sel.score([_equity()])
        d = decision.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_route_is_string(self) -> None:
        """route in to_dict must be a string, not a MarketRoute enum."""
        sel = MarketSelector()
        decision = sel.score([_equity()])
        d = decision.to_dict()
        assert isinstance(d["route"], str)
        assert d["route"] in {r.value for r in MarketRoute}

    def test_to_dict_json_serializable(self) -> None:
        """to_dict output must be fully JSON-serializable."""
        sel = MarketSelector()
        records = [_equity(), _crypto()]
        decision = sel.score(records, regime_label="RISK_OFF")
        d = decision.to_dict()
        # Must not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_dict_all_required_fields_present(self) -> None:
        required = {
            "route", "equity_score", "crypto_score", "equity_record_count",
            "crypto_record_count", "record_count", "regime_label",
            "regime_vetoed", "risk_off_penalty_applied", "threshold_used",
            "both_threshold_used", "rationale", "scored_at",
            "top_equity_records", "top_crypto_records", "version",
        }
        sel = MarketSelector()
        d = sel.score([_equity()]).to_dict()
        assert required.issubset(set(d.keys()))

    def test_to_dict_version_is_v1(self) -> None:
        sel = MarketSelector()
        d = sel.score([]).to_dict()
        assert d["version"] == "v1"

    def test_to_dict_top_records_serialized(self) -> None:
        sel = MarketSelector()
        records = [_equity()]
        d = sel.score(records).to_dict()
        assert isinstance(d["top_equity_records"], list)
        if d["top_equity_records"]:
            rec = d["top_equity_records"][0]
            assert "canonical_symbol" in rec
            assert "event_type" in rec
            assert "asset" in rec
            assert "raw_weight" in rec

    def test_scored_record_to_dict(self) -> None:
        sr = ScoredRecord(
            canonical_symbol="BTC",
            event_type="listing",
            asset="crypto",
            raw_weight=0.72345,
        )
        d = sr.to_dict()
        assert d["canonical_symbol"] == "BTC"
        assert d["event_type"] == "listing"
        assert d["asset"] == "crypto"
        assert d["raw_weight"] == 0.72345

    def test_risk_off_penalty_flag_in_dict(self) -> None:
        sel = MarketSelector()
        d_risk = sel.score([_equity()], regime_label="RISK_OFF").to_dict()
        d_normal = sel.score([_equity()], regime_label="UNKNOWN").to_dict()
        assert d_risk["risk_off_penalty_applied"] is True
        assert d_normal["risk_off_penalty_applied"] is False


# ===========================================================================
# Extended affinity table tests
# ===========================================================================


class TestExtendedAffinityTable:
    """Verify the extended event-type affinity table entries from the spec."""

    def test_guidance_is_equity_dominant(self) -> None:
        assert EVENT_ASSET_AFFINITY["guidance"]["equity"] >= 0.90
        assert EVENT_ASSET_AFFINITY["guidance"]["crypto"] <= 0.10

    def test_dividend_is_equity_only(self) -> None:
        assert EVENT_ASSET_AFFINITY["dividend"]["equity"] >= 0.80
        assert EVENT_ASSET_AFFINITY["dividend"]["crypto"] == 0.0

    def test_analyst_is_equity_leaning(self) -> None:
        assert EVENT_ASSET_AFFINITY["analyst"]["equity"] >= 0.70
        assert EVENT_ASSET_AFFINITY["analyst"]["equity"] > EVENT_ASSET_AFFINITY["analyst"]["crypto"]

    def test_protocol_upgrade_is_crypto_dominant(self) -> None:
        assert EVENT_ASSET_AFFINITY["protocol_upgrade"]["crypto"] >= 0.90
        assert EVENT_ASSET_AFFINITY["protocol_upgrade"]["equity"] <= 0.10

    def test_exchange_hack_is_crypto_only(self) -> None:
        assert EVENT_ASSET_AFFINITY["exchange_hack"]["crypto"] == 1.0
        assert EVENT_ASSET_AFFINITY["exchange_hack"]["equity"] == 0.0

    def test_onchain_is_crypto_dominant(self) -> None:
        assert EVENT_ASSET_AFFINITY["onchain"]["crypto"] >= 0.90
        assert EVENT_ASSET_AFFINITY["onchain"]["equity"] == 0.0

    def test_sector_theme_is_cross_asset(self) -> None:
        """sector_theme has meaningful affinity for both asset classes."""
        assert EVENT_ASSET_AFFINITY["sector_theme"]["equity"] >= 0.60
        assert EVENT_ASSET_AFFINITY["sector_theme"]["crypto"] >= 0.50

    def test_social_buzz_is_crypto_leaning(self) -> None:
        assert EVENT_ASSET_AFFINITY["social_buzz"]["crypto"] > EVENT_ASSET_AFFINITY["social_buzz"]["equity"]

    def test_affinity_table_covers_all_spec_entries(self) -> None:
        """All event types from the design spec must be present in the table."""
        required_entries = {
            "earnings", "guidance", "m_and_a", "dividend", "analyst",
            "macro", "regulatory", "protocol_upgrade", "exchange_hack",
            "onchain", "sector_theme", "social_buzz",
        }
        assert required_entries.issubset(set(EVENT_ASSET_AFFINITY.keys()))

    def test_affinity_case_insensitive_via_selector(self) -> None:
        """Event type lookup is case-insensitive (uppercase input works)."""
        sel = MarketSelector()
        rec_lower = _rec(asset="equity", event_type="earnings")
        rec_upper = _rec(asset="equity", event_type="EARNINGS")
        score_lower = sel.score([rec_lower]).equity_score
        score_upper = sel.score([rec_upper]).equity_score
        assert score_lower == pytest.approx(score_upper, rel=1e-9)


# ===========================================================================
# Sentiment factor tests
# ===========================================================================


class TestSentimentFactor:
    """Verify that sentiment magnitude modulates the scoring."""

    def test_zero_sentiment_reduces_score(self) -> None:
        """A record with zero sentiment scores lower than one with high sentiment."""
        sel = MarketSelector()
        rec_high = _equity(sentiment_score=1.0, impact_score=0.8, confidence=0.8)
        rec_zero = _equity(sentiment_score=0.0, impact_score=0.8, confidence=0.8)
        score_high = sel.score([rec_high]).equity_score
        score_zero = sel.score([rec_zero]).equity_score
        assert score_high > score_zero

    def test_positive_and_negative_equal_magnitude_same_score(self) -> None:
        """Positive and negative sentiment of same magnitude produce the same score."""
        sel = MarketSelector()
        rec_pos = _equity(sentiment_score=0.6, impact_score=0.7, confidence=0.7)
        rec_neg = _equity(sentiment_score=-0.6, impact_score=0.7, confidence=0.7)
        score_pos = sel.score([rec_pos]).equity_score
        score_neg = sel.score([rec_neg]).equity_score
        assert score_pos == pytest.approx(score_neg, rel=1e-9)

    def test_sentiment_factor_bounds(self) -> None:
        """sentiment_factor is in [0.5, 1.0] so it only reduces scores, never exceeds base."""
        # At sentiment=0 factor=0.5; at sentiment=1 factor=1.0
        sel = MarketSelector()
        rec_zero_sent = _equity(sentiment_score=0.0, impact_score=0.9, confidence=0.9)
        rec_full_sent = _equity(sentiment_score=1.0, impact_score=0.9, confidence=0.9)
        s_zero = sel.score([rec_zero_sent]).equity_score
        s_full = sel.score([rec_full_sent]).equity_score
        # ratio should be approximately 0.5 / 1.0 = 0.5
        assert abs(s_zero / s_full - 0.5) < 1e-6


# ===========================================================================
# risk_off_penalty_applied flag tests
# ===========================================================================


class TestRiskOffPenaltyFlag:
    """Verify the risk_off_penalty_applied veto flag."""

    def test_risk_off_sets_flag(self) -> None:
        sel = MarketSelector()
        decision = sel.score([_equity()], regime_label="RISK_OFF")
        assert decision.risk_off_penalty_applied is True

    def test_other_regimes_do_not_set_flag(self) -> None:
        sel = MarketSelector()
        for regime in ("UNKNOWN", "NORMAL", "EVENT_DRIVEN", "PANIC"):
            decision = sel.score([_equity()], regime_label=regime)
            assert decision.risk_off_penalty_applied is False, (
                f"Expected risk_off_penalty_applied=False for regime={regime!r}"
            )
