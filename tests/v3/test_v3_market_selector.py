"""Focused tests for the deterministic V3 news-first market selector."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any

import pytest

from spectraquant_v3.core.enums import MarketRoute
from spectraquant_v3.core.news_schema import NewsIntelligenceRecord
from spectraquant_v3.intelligence.market_selector import (
    EVENT_ASSET_AFFINITY,
    ContributingEventSummary,
    MarketRegimes,
    MarketRiskFlags,
    MarketSelector,
    MarketSelectorConfig,
    MarketSelectorDecision,
    MarketSelectorInput,
    ScoreBreakdown,
    SelectorRationale,
    VetoFlags,
    canonicalize_event_type,
    get_event_asset_affinity,
)

_AS_OF = "2026-03-11T09:00:00Z"
_FRESH_TS = "2026-03-11T08:45:00Z"
_STALE_TS = "2026-03-09T08:45:00Z"


def _event(
    *,
    canonical_symbol: str,
    asset: str,
    event_type: str,
    timestamp: str = _FRESH_TS,
    sentiment_score: float = 0.8,
    impact_score: float = 0.9,
    confidence: float = 0.9,
    article_count: int = 5,
    **overrides: Any,
) -> NewsIntelligenceRecord:
    return NewsIntelligenceRecord(
        canonical_symbol=canonical_symbol,
        asset=asset,
        timestamp=timestamp,
        event_type=event_type,
        sentiment_score=sentiment_score,
        impact_score=impact_score,
        confidence=confidence,
        article_count=article_count,
        source_urls=overrides.pop("source_urls", ["https://example.com/news"]),
        rationale=overrides.pop("rationale", "normalized event"),
        provider=overrides.pop("provider", "test-provider"),
        raw_response=overrides.pop("raw_response", {"provider_specific": True}),
        **overrides,
    )


def _selector_input(
    events: list[NewsIntelligenceRecord],
    *,
    regimes: MarketRegimes | None = None,
    risk_flags: MarketRiskFlags | None = None,
    config: MarketSelectorConfig | None = None,
) -> MarketSelectorInput:
    return MarketSelectorInput(
        as_of_utc=_AS_OF,
        news_events=events,
        regimes=regimes or MarketRegimes(),
        risk_flags=risk_flags or MarketRiskFlags(),
        config=config,
    )


class TestMarketSelectorRouting:
    def test_empty_records_returns_run_none(self) -> None:
        selector = MarketSelector()
        decision = selector.score_input(_selector_input([]))

        assert decision.route == MarketRoute.RUN_NONE
        assert decision.equity_score == 0.0
        assert decision.crypto_score == 0.0
        assert decision.rationale.primary_reason.startswith("No actionable")

    def test_equity_dominant_events_return_run_equities(self) -> None:
        selector = MarketSelector()
        events = [
            _event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS"),
            _event(canonical_symbol="TCS.NS", asset="equity", event_type="GUIDANCE", impact_score=0.85),
            _event(canonical_symbol="BTC", asset="crypto", event_type="SOCIAL_BUZZ", impact_score=0.22, confidence=0.35),
        ]

        decision = selector.score_input(_selector_input(events))

        assert decision.route == MarketRoute.RUN_EQUITIES
        assert decision.equity_score > decision.crypto_score
        assert decision.equity_score >= decision.thresholds.high_opportunity_threshold
        assert decision.rationale.top_contributing_events
        assert decision.rationale.top_contributing_events[0].asset == "equity"

    def test_crypto_dominant_events_return_run_crypto(self) -> None:
        selector = MarketSelector()
        events = [
            _event(canonical_symbol="BTC", asset="crypto", event_type="PROTOCOL_UPGRADE"),
            _event(canonical_symbol="ETH", asset="crypto", event_type="ONCHAIN", impact_score=0.88),
            _event(canonical_symbol="AAPL", asset="equity", event_type="SOCIAL_BUZZ", impact_score=0.18, confidence=0.30),
        ]

        decision = selector.score_input(_selector_input(events))

        assert decision.route == MarketRoute.RUN_CRYPTO
        assert decision.crypto_score > decision.equity_score
        assert decision.crypto_score >= decision.thresholds.high_opportunity_threshold

    def test_balanced_strong_events_return_run_both(self) -> None:
        selector = MarketSelector()
        events = [
            _event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS"),
            _event(canonical_symbol="MSFT", asset="equity", event_type="GUIDANCE", impact_score=0.88),
            _event(canonical_symbol="BTC", asset="crypto", event_type="PROTOCOL_UPGRADE"),
            _event(canonical_symbol="ETH", asset="crypto", event_type="ONCHAIN", impact_score=0.88),
        ]

        decision = selector.score_input(_selector_input(events))

        assert decision.route == MarketRoute.RUN_BOTH
        assert decision.equity_score >= decision.thresholds.high_opportunity_threshold
        assert decision.crypto_score >= decision.thresholds.high_opportunity_threshold

    def test_weak_events_return_run_none(self) -> None:
        selector = MarketSelector()
        events = [
            _event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS", impact_score=0.08, confidence=0.10),
            _event(canonical_symbol="BTC", asset="crypto", event_type="ONCHAIN", impact_score=0.06, confidence=0.10),
        ]

        decision = selector.score_input(_selector_input(events))

        assert decision.route == MarketRoute.RUN_NONE
        assert decision.equity_score < decision.thresholds.low_opportunity_floor
        assert decision.crypto_score < decision.thresholds.low_opportunity_floor

    def test_score_adapter_remains_available(self) -> None:
        selector = MarketSelector()
        records = [_event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS")]

        direct = selector.score(records, regime_label="EVENT_DRIVEN", as_of_utc=_AS_OF)
        wrapped = selector.score_input(
            _selector_input(records, regimes=MarketRegimes(global_regime="EVENT_DRIVEN"))
        )

        assert direct.to_dict() == wrapped.to_dict()


class TestRegimesAndRiskFlags:
    def test_panic_veto_forces_run_none(self) -> None:
        selector = MarketSelector()
        events = [
            _event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS"),
            _event(canonical_symbol="BTC", asset="crypto", event_type="PROTOCOL_UPGRADE"),
        ]

        decision = selector.score_input(
            _selector_input(
                events,
                regimes=MarketRegimes(global_regime="PANIC"),
                risk_flags=MarketRiskFlags(panic_mode=True),
            )
        )

        assert decision.route == MarketRoute.RUN_NONE
        assert decision.veto_flags.panic_veto is True
        assert "Panic" in decision.rationale.primary_reason

    def test_risk_off_penalty_is_applied(self) -> None:
        selector = MarketSelector()
        events = [_event(canonical_symbol="BTC", asset="crypto", event_type="REGULATORY", impact_score=0.82)]

        neutral = selector.score_input(_selector_input(events))
        risk_off = selector.score_input(
            _selector_input(events, regimes=MarketRegimes(global_regime="NORMAL", crypto_regime="RISK_OFF"))
        )

        assert risk_off.crypto_score < neutral.crypto_score
        assert risk_off.veto_flags.risk_off_penalty_applied is True

    def test_event_driven_boost_is_applied(self) -> None:
        selector = MarketSelector()
        events = [_event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS", impact_score=0.55)]

        neutral = selector.score_input(_selector_input(events))
        boosted = selector.score_input(
            _selector_input(events, regimes=MarketRegimes(global_regime="NORMAL", equity_regime="EVENT_DRIVEN"))
        )

        assert boosted.equity_score > neutral.equity_score
        assert any("EVENT_DRIVEN" in reason for reason in boosted.rationale.secondary_reasons)

    def test_high_cross_asset_stress_penalizes_scores(self) -> None:
        selector = MarketSelector()
        events = [_event(canonical_symbol="BTC", asset="crypto", event_type="MACRO", impact_score=0.75)]

        neutral = selector.score_input(_selector_input(events))
        stressed = selector.score_input(
            _selector_input(events, risk_flags=MarketRiskFlags(high_cross_asset_stress=True))
        )

        assert stressed.crypto_score < neutral.crypto_score
        assert stressed.veto_flags.risk_off_penalty_applied is True


class TestScoringAndDeterminism:
    def test_recency_decay_prefers_fresher_events(self) -> None:
        selector = MarketSelector()
        fresh = _event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS", timestamp=_FRESH_TS)
        stale = _event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS", timestamp=_STALE_TS)

        fresh_decision = selector.score_input(_selector_input([fresh]))
        stale_decision = selector.score_input(_selector_input([stale]))

        assert fresh_decision.equity_score > stale_decision.equity_score

    def test_selector_is_deterministic_for_identical_inputs(self) -> None:
        selector = MarketSelector()
        selector_input = _selector_input(
            [
                _event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS"),
                _event(canonical_symbol="BTC", asset="crypto", event_type="REGULATORY", sentiment_score=-0.7),
            ],
            regimes=MarketRegimes(global_regime="NORMAL", equity_regime="EVENT_DRIVEN", crypto_regime="RISK_OFF"),
        )

        first = selector.score_input(selector_input)
        second = selector.score_input(selector_input)

        assert first.to_dict() == second.to_dict()

    def test_provider_specific_fields_do_not_affect_score(self) -> None:
        selector = MarketSelector()
        base_event = _event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS")
        variant = _event(
            canonical_symbol="INFY.NS",
            asset="equity",
            event_type="EARNINGS",
            provider="different-provider",
            source_urls=["https://other.example.com/item"],
            rationale="different provider payload",
            raw_response={"opaque": "payload"},
        )

        base_decision = selector.score_input(_selector_input([base_event]))
        variant_decision = selector.score_input(_selector_input([variant]))

        assert base_decision.equity_score == pytest.approx(variant_decision.equity_score)
        assert base_decision.route == variant_decision.route

    def test_top_contributing_events_are_present_and_sorted(self) -> None:
        selector = MarketSelector()
        events = [
            _event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS", impact_score=0.95),
            _event(canonical_symbol="TCS.NS", asset="equity", event_type="GUIDANCE", impact_score=0.65),
            _event(canonical_symbol="BTC", asset="crypto", event_type="SOCIAL_BUZZ", impact_score=0.12, confidence=0.25),
        ]

        decision = selector.score_input(_selector_input(events))
        contributions = [item.contribution for item in decision.rationale.top_contributing_events]

        assert decision.rationale.top_contributing_events
        assert contributions == sorted(contributions, reverse=True)


class TestAffinityAndSerialization:
    def test_event_type_affinity_table_matches_contract(self) -> None:
        assert EVENT_ASSET_AFFINITY["EARNINGS"] == {"equity": 1.00, "crypto": 0.05}
        assert EVENT_ASSET_AFFINITY["PROTOCOL_UPGRADE"] == {"equity": 0.05, "crypto": 0.95}
        assert EVENT_ASSET_AFFINITY["UNKNOWN"] == {"equity": 0.50, "crypto": 0.50}
        assert canonicalize_event_type("M&A") == "M_AND_A"
        assert canonicalize_event_type("m&a") == "M_AND_A"
        assert get_event_asset_affinity("totally_new_event", "equity") == pytest.approx(0.50)

    def test_json_serialization_round_trip_is_stable(self) -> None:
        decision = MarketSelectorDecision(
            as_of_utc="2026-03-11T09:00:00+00:00",
            decision=MarketRoute.RUN_EQUITIES,
            scores=ScoreBreakdown(0.64, 0.31),
            thresholds=MarketSelectorConfig(),
            regimes=MarketRegimes(global_regime="NORMAL", equity_regime="EVENT_DRIVEN", crypto_regime="RISK_OFF"),
            veto_flags=VetoFlags(panic_veto=False, risk_off_penalty_applied=True),
            rationale=SelectorRationale(
                primary_reason="Equity catalysts are stronger.",
                secondary_reasons=["Positive earnings surprise."],
                top_contributing_events=[
                    ContributingEventSummary(
                        canonical_symbol="INFY.NS",
                        asset="equity",
                        event_type="EARNINGS",
                        contribution=0.22,
                    )
                ],
            ),
        )

        payload = decision.to_dict()
        restored = MarketSelectorDecision.from_dict(json.loads(json.dumps(payload)))

        assert list(payload.keys()) == [
            "as_of_utc",
            "decision",
            "scores",
            "thresholds",
            "regimes",
            "veto_flags",
            "rationale",
            "version",
        ]
        assert restored.to_dict() == payload
        assert restored.decision == MarketRoute.RUN_EQUITIES
        assert restored.rationale.top_contributing_events[0].canonical_symbol == "INFY.NS"

    def test_input_round_trip_uses_contract_shape(self) -> None:
        selector_input = _selector_input(
            [
                _event(canonical_symbol="INFY.NS", asset="equity", event_type="EARNINGS"),
                _event(canonical_symbol="BTC", asset="crypto", event_type="REGULATORY"),
            ],
            regimes=MarketRegimes(global_regime="NORMAL", equity_regime="EVENT_DRIVEN", crypto_regime="RISK_OFF"),
            risk_flags=MarketRiskFlags(panic_mode=False, high_cross_asset_stress=False),
            config=MarketSelectorConfig(),
        )

        payload = selector_input.to_dict()
        restored = MarketSelectorInput.from_dict(json.loads(json.dumps(payload)))
        normalized_as_of = datetime.fromisoformat(_AS_OF.replace("Z", "+00:00")).isoformat()

        assert sorted(payload.keys()) == ["as_of_utc", "config", "news_events", "regimes", "risk_flags"]
        expected_payload = dict(payload)
        expected_payload["as_of_utc"] = normalized_as_of
        assert restored.to_dict() == expected_payload
        assert restored.news_events[0].canonical_symbol == "INFY.NS"
        assert restored.records == restored.news_events
