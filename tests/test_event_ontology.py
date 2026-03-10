"""Tests for the market event ontology module."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from spectraquant.news.event_ontology import (
    CorporateActionEvent,
    EarningsEvent,
    EVENT_REGISTRY,
    MacroEvent,
    MAndAEvent,
    OperationsDisruptionEvent,
    RegulatoryEvent,
    RiskEvent,
    create_event,
)

_NOW = datetime(2024, 1, 15, tzinfo=timezone.utc)
_BASE = dict(event_id="evt_001", ticker="AAPL", detected_at=_NOW)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_event_registry_contains_all_types() -> None:
    expected = {
        "earnings", "regulatory", "macro", "corporate_action",
        "operations_disruption", "risk", "m_and_a",
    }
    assert expected == set(EVENT_REGISTRY)


def test_create_event_factory_earnings() -> None:
    evt = create_event(
        "earnings",
        event_id="e1",
        ticker="AAPL",
        detected_at=_NOW,
        reported_eps=2.5,
        consensus_eps=2.0,
    )
    assert isinstance(evt, EarningsEvent)


def test_create_event_factory_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown event type"):
        create_event("unknown_type", event_id="e1", ticker="X", detected_at=_NOW)


# ---------------------------------------------------------------------------
# EarningsEvent
# ---------------------------------------------------------------------------

def test_earnings_event_beat_or_miss_derived() -> None:
    evt = EarningsEvent(
        **_BASE,
        event_type="earnings",
        reported_eps=2.5,
        consensus_eps=2.0,
    )
    assert evt.beat_or_miss == "beat"


def test_earnings_event_miss_derived() -> None:
    evt = EarningsEvent(
        **_BASE,
        event_type="earnings",
        reported_eps=1.8,
        consensus_eps=2.0,
    )
    assert evt.beat_or_miss == "miss"


def test_earnings_event_in_line_derived() -> None:
    evt = EarningsEvent(
        **_BASE,
        event_type="earnings",
        reported_eps=2.0,
        consensus_eps=2.0,
    )
    assert evt.beat_or_miss == "in-line"


def test_earnings_event_validate_valid() -> None:
    evt = EarningsEvent(
        **_BASE,
        event_type="earnings",
        reported_eps=2.5,
        consensus_eps=2.0,
    )
    assert evt.validate() == []
    assert evt.is_valid() is True


def test_earnings_event_validate_missing_required() -> None:
    evt = EarningsEvent(**_BASE, event_type="earnings")
    missing = evt.validate()
    assert "reported_eps" in missing
    assert "consensus_eps" in missing
    assert evt.is_valid() is False


# ---------------------------------------------------------------------------
# RegulatoryEvent
# ---------------------------------------------------------------------------

def test_regulatory_event_creation() -> None:
    evt = RegulatoryEvent(
        **_BASE,
        event_type="regulatory",
        regulator="SEBI",
        action_type="fine",
        fine_amount=50.0,
    )
    assert evt.is_valid() is True
    assert evt.fine_amount == 50.0


def test_regulatory_event_missing_regulator() -> None:
    evt = RegulatoryEvent(**_BASE, event_type="regulatory", action_type="fine")
    assert "regulator" in evt.validate()


# ---------------------------------------------------------------------------
# MacroEvent
# ---------------------------------------------------------------------------

def test_macro_event_surprise_direction_derived() -> None:
    evt = MacroEvent(
        **_BASE,
        event_type="macro",
        indicator="CPI",
        actual_value=6.5,
        consensus_value=6.0,
    )
    assert evt.surprise_direction == "positive_surprise"


def test_macro_event_negative_surprise() -> None:
    evt = MacroEvent(
        **_BASE,
        event_type="macro",
        indicator="GDP",
        actual_value=5.5,
        consensus_value=6.0,
    )
    assert evt.surprise_direction == "negative_surprise"


def test_macro_event_in_line() -> None:
    evt = MacroEvent(
        **_BASE,
        event_type="macro",
        indicator="REPO_RATE",
        actual_value=6.5,
        consensus_value=6.5,
    )
    assert evt.surprise_direction == "in-line"


def test_macro_event_validate_missing_actual() -> None:
    evt = MacroEvent(**_BASE, event_type="macro", indicator="CPI")
    assert "actual_value" in evt.validate()


# ---------------------------------------------------------------------------
# CorporateActionEvent
# ---------------------------------------------------------------------------

def test_corporate_action_dividend() -> None:
    evt = CorporateActionEvent(
        **_BASE,
        event_type="corporate_action",
        action_subtype="dividend",
        amount=5.0,
        ex_date=_NOW,
    )
    assert evt.is_valid() is True
    assert evt.amount == 5.0


def test_corporate_action_missing_subtype() -> None:
    evt = CorporateActionEvent(**_BASE, event_type="corporate_action")
    assert "action_subtype" in evt.validate()


# ---------------------------------------------------------------------------
# OperationsDisruptionEvent
# ---------------------------------------------------------------------------

def test_operations_disruption_high_severity() -> None:
    evt = OperationsDisruptionEvent(
        **_BASE,
        event_type="operations_disruption",
        disruption_type="plant_shutdown",
        severity="high",
        estimated_revenue_impact_pct=5.0,
    )
    assert evt.is_valid() is True
    assert evt.severity == "high"


def test_operations_disruption_missing_severity() -> None:
    evt = OperationsDisruptionEvent(
        **_BASE,
        event_type="operations_disruption",
        disruption_type="cyber_attack",
    )
    assert "severity" in evt.validate()


# ---------------------------------------------------------------------------
# RiskEvent
# ---------------------------------------------------------------------------

def test_risk_event_credit_downgrade() -> None:
    evt = RiskEvent(
        **_BASE,
        event_type="risk",
        risk_category="credit_downgrade",
        impact_severity="high",
        credit_rating_change="BBB+ → BBB",
    )
    assert evt.is_valid() is True
    assert evt.credit_rating_change == "BBB+ → BBB"


def test_risk_event_missing_category() -> None:
    evt = RiskEvent(**_BASE, event_type="risk", impact_severity="high")
    assert "risk_category" in evt.validate()


# ---------------------------------------------------------------------------
# MAndAEvent
# ---------------------------------------------------------------------------

def test_manda_event_acquisition() -> None:
    evt = MAndAEvent(
        **_BASE,
        event_type="m_and_a",
        deal_type="acquisition",
        target_ticker="MSFT",
        deal_value_millions=1000.0,
        premium_pct=25.0,
        deal_stage="announced",
        completion_probability=0.85,
    )
    assert evt.is_valid() is True
    assert evt.premium_pct == 25.0
    assert evt.completion_probability == 0.85


def test_manda_event_missing_target_ticker() -> None:
    evt = MAndAEvent(**_BASE, event_type="m_and_a", deal_type="acquisition")
    assert "target_ticker" in evt.validate()


def test_manda_event_missing_deal_type() -> None:
    evt = MAndAEvent(**_BASE, event_type="m_and_a", target_ticker="MSFT")
    assert "deal_type" in evt.validate()


# ---------------------------------------------------------------------------
# Base class shared behaviour
# ---------------------------------------------------------------------------

def test_base_event_uncertainty_flags() -> None:
    evt = EarningsEvent(
        **_BASE,
        event_type="earnings",
        reported_eps=2.5,
        consensus_eps=2.0,
        uncertainty_flags=["low_source_reliability", "conflicting_signals"],
    )
    assert len(evt.uncertainty_flags) == 2


def test_base_event_source_reliability_default() -> None:
    evt = EarningsEvent(**_BASE, event_type="earnings")
    assert evt.source_reliability == 1.0


def test_base_event_metadata() -> None:
    evt = RegulatoryEvent(
        **_BASE,
        event_type="regulatory",
        regulator="SEBI",
        action_type="fine",
        metadata={"original_headline": "SEBI fines AAPL ₹50 crore"},
    )
    assert evt.metadata["original_headline"].startswith("SEBI")
