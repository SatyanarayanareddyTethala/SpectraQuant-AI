"""Tests for database models and indexes."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from spectraquant.intelligence.db.models import (
    Base,
    PremarketPlan,
    PlanTrade,
    Alert,
    Fill,
    TradeOutcome,
    ModelRegistry,
)


def test_premarket_plan_model_has_required_fields():
    """Test PremarketPlan model has all required fields."""
    plan = PremarketPlan(
        plan_date=datetime.now(timezone.utc),
        plan_json='{"test": "data"}',
        simulation=True,
    )
    
    assert plan.plan_date is not None
    assert plan.simulation is True
    assert plan.plan_json == '{"test": "data"}'


def test_plan_trade_model_has_required_fields():
    """Test PlanTrade model has all required fields."""
    trade = PlanTrade(
        plan_id=1,
        symbol="AAPL",
        direction="long",
        entry=150.0,
        stop_loss=145.0,
        target=160.0,
        risk_per_share=5.0,
        shares=100,
    )
    
    assert trade.symbol == "AAPL"
    assert trade.direction == "long"
    assert trade.entry == 150.0
    assert trade.shares == 100


def test_alert_model_has_dedupe_key():
    """Test Alert model has dedupe_key field."""
    alert = Alert(
        category="premarket",
        dedupe_key="PLAN:2024-01-15",
        payload='{"test": "alert"}',
        status="pending",
    )
    
    assert alert.dedupe_key == "PLAN:2024-01-15"
    assert alert.category == "premarket"
    assert alert.status == "pending"


def test_alert_model_indexes():
    """Test that Alert model has required indexes."""
    # Check that __table_args__ contains the expected indexes
    indexes = {idx.name for idx in Alert.__table_args__}
    
    assert "ix_alerts_dedupe" in indexes
    assert "ix_alerts_symbol_ts" in indexes


def test_plan_trade_model_indexes():
    """Test that PlanTrade model has required indexes."""
    indexes = {idx.name for idx in PlanTrade.__table_args__}
    
    assert "ix_plan_trades_plan_symbol" in indexes


def test_fill_model_has_required_fields():
    """Test Fill model has all required fields."""
    fill = Fill(
        plan_id=1,
        symbol="AAPL",
        side="buy",
        qty=100,
        price=150.0,
        commission=20.0,
        ts=datetime.now(timezone.utc),
    )
    
    assert fill.symbol == "AAPL"
    assert fill.qty == 100
    assert fill.price == 150.0
    assert fill.side == "buy"


def test_fill_model_indexes():
    """Test that Fill model has required indexes."""
    indexes = {idx.name for idx in Fill.__table_args__}
    
    assert "ix_fills_symbol_ts" in indexes


def test_outcome_model_has_required_fields():
    """Test TradeOutcome model has all required fields."""
    outcome = TradeOutcome(
        trade_id=1,
        symbol="AAPL",
        pnl=500.0,
        mae=-200.0,
        mfe=800.0,
    )
    
    assert outcome.symbol == "AAPL"
    assert outcome.pnl == 500.0
    assert outcome.mae == -200.0
    assert outcome.mfe == 800.0


def test_outcome_model_indexes():
    """Test that TradeOutcome model has required indexes."""
    indexes = {idx.name for idx in TradeOutcome.__table_args__}
    
    assert "ix_outcomes_symbol_ts" in indexes


def test_model_registry_has_required_fields():
    """Test ModelRegistry model has all required fields."""
    model = ModelRegistry(
        model_name="test_model_v1",
        version="1.0.0",
        status="candidate",
        metrics_json='{"sharpe": 1.5}',
    )
    
    assert model.model_name == "test_model_v1"
    assert model.status == "candidate"
    assert "sharpe" in model.metrics_json


def test_all_models_inherit_from_base():
    """Test that all models inherit from Base."""
    models = [
        PremarketPlan,
        PlanTrade,
        Alert,
        Fill,
        TradeOutcome,
        ModelRegistry,
    ]
    
    for model in models:
        assert issubclass(model, Base)
