"""Tests for meta-policy system."""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from spectraquant.meta_policy.regime import detect_regime, RegimeState
from spectraquant.meta_policy.performance_tracker import (
    compute_expert_weights,
    load_historical_performance,
)
from spectraquant.meta_policy.arbiter import (
    rule_based_selection,
    performance_weighted_blending,
    apply_risk_guardrails,
)


def test_detect_regime_missing_data():
    """Test regime detection with missing index data."""
    config = {
        "meta_policy": {
            "regime": {
                "index_ticker": "^NONEXISTENT",
                "vol_lookback": 20,
                "trend_fast": 20,
                "trend_slow": 50,
                "high_vol_threshold": 0.25,
            }
        }
    }
    
    regime = detect_regime(config, "nonexistent_dir")
    
    # Should return default regime
    assert regime.volatility == "normal"
    assert regime.trend == "neutral"


def test_compute_expert_weights_empty():
    """Test computing weights with no performance data."""
    config = {
        "experts": {
            "list": ["trend", "momentum", "value"],
        },
        "meta_policy": {
            "decay": 0.97,
            "weight_floor": 0.05,
            "weight_cap": 0.60,
            "min_trades_for_trust": 20,
        }
    }
    
    performance = pd.DataFrame(columns=["expert", "date", "trades", "win_rate", "avg_return"])
    weights = compute_expert_weights(performance, config)
    
    # Should return equal weights
    assert len(weights) == 3
    assert all(abs(w - 1/3) < 0.01 for w in weights.values())


def test_compute_expert_weights_basic():
    """Test computing weights with performance data."""
    config = {
        "meta_policy": {
            "decay": 0.97,
            "weight_floor": 0.05,
            "weight_cap": 0.60,
            "min_trades_for_trust": 20,
        }
    }
    
    # Create performance data
    now = datetime.now(timezone.utc)
    performance = pd.DataFrame([
        {
            "expert": "trend",
            "date": now - timedelta(days=1),
            "trades": 30,
            "win_rate": 0.7,
            "avg_return": 0.02,
        },
        {
            "expert": "momentum",
            "date": now - timedelta(days=1),
            "trades": 25,
            "win_rate": 0.5,
            "avg_return": 0.0,
        },
    ])
    
    weights = compute_expert_weights(performance, config)
    
    assert len(weights) == 2
    assert "trend" in weights
    assert "momentum" in weights
    # Trend should have higher weight
    assert weights["trend"] > weights["momentum"]


def test_rule_based_selection_empty():
    """Test rule-based selection with empty signals."""
    signals = pd.DataFrame(columns=["ticker", "action", "score", "expert"])
    regime = RegimeState(
        volatility="normal",
        trend="up",
        timestamp=pd.Timestamp.now(tz="UTC"),
        vol_value=0.02,
        trend_value=0.05,
    )
    config = {}
    
    result = rule_based_selection(signals, regime, config)
    assert result.empty


def test_rule_based_selection_basic():
    """Test rule-based selection."""
    signals = pd.DataFrame([
        {"ticker": "T1.NS", "action": "BUY", "score": 70, "expert": "trend", "weight": 0.3},
        {"ticker": "T1.NS", "action": "SELL", "score": 60, "expert": "volatility", "weight": 0.2},
        {"ticker": "T2.NS", "action": "BUY", "score": 75, "expert": "momentum", "weight": 0.3},
    ])
    
    # High volatility + uptrend regime favors momentum and trend
    regime = RegimeState(
        volatility="high",
        trend="up",
        timestamp=pd.Timestamp.now(tz="UTC"),
        vol_value=0.30,
        trend_value=0.05,
    )
    config = {}
    
    result = rule_based_selection(signals, regime, config)
    
    # Should filter to preferred experts for this regime
    assert not result.empty


def test_performance_weighted_blending_empty():
    """Test blending with empty signals."""
    signals = pd.DataFrame(columns=["ticker", "action", "score", "expert"])
    weights = {"trend": 0.5, "momentum": 0.5}
    config = {}
    
    result = performance_weighted_blending(signals, weights, config)
    assert result.empty


def test_performance_weighted_blending_basic():
    """Test performance-weighted blending."""
    signals = pd.DataFrame([
        {"ticker": "T1.NS", "action": "BUY", "score": 70, "expert": "trend"},
        {"ticker": "T1.NS", "action": "BUY", "score": 75, "expert": "momentum"},
        {"ticker": "T1.NS", "action": "HOLD", "score": 50, "expert": "value"},
        {"ticker": "T2.NS", "action": "SELL", "score": 65, "expert": "trend"},
    ])
    
    weights = {"trend": 0.4, "momentum": 0.4, "value": 0.2}
    config = {}
    
    result = performance_weighted_blending(signals, weights, config)
    
    assert not result.empty
    assert "ticker" in result.columns
    assert "action" in result.columns
    # T1 should be BUY (two BUY signals with high weights)
    t1_action = result[result["ticker"] == "T1.NS"]["action"].iloc[0]
    assert t1_action == "BUY"


def test_apply_risk_guardrails_empty():
    """Test risk guardrails with empty signals."""
    signals = pd.DataFrame(columns=["ticker", "action", "score"])
    config = {"meta_policy": {"risk_guardrails": {}}}
    
    result = apply_risk_guardrails(signals, config, "data/prices")
    assert result.empty


def test_apply_risk_guardrails_calibration():
    """Test calibration filter."""
    signals = pd.DataFrame([
        {"ticker": "T1.NS", "action": "BUY", "score": 80},
        {"ticker": "T2.NS", "action": "BUY", "score": 50},
        {"ticker": "T3.NS", "action": "SELL", "score": 40},
    ])
    
    config = {
        "meta_policy": {
            "risk_guardrails": {
                "min_calibration": 0.55,  # 55% threshold
            }
        }
    }
    
    result = apply_risk_guardrails(signals, config, "data/prices")
    
    # Should filter out T2 and T3
    assert len(result) == 1
    assert result["ticker"].iloc[0] == "T1.NS"


def test_load_historical_performance_no_data():
    """Test loading performance with no files."""
    config = {
        "experts": {
            "output_dir": "nonexistent_reports/experts"
        }
    }
    
    result = load_historical_performance(config, lookback_days=90)
    assert result.empty
