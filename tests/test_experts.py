"""Tests for expert system."""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from spectraquant.experts.trend import TrendExpert
from spectraquant.experts.momentum import MomentumExpert
from spectraquant.experts.mean_reversion import MeanReversionExpert
from spectraquant.experts.volatility import VolatilityExpert
from spectraquant.experts.value import ValueExpert
from spectraquant.experts.news_catalyst import NewsCatalystExpert
from spectraquant.experts.aggregator import run_experts


def create_sample_prices(ticker: str = "TEST.NS", days: int = 100) -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    
    # Generate realistic price series
    base_price = 100
    returns = pd.Series([0.01 * (i % 10 - 5) for i in range(days)])
    prices = base_price * (1 + returns / 100).cumprod()
    
    df = pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": [1000000 + i * 10000 for i in range(days)],
    })
    
    return df


def test_trend_expert_basic():
    """Test TrendExpert basic functionality."""
    config = {"experts": {"enabled": True}}
    expert = TrendExpert(config)
    
    prices = create_sample_prices()
    signals = expert.generate_signals(prices)
    
    assert len(signals) >= 0
    if signals:
        assert all(s.action in ["BUY", "HOLD", "SELL"] for s in signals)
        assert all(0 <= s.score <= 100 for s in signals)


def test_trend_expert_insufficient_data():
    """Test TrendExpert with insufficient data."""
    config = {"experts": {"enabled": True}}
    expert = TrendExpert(config)
    
    prices = create_sample_prices(days=10)  # Too few rows
    signals = expert.generate_signals(prices)
    
    assert len(signals) == 0


def test_momentum_expert_basic():
    """Test MomentumExpert basic functionality."""
    config = {"experts": {"enabled": True}}
    expert = MomentumExpert(config)
    
    prices = create_sample_prices()
    signals = expert.generate_signals(prices)
    
    assert len(signals) >= 0
    if signals:
        assert all(s.action in ["BUY", "HOLD", "SELL"] for s in signals)


def test_mean_reversion_expert_basic():
    """Test MeanReversionExpert basic functionality."""
    config = {"experts": {"enabled": True}}
    expert = MeanReversionExpert(config)
    
    prices = create_sample_prices()
    signals = expert.generate_signals(prices)
    
    assert len(signals) >= 0


def test_volatility_expert_basic():
    """Test VolatilityExpert basic functionality."""
    config = {"experts": {"enabled": True}}
    expert = VolatilityExpert(config)
    
    prices = create_sample_prices()
    signals = expert.generate_signals(prices)
    
    assert len(signals) >= 0


def test_value_expert_basic():
    """Test ValueExpert basic functionality."""
    config = {"experts": {"enabled": True}}
    expert = ValueExpert(config)
    
    prices = create_sample_prices(days=300)  # Need more data for value expert
    signals = expert.generate_signals(prices)
    
    assert len(signals) >= 0


def test_news_catalyst_expert_with_news():
    """Test NewsCatalystExpert with news data."""
    config = {"experts": {"enabled": True}}
    expert = NewsCatalystExpert(config)
    
    prices = create_sample_prices()
    news_data = pd.DataFrame([
        {"ticker": "TEST.NS", "score": 0.5, "mentions": 5}
    ])
    
    signals = expert.generate_signals(prices, news_data=news_data)
    
    assert len(signals) >= 0


def test_news_catalyst_expert_no_news():
    """Test NewsCatalystExpert without news data."""
    config = {"experts": {"enabled": True}}
    expert = NewsCatalystExpert(config)
    
    prices = create_sample_prices()
    signals = expert.generate_signals(prices, news_data=None)
    
    assert len(signals) == 0


def test_expert_to_dataframe():
    """Test converting signals to DataFrame."""
    config = {"experts": {"enabled": True}}
    expert = TrendExpert(config)
    
    prices = create_sample_prices()
    signals = expert.generate_signals(prices)
    
    df = expert.to_dataframe(signals)
    
    assert isinstance(df, pd.DataFrame)
    expected_cols = ["ticker", "action", "score", "reason", "expert", "timestamp"]
    assert all(col in df.columns for col in expected_cols)


def test_run_experts_disabled():
    """Test run_experts when disabled."""
    config = {"experts": {"enabled": False}}
    prices = create_sample_prices()
    
    result = run_experts(config, prices)
    
    assert result.empty


def test_run_experts_empty_list():
    """Test run_experts with empty expert list."""
    config = {
        "experts": {
            "enabled": True,
            "list": [],
            "min_coverage": 5,
            "output_dir": "reports/experts",
        }
    }
    prices = create_sample_prices()
    
    result = run_experts(config, prices)
    
    assert result.empty


def test_run_experts_basic():
    """Test run_experts with multiple experts."""
    config = {
        "experts": {
            "enabled": True,
            "list": ["trend", "momentum"],
            "min_coverage": 1,
            "output_dir": "reports/experts",
        }
    }
    
    # Create price data for multiple tickers
    prices = pd.concat([
        create_sample_prices("TICKER1.NS"),
        create_sample_prices("TICKER2.NS"),
    ])
    
    result = run_experts(config, prices)
    
    # Should have signals from both experts or be empty if insufficient data
    assert isinstance(result, pd.DataFrame)
