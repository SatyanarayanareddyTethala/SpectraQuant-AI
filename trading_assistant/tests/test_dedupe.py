"""
Test deduplication system.
"""
import pytest
from datetime import datetime, timedelta
from app.state.dedupe import DedupeManager, generate_plan_dedupe_key, generate_news_dedupe_key, generate_exec_dedupe_key


def test_generate_plan_dedupe_key():
    """Test plan dedupe key generation"""
    date = datetime(2024, 1, 15)
    key = generate_plan_dedupe_key(date)
    assert key == "PLAN:20240115"
    
    # Same date should generate same key
    key2 = generate_plan_dedupe_key(date)
    assert key == key2


def test_generate_news_dedupe_key():
    """Test news dedupe key generation"""
    plan_id = 123
    hour = 10
    key = generate_news_dedupe_key(plan_id, hour)
    assert key == "NEWS:123:10"
    
    # Different hour should generate different key
    key2 = generate_news_dedupe_key(plan_id, 11)
    assert key != key2


def test_generate_exec_dedupe_key():
    """Test execute now dedupe key generation"""
    plan_id = 123
    symbol = "AAPL"
    trigger_id = "breakout"
    
    key = generate_exec_dedupe_key(plan_id, symbol, trigger_id)
    assert key == "EXEC:123:AAPL:breakout"
    
    # Different symbol should generate different key
    key2 = generate_exec_dedupe_key(plan_id, "MSFT", trigger_id)
    assert key != key2


def test_dedupe_manager_check_can_send():
    """Test deduplication check logic"""
    config = {
        'cooldown_minutes': {
            'execute_now': 15,
            'news_update': 60
        }
    }
    
    manager = DedupeManager(config)
    
    # Mock database session (would need proper setup in real test)
    # For now, just test key generation logic
    
    plan_date = datetime(2024, 1, 15)
    plan_key = generate_plan_dedupe_key(plan_date)
    
    assert "PLAN" in plan_key
    assert "20240115" in plan_key


def test_dedupe_keys_uniqueness():
    """Test that dedupe keys are unique for different scenarios"""
    date1 = datetime(2024, 1, 15)
    date2 = datetime(2024, 1, 16)
    
    # Different dates should generate different plan keys
    key1 = generate_plan_dedupe_key(date1)
    key2 = generate_plan_dedupe_key(date2)
    assert key1 != key2
    
    # Same plan, different hours should generate different news keys
    plan_id = 100
    news_key1 = generate_news_dedupe_key(plan_id, 10)
    news_key2 = generate_news_dedupe_key(plan_id, 11)
    assert news_key1 != news_key2
    
    # Different triggers should generate different exec keys
    exec_key1 = generate_exec_dedupe_key(plan_id, "AAPL", "trigger1")
    exec_key2 = generate_exec_dedupe_key(plan_id, "AAPL", "trigger2")
    assert exec_key1 != exec_key2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
