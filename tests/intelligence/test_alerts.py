"""Tests for alert deduplication and cooldown logic."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from spectraquant.intelligence.state import DedupeManager


def test_dedupe_manager_plan_key():
    """Test plan key generation."""
    date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    key = DedupeManager.plan_key(date)
    assert key == "PLAN:20240115"
    
    date2 = datetime(2024, 1, 16, tzinfo=timezone.utc)
    key2 = DedupeManager.plan_key(date2)
    assert key2 == "PLAN:20240116"
    assert key != key2


def test_dedupe_manager_news_key():
    """Test news key generation."""
    date = datetime(2024, 1, 15, 10, tzinfo=timezone.utc)
    key = DedupeManager.news_key(123, date)
    assert key == "NEWS:123:2024011510"
    
    # Same input should generate same key
    key2 = DedupeManager.news_key(123, date)
    assert key == key2
    
    # Different input should generate different key
    date2 = datetime(2024, 1, 15, 11, tzinfo=timezone.utc)
    key3 = DedupeManager.news_key(123, date2)
    assert key != key3


def test_dedupe_manager_exec_key():
    """Test execution key generation."""
    key = DedupeManager.exec_key(123, "AAPL", "trigger1")
    assert key == "EXEC:123:AAPL:trigger1"


def test_is_duplicate_returns_true_for_existing_key():
    """Test that is_duplicate returns True when alert exists."""
    mock_session = Mock()
    mock_query = Mock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value.first.return_value = Mock()  # Alert exists
    
    result = DedupeManager.is_duplicate("PLAN:20240115", mock_session)
    
    assert result is True


def test_is_duplicate_returns_false_for_new_key():
    """Test that is_duplicate returns False when alert doesn't exist."""
    mock_session = Mock()
    mock_query = Mock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value.first.return_value = None  # No alert
    
    result = DedupeManager.is_duplicate("PLAN:20240115", mock_session)
    
    assert result is False


def test_cooldown_active_returns_true_within_cooldown():
    """Test that cooldown_active returns True when recent alert exists."""
    mock_session = Mock()
    mock_query = Mock()
    mock_session.query.return_value = mock_query
    
    # Create a recent alert (5 minutes ago)
    recent_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    mock_alert = Mock()
    mock_alert.created_at = recent_time
    mock_query.filter.return_value.first.return_value = mock_alert
    
    result = DedupeManager.cooldown_active(
        symbol="AAPL",
        category="premarket",
        session=mock_session,
        cooldown_seconds=900  # 15 minutes
    )
    
    assert result is True


def test_cooldown_active_returns_false_outside_cooldown():
    """Test that cooldown_active returns False when alert is old."""
    mock_session = Mock()
    mock_query = Mock()
    mock_session.query.return_value = mock_query
    
    # Mock needs proper chaining - filter returns another mock that has first()
    mock_filter_result = Mock()
    mock_query.filter.return_value = mock_filter_result
    mock_filter_result.first.return_value = None  # No recent alerts found
    
    result = DedupeManager.cooldown_active(
        symbol="AAPL",
        category="premarket",
        session=mock_session,
        cooldown_seconds=900  # 15 minutes
    )
    
    assert result is False


def test_cooldown_active_returns_false_when_no_alerts():
    """Test that cooldown_active returns False when no alerts exist."""
    mock_session = Mock()
    mock_query = Mock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value.first.return_value = None
    
    result = DedupeManager.cooldown_active(
        symbol="AAPL",
        category="premarket",
        session=mock_session,
        cooldown_seconds=900
    )
    
    assert result is False
