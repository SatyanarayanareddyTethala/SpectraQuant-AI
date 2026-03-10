"""Tests for the intelligence scheduler."""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

from spectraquant.intelligence.scheduler import IntelligenceScheduler


def test_scheduler_can_be_imported():
    """Test that scheduler module can be imported."""
    assert IntelligenceScheduler is not None


def test_scheduler_initialization_with_config():
    """Test that scheduler can be initialized with a config object."""
    # Create a mock config with all required attributes
    mock_config = Mock()
    mock_config.market = Mock()
    mock_config.market.timezone = "America/New_York"
    
    mock_config.scheduler = Mock()
    mock_config.scheduler.premarket_cron = "28 8 * * 1-5"
    mock_config.scheduler.hourly_news_cron = "5 * * * 1-5"
    mock_config.scheduler.intraday_interval_seconds = 60
    mock_config.scheduler.nightly_cron = "0 18 * * 1-5"
    mock_config.scheduler.weekly_cron = "0 2 * * 6"
    
    mock_config.learning = Mock()
    mock_config.learning.retrain_hour = 2  # Must be an integer for cron
    
    # Patch where BackgroundScheduler is imported in scheduler.py
    with patch("apscheduler.schedulers.background.BackgroundScheduler") as mock_bg:
        mock_scheduler_instance = MagicMock()
        mock_bg.return_value = mock_scheduler_instance
        
        scheduler = IntelligenceScheduler(config=mock_config)
        
        # Verify scheduler was created with correct timezone
        mock_bg.assert_called_once_with(timezone="America/New_York")
        
        # Verify jobs were added
        assert mock_scheduler_instance.add_job.called


def test_scheduler_has_start_method():
    """Test that scheduler has start method."""
    mock_config = Mock()
    mock_config.market = Mock()
    mock_config.market.timezone = "UTC"
    mock_config.scheduler = Mock()
    mock_config.scheduler.premarket_cron = "28 8 * * 1-5"
    mock_config.scheduler.hourly_news_cron = "5 * * * 1-5"
    mock_config.scheduler.intraday_interval_seconds = 60
    mock_config.scheduler.nightly_cron = "0 18 * * 1-5"
    mock_config.scheduler.weekly_cron = "0 2 * * 6"
    mock_config.learning = Mock()
    mock_config.learning.retrain_hour = 2
    
    with patch("apscheduler.schedulers.background.BackgroundScheduler"):
        scheduler = IntelligenceScheduler(config=mock_config)
        assert hasattr(scheduler, 'start')
        assert callable(scheduler.start)


def test_scheduler_has_stop_method():
    """Test that scheduler has stop method."""
    mock_config = Mock()
    mock_config.market = Mock()
    mock_config.market.timezone = "UTC"
    mock_config.scheduler = Mock()
    mock_config.scheduler.premarket_cron = "28 8 * * 1-5"
    mock_config.scheduler.hourly_news_cron = "5 * * * 1-5"
    mock_config.scheduler.intraday_interval_seconds = 60
    mock_config.scheduler.nightly_cron = "0 18 * * 1-5"
    mock_config.scheduler.weekly_cron = "0 2 * * 6"
    mock_config.learning = Mock()
    mock_config.learning.retrain_hour = 2
    
    with patch("apscheduler.schedulers.background.BackgroundScheduler"):
        scheduler = IntelligenceScheduler(config=mock_config)
        assert hasattr(scheduler, 'stop')
        assert callable(scheduler.stop)


def test_scheduler_has_get_jobs_method():
    """Test that scheduler has scheduler attribute for getting jobs."""
    mock_config = Mock()
    mock_config.market = Mock()
    mock_config.market.timezone = "UTC"
    mock_config.scheduler = Mock()
    mock_config.scheduler.premarket_cron = "28 8 * * 1-5"
    mock_config.scheduler.hourly_news_cron = "5 * * * 1-5"
    mock_config.scheduler.intraday_interval_seconds = 60
    mock_config.scheduler.nightly_cron = "0 18 * * 1-5"
    mock_config.scheduler.weekly_cron = "0 2 * * 6"
    mock_config.learning = Mock()
    mock_config.learning.retrain_hour = 2
    
    with patch("apscheduler.schedulers.background.BackgroundScheduler"):
        scheduler = IntelligenceScheduler(config=mock_config)
        # The scheduler should have a .scheduler attribute from APScheduler
        assert hasattr(scheduler, 'scheduler')
