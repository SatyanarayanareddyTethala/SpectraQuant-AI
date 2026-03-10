"""Tests for email notification system."""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

from spectraquant.intelligence.emailer import EmailNotifier


def test_email_notifier_send_alert_success():
    """Test successful email sending."""
    config = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username_env": "SMTP_USERNAME",
        "password_env": "SMTP_PASSWORD",
        "from_addr": "test@example.com",
        "to_addrs": ["recipient@example.com"],
    }
    
    with patch.dict("os.environ", {"SMTP_USERNAME": "test@example.com", "SMTP_PASSWORD": "password"}):
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            notifier = EmailNotifier()
            result = notifier.send_alert(
                subject="Test Alert",
                body="This is a test",
                config=config,
            )
            
            assert result is True
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once()
            mock_server.sendmail.assert_called_once()


def test_email_notifier_missing_credentials():
    """Test that email send fails gracefully when credentials are missing."""
    config = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username_env": "SMTP_USERNAME",
        "password_env": "SMTP_PASSWORD",
        "from_addr": "test@example.com",
        "to_addrs": ["recipient@example.com"],
    }
    
    with patch.dict("os.environ", {}, clear=True):
        notifier = EmailNotifier()
        result = notifier.send_alert(
            subject="Test Alert",
            body="This is a test",
            config=config,
        )
        
        assert result is False


def test_email_notifier_missing_recipients():
    """Test that email send fails gracefully when no recipients configured."""
    config = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username_env": "SMTP_USERNAME",
        "password_env": "SMTP_PASSWORD",
        "from_addr": "test@example.com",
        "to_addrs": [],  # Empty recipients
    }
    
    with patch.dict("os.environ", {"SMTP_USERNAME": "test@example.com", "SMTP_PASSWORD": "password"}):
        notifier = EmailNotifier()
        result = notifier.send_alert(
            subject="Test Alert",
            body="This is a test",
            config=config,
        )
        
        assert result is False


def test_email_notifier_connection_failure():
    """Test that email send fails gracefully on connection error."""
    config = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username_env": "SMTP_USERNAME",
        "password_env": "SMTP_PASSWORD",
        "from_addr": "test@example.com",
        "to_addrs": ["recipient@example.com"],
    }
    
    with patch.dict("os.environ", {"SMTP_USERNAME": "test@example.com", "SMTP_PASSWORD": "password"}):
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.side_effect = Exception("Connection failed")
            
            notifier = EmailNotifier()
            result = notifier.send_alert(
                subject="Test Alert",
                body="This is a test",
                config=config,
            )
            
            assert result is False


def test_email_notifier_process_pending():
    """Test processing pending alerts from database."""
    from spectraquant.intelligence.db.models import Alert
    
    # Create mock session
    mock_session = Mock()
    mock_query = Mock()
    mock_session.query.return_value = mock_query
    
    # Create mock pending alerts
    mock_alert1 = Mock(spec=Alert)
    mock_alert1.alert_id = 1
    mock_alert1.category = "premarket"
    mock_alert1.symbol = "AAPL"
    mock_alert1.payload = "Test alert 1"
    
    mock_alert2 = Mock(spec=Alert)
    mock_alert2.alert_id = 2
    mock_alert2.category = "news"
    mock_alert2.symbol = "MSFT"
    mock_alert2.payload = "Test alert 2"
    
    # Make filter() return a mock that has order_by() which returns a mock that has all()
    mock_filter = Mock()
    mock_order_by = Mock()
    mock_query.filter.return_value = mock_filter
    mock_filter.order_by.return_value = mock_order_by
    mock_order_by.all.return_value = [mock_alert1, mock_alert2]
    
    config = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username_env": "SMTP_USERNAME",
        "password_env": "SMTP_PASSWORD",
        "from_addr": "test@example.com",
        "to_addrs": ["recipient@example.com"],
    }
    
    with patch.dict("os.environ", {"SMTP_USERNAME": "test@example.com", "SMTP_PASSWORD": "password"}):
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            notifier = EmailNotifier()
            count = notifier.process_pending(mock_session, config)
            
            # Should have sent 2 emails
            assert count == 2
            mock_session.commit.assert_called()
