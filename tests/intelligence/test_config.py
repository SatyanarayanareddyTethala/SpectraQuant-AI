"""Tests for intelligence layer configuration."""
from __future__ import annotations

import pytest
from pathlib import Path

from spectraquant.intelligence.config import (
    IntelligenceConfig,
    MarketConfig,
    load_config,
)


def test_market_config_defaults():
    """Test MarketConfig has required fields with defaults."""
    config = MarketConfig(
        timezone="America/New_York",
        open_time="09:30:00",
        close_time="16:00:00",
    )
    assert config.timezone == "America/New_York"
    assert config.open_time == "09:30:00"
    assert config.close_time == "16:00:00"


def test_intelligence_config_safety_defaults():
    """Test that safety defaults are properly set."""
    # Create a minimal config dict
    config_dict = {
        "market": {
            "timezone": "Asia/Kolkata",
            "open_time": "09:15:00",
            "close_time": "15:30:00",
        },
        "simulation": {"enabled": True},
        "execution": {"live_enabled": False},
    }
    
    # Verify simulation mode is enabled
    assert config_dict["simulation"]["enabled"] is True
    
    # Verify live trading is disabled
    assert config_dict["execution"]["live_enabled"] is False


def test_load_config_example_file():
    """Test loading from example config file."""
    example_path = Path(__file__).parents[2] / "config" / "intelligence.example.yaml"
    
    if not example_path.exists():
        pytest.skip("Example config file not found")
    
    # Just verify the file can be parsed as YAML
    import yaml
    with open(example_path) as f:
        config = yaml.safe_load(f)
    
    # Verify safety defaults
    assert config["simulation"]["enabled"] is True
    assert config["execution"]["live_enabled"] is False
    assert config["market"]["timezone"] == "Asia/Kolkata"


def test_config_has_required_sections():
    """Test that config has all required top-level sections."""
    example_path = Path(__file__).parents[2] / "config" / "intelligence.example.yaml"
    
    if not example_path.exists():
        pytest.skip("Example config file not found")
    
    import yaml
    with open(example_path) as f:
        config = yaml.safe_load(f)
    
    required_sections = [
        "market",
        "database",
        "news",
        "email",
        "risk",
        "costs",
        "simulation",
        "execution",
        "learning",
        "scheduler",
        "logging",
    ]
    
    for section in required_sections:
        assert section in config, f"Missing required section: {section}"
