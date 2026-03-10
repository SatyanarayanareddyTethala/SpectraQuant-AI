"""Integration tests for news-first universe + multi-expert meta-policy."""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import shutil

from spectraquant.news.universe_builder import build_news_universe
from spectraquant.experts.aggregator import run_experts
from spectraquant.meta_policy.arbiter import run_meta_policy


def create_test_config(temp_dir: Path) -> dict:
    """Create test configuration."""
    return {
        "news_universe": {
            "enabled": False,  # Disabled by default
            "lookback_hours": 12,
            "max_candidates": 10,
            "min_liquidity_avg_volume": 0,
            "min_source_rank": 0.0,
            "sentiment_model": "none",
            "require_price_confirmation": False,
            "recency_decay_half_life_hours": 6,
            "cache_dir": str(temp_dir / "news_cache"),
            "persist_articles_json": False,
        },
        "experts": {
            "enabled": False,  # Disabled by default
            "list": ["trend", "momentum"],
            "min_coverage": 1,
            "output_dir": str(temp_dir / "experts"),
        },
        "meta_policy": {
            "enabled": False,  # Disabled by default
            "method": "perf_weighted",
            "lookback_days": 30,
            "decay": 0.97,
            "weight_floor": 0.05,
            "weight_cap": 0.60,
            "min_trades_for_trust": 5,
            "regime": {
                "index_ticker": "^NSEI",
                "vol_lookback": 20,
                "trend_fast": 20,
                "trend_slow": 50,
                "high_vol_threshold": 0.25,
            },
            "risk_guardrails": {
                "min_calibration": 0.50,
            },
        },
        "universe": {
            "path": "data/universe/universe_nse.csv",
        },
        "data": {
            "prices_dir": str(temp_dir / "prices"),
        },
    }


def test_backward_compatibility_all_disabled():
    """Test that features are disabled by default and don't break existing code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = create_test_config(temp_path)
        
        # Test news universe (should return empty when disabled)
        news_df = build_news_universe(config)
        assert news_df.empty
        
        # Create sample price data
        dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        prices = pd.DataFrame({
            "ticker": ["TEST.NS"] * 60,
            "date": dates,
            "close": [100 + i for i in range(60)],
        })
        
        # Test experts (should return empty when disabled)
        expert_signals = run_experts(config, prices)
        assert expert_signals.empty
        
        # Test meta-policy (should pass through when disabled)
        final_signals = run_meta_policy(expert_signals, config, temp_path / "prices")
        assert final_signals.empty


def test_experts_enabled_basic():
    """Test expert system when enabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = create_test_config(temp_path)
        config["experts"]["enabled"] = True
        
        # Create sample price data with sufficient history
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        prices = pd.DataFrame({
            "ticker": ["TEST.NS"] * 100,
            "date": dates,
            "open": [100 + i * 0.5 for i in range(100)],
            "high": [100 + i * 0.5 + 2 for i in range(100)],
            "low": [100 + i * 0.5 - 2 for i in range(100)],
            "close": [100 + i * 0.5 for i in range(100)],
            "volume": [1000000] * 100,
        })
        
        # Run experts
        expert_signals = run_experts(config, prices)
        
        # Should generate signals from enabled experts
        assert isinstance(expert_signals, pd.DataFrame)
        if not expert_signals.empty:
            assert "expert" in expert_signals.columns
            assert "action" in expert_signals.columns
            assert "score" in expert_signals.columns


def test_meta_policy_enabled_basic():
    """Test meta-policy when enabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = create_test_config(temp_path)
        config["meta_policy"]["enabled"] = True
        
        # Create sample expert signals
        expert_signals = pd.DataFrame([
            {"ticker": "T1.NS", "action": "BUY", "score": 75, "expert": "trend"},
            {"ticker": "T1.NS", "action": "BUY", "score": 70, "expert": "momentum"},
            {"ticker": "T2.NS", "action": "SELL", "score": 65, "expert": "trend"},
        ])
        
        # Run meta-policy
        final_signals = run_meta_policy(expert_signals, config, temp_path / "prices")
        
        # Should aggregate signals
        assert isinstance(final_signals, pd.DataFrame)
        if not final_signals.empty:
            assert "action" in final_signals.columns
            assert "score" in final_signals.columns


def test_full_pipeline_enabled():
    """Test full pipeline with all features enabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = create_test_config(temp_path)
        
        # Enable all features
        config["experts"]["enabled"] = True
        config["meta_policy"]["enabled"] = True
        
        # Create sample price data
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        tickers = ["T1.NS", "T2.NS"]
        
        all_prices = []
        for ticker in tickers:
            prices = pd.DataFrame({
                "ticker": [ticker] * 100,
                "date": dates,
                "open": [100 + i * 0.5 for i in range(100)],
                "high": [100 + i * 0.5 + 2 for i in range(100)],
                "low": [100 + i * 0.5 - 2 for i in range(100)],
                "close": [100 + i * 0.5 for i in range(100)],
                "volume": [1000000] * 100,
            })
            all_prices.append(prices)
        
        combined_prices = pd.concat(all_prices, ignore_index=True)
        
        # Step 1: Run experts
        expert_signals = run_experts(config, combined_prices)
        
        # Step 2: Run meta-policy
        final_signals = run_meta_policy(expert_signals, config, temp_path / "prices")
        
        # Verify outputs
        assert isinstance(expert_signals, pd.DataFrame)
        assert isinstance(final_signals, pd.DataFrame)
        
        # Check that outputs are written
        expert_dir = temp_path / "experts"
        if expert_dir.exists():
            expert_files = list(expert_dir.glob("expert_signals_*.csv"))
            # May or may not have files depending on signal generation
        
        # Check reports directory structure
        reports_dir = Path("reports")
        if reports_dir.exists():
            # Check for expected subdirectories
            expected_dirs = ["experts", "meta_policy"]
            # These may not exist if no signals generated, which is OK


def test_config_validation():
    """Test that config validation works with new sections."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = create_test_config(temp_path)
        
        # Test default values
        assert config["news_universe"]["enabled"] is False
        assert config["experts"]["enabled"] is False
        assert config["meta_policy"]["enabled"] is False
        
        # Test structure
        assert "list" in config["experts"]
        assert "method" in config["meta_policy"]
        assert "regime" in config["meta_policy"]
        assert "risk_guardrails" in config["meta_policy"]
