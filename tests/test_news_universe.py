"""Tests for news universe builder."""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import datetime, timezone

from spectraquant.news.universe_builder import (
    dedupe_articles,
    load_universe_mapping,
    score_impact,
    apply_liquidity_filter,
    apply_price_confirmation,
)


def test_dedupe_articles():
    """Test article deduplication."""
    articles = [
        {
            "title": "Stock Market News",
            "source_name": "Reuters",
            "published_at_utc": "2024-01-01T10:00:00Z",
        },
        {
            "title": "Stock Market News",
            "source_name": "Reuters",
            "published_at_utc": "2024-01-01T10:00:00Z",
        },
        {
            "title": "Different News",
            "source_name": "Bloomberg",
            "published_at_utc": "2024-01-01T11:00:00Z",
        },
    ]
    
    result = dedupe_articles(articles)
    assert len(result) == 2


def test_load_universe_mapping_empty():
    """Test loading universe mapping with missing file."""
    result = load_universe_mapping("nonexistent.csv")
    assert result["tickers"] == []
    assert result["ticker_to_company"] == {}
    assert result["aliases"] == {}


def test_score_impact_empty():
    """Test score_impact with empty articles."""
    config = {"news_universe": {"enabled": True}}
    universe_mapping = {"tickers": ["TCS.NS"], "ticker_to_company": {}, "aliases": {}}
    
    result = score_impact([], universe_mapping, config)
    assert result.empty


def test_score_impact_basic():
    """Test basic score_impact functionality."""
    articles = [
        {
            "title": "Tata Consultancy Services announces new deal",
            "description": "TCS wins major contract",
            "content": "Tata Consultancy Services (TCS) has won a major contract",
            "source_name": "Reuters",
            "published_at_utc": "2024-01-01T10:00:00Z",
        }
    ]
    
    config = {
        "news_universe": {
            "enabled": True,
            "recency_decay_half_life_hours": 6,
            "min_source_rank": 0.0,
            "sentiment_model": "none",
        }
    }
    
    universe_mapping = {
        "tickers": ["TCS.NS"],
        "ticker_to_company": {"TCS.NS": "Tata Consultancy Services"},
        "aliases": {},
    }
    
    result = score_impact(articles, universe_mapping, config)
    # May be empty if matching fails, which is OK for basic test
    assert "ticker" in result.columns
    assert "score" in result.columns


def test_apply_liquidity_filter_empty():
    """Test liquidity filter with empty candidates."""
    candidates = pd.DataFrame(columns=["ticker", "score"])
    result = apply_liquidity_filter(candidates, "data/prices", 100000)
    assert result.empty


def test_apply_liquidity_filter_no_data():
    """Test liquidity filter when no price data exists."""
    candidates = pd.DataFrame([{"ticker": "TEST.NS", "score": 1.0}])
    result = apply_liquidity_filter(candidates, "nonexistent_dir", 100000)
    assert len(result) == 1  # Should keep candidates when no data available


def test_apply_price_confirmation_disabled():
    """Test price confirmation when disabled."""
    candidates = pd.DataFrame([{"ticker": "TEST.NS", "score": 1.0}])
    config = {"news_universe": {"require_price_confirmation": False}}
    
    result = apply_price_confirmation(candidates, "data/prices", config)
    assert "confirm_status" in result.columns
    assert result["confirm_status"].iloc[0] == "skipped"


def test_apply_price_confirmation_unknown():
    """Test price confirmation with no price data."""
    candidates = pd.DataFrame([{"ticker": "TEST.NS", "score": 1.0}])
    config = {
        "news_universe": {
            "require_price_confirmation": True,
            "confirmation": {
                "method": "gap_or_volume",
                "gap_abs_return_threshold": 0.015,
                "volume_z_threshold": 1.5,
                "lookback_days": 20,
            }
        }
    }
    
    result = apply_price_confirmation(candidates, "nonexistent_dir", config)
    assert "confirm_status" in result.columns
    # Should mark as unknown and keep candidate
    assert result["confirm_status"].iloc[0] == "unknown"
    assert len(result) == 1


def test_write_latest_news_universe_writes_json(tmp_path):
    """write_latest_news_universe persists news_universe_latest.json with tickers and asof_utc."""
    import json
    import os
    from spectraquant.utils.news_universe import write_latest_news_universe

    candidates = pd.DataFrame({"ticker": ["X.NS", "Y.NS"], "score": [2.0, 1.0]})
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_latest_news_universe(candidates, source_csv_path="some.csv")
    finally:
        os.chdir(old_cwd)

    p = tmp_path / "data" / "news_cache" / "news_universe_latest.json"
    assert p.exists(), "news_universe_latest.json must be written"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data["tickers"]) == {"X.NS", "Y.NS"}
    assert "asof_utc" in data
    assert data["source_candidates_csv"] == "some.csv"


def test_write_latest_news_universe_idempotent(tmp_path):
    """Calling write_latest_news_universe twice overwrites the file (deterministic)."""
    import json
    import os
    from spectraquant.utils.news_universe import write_latest_news_universe

    candidates1 = pd.DataFrame({"ticker": ["A.NS"], "score": [1.0]})
    candidates2 = pd.DataFrame({"ticker": ["B.NS", "C.NS"], "score": [2.0, 1.5]})
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_latest_news_universe(candidates1)
        write_latest_news_universe(candidates2)
    finally:
        os.chdir(old_cwd)

    p = tmp_path / "data" / "news_cache" / "news_universe_latest.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data["tickers"]) == {"B.NS", "C.NS"}, "Latest call should overwrite the file"
