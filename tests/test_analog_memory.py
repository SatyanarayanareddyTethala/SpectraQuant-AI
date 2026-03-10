"""Tests for the historical analog memory module."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from spectraquant.news.analog_memory import AnalogMemory, AnalogResult, EventRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 15, tzinfo=timezone.utc)


def _make_record(
    event_id: str,
    event_type: str = "earnings_beat",
    ticker: str = "AAPL",
    sector: str = "Technology",
    event_text: str = "Apple beats earnings estimates",
    volatility_regime: float = 0.20,
    liquidity_state: float = 0.005,
    trend_regime: float = 0.5,
    observed_return_intraday: float | None = 0.02,
    observed_return_shortterm: float | None = 0.04,
    observed_return_medium: float | None = 0.06,
) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        event_type=event_type,
        ticker=ticker,
        sector=sector,
        event_text=event_text,
        volatility_regime=volatility_regime,
        liquidity_state=liquidity_state,
        trend_regime=trend_regime,
        observed_return_intraday=observed_return_intraday,
        observed_return_shortterm=observed_return_shortterm,
        observed_return_medium=observed_return_medium,
    )


# ---------------------------------------------------------------------------
# EventRecord tests
# ---------------------------------------------------------------------------

def test_event_record_creation() -> None:
    rec = _make_record("evt_001")
    assert rec.event_id == "evt_001"
    assert rec.ticker == "AAPL"
    assert rec.sector == "Technology"


def test_event_record_make_id() -> None:
    eid = EventRecord.make_id("AAPL", "earnings_beat", "2024-01-15")
    assert len(eid) == 16
    assert isinstance(eid, str)
    # Deterministic
    assert eid == EventRecord.make_id("AAPL", "earnings_beat", "2024-01-15")


def test_event_record_embedding_defaults_none() -> None:
    rec = _make_record("evt_002")
    assert rec.embedding is None


# ---------------------------------------------------------------------------
# AnalogMemory storage tests
# ---------------------------------------------------------------------------

def test_analog_memory_add_and_len() -> None:
    mem = AnalogMemory()
    assert len(mem) == 0
    mem.add(_make_record("evt_001"))
    assert len(mem) == 1
    mem.add(_make_record("evt_002", ticker="MSFT"))
    assert len(mem) == 2


def test_analog_memory_add_batch() -> None:
    mem = AnalogMemory()
    records = [_make_record(f"evt_{i:03d}") for i in range(5)]
    mem.add_batch(records)
    assert len(mem) == 5


def test_analog_memory_remove() -> None:
    mem = AnalogMemory()
    mem.add(_make_record("evt_001"))
    assert len(mem) == 1
    removed = mem.remove("evt_001")
    assert removed is True
    assert len(mem) == 0


def test_analog_memory_remove_nonexistent() -> None:
    mem = AnalogMemory()
    removed = mem.remove("nonexistent")
    assert removed is False


def test_analog_memory_overwrites_on_readd() -> None:
    mem = AnalogMemory()
    r1 = _make_record("evt_001", observed_return_intraday=0.01)
    r2 = _make_record("evt_001", observed_return_intraday=0.05)
    mem.add(r1)
    mem.add(r2)
    assert len(mem) == 1
    retrieved = list(mem._records.values())[0]
    assert retrieved.observed_return_intraday == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Similarity tests
# ---------------------------------------------------------------------------

def test_sector_similarity_same_sector() -> None:
    mem = AnalogMemory()
    q = _make_record("q", sector="Technology")
    c = _make_record("c", sector="Technology")
    assert mem._sector_similarity(q, c) == 1.0


def test_sector_similarity_different_sector() -> None:
    mem = AnalogMemory()
    q = _make_record("q", sector="Technology")
    c = _make_record("c", sector="Healthcare")
    assert mem._sector_similarity(q, c) == 0.0


def test_volatility_similarity_identical() -> None:
    mem = AnalogMemory()
    q = _make_record("q", volatility_regime=0.20)
    c = _make_record("c", volatility_regime=0.20)
    assert mem._volatility_similarity(q, c) == pytest.approx(1.0)


def test_volatility_similarity_far_apart() -> None:
    mem = AnalogMemory()
    q = _make_record("q", volatility_regime=0.10)
    c = _make_record("c", volatility_regime=0.80)
    # Large difference → very low similarity
    assert mem._volatility_similarity(q, c) < 0.01


def test_semantic_similarity_identical_text() -> None:
    mem = AnalogMemory()
    q = _make_record("q", event_text="Apple beats earnings estimates strongly")
    c = _make_record("c", event_text="Apple beats earnings estimates strongly")
    assert mem._semantic_similarity(q, c) == pytest.approx(1.0)


def test_semantic_similarity_embedding_fallback_to_jaccard() -> None:
    mem = AnalogMemory()
    q = _make_record("q", event_text="Apple beats earnings")
    c = _make_record("c", event_text="Apple misses earnings")
    # Embeddings are None → falls back to Jaccard
    sim = mem._semantic_similarity(q, c)
    assert 0.0 < sim < 1.0


def test_semantic_similarity_with_embeddings() -> None:
    mem = AnalogMemory()
    q = _make_record("q")
    c = _make_record("c")
    q.embedding = np.array([1.0, 0.0, 0.0])
    c.embedding = np.array([1.0, 0.0, 0.0])
    assert mem._semantic_similarity(q, c) == pytest.approx(1.0)


def test_semantic_similarity_orthogonal_embeddings() -> None:
    mem = AnalogMemory()
    q = _make_record("q")
    c = _make_record("c")
    q.embedding = np.array([1.0, 0.0])
    c.embedding = np.array([0.0, 1.0])
    assert mem._semantic_similarity(q, c) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------

def test_retrieve_returns_top_k() -> None:
    mem = AnalogMemory()
    mem.add_batch([_make_record(f"evt_{i:03d}", ticker=f"T{i}") for i in range(10)])
    query = _make_record("query", ticker="QUERY")
    results = mem.retrieve(query, top_k=3)
    assert len(results) <= 3


def test_retrieve_excludes_query_itself() -> None:
    mem = AnalogMemory()
    rec = _make_record("evt_001")
    mem.add(rec)
    results = mem.retrieve(rec, top_k=10)
    ids = [r.record.event_id for r in results]
    assert "evt_001" not in ids


def test_retrieve_sorted_by_similarity() -> None:
    mem = AnalogMemory()
    # Record in same sector → higher similarity
    r_same = _make_record("evt_001", sector="Technology", volatility_regime=0.20)
    # Record in different sector → lower similarity
    r_diff = _make_record("evt_002", sector="Healthcare", volatility_regime=0.50)
    mem.add_batch([r_same, r_diff])

    query = _make_record("q", sector="Technology", volatility_regime=0.20, ticker="QUERY")
    results = mem.retrieve(query, top_k=5)
    assert len(results) == 2
    # evt_001 should rank higher
    assert results[0].record.event_id == "evt_001"
    assert results[0].similarity >= results[1].similarity


def test_retrieve_min_similarity_filter() -> None:
    mem = AnalogMemory()
    r = _make_record("evt_001", sector="Healthcare")
    mem.add(r)

    query = _make_record("q", sector="Technology", ticker="QUERY")
    # Set very high min_similarity to exclude the result
    results = mem.retrieve(query, top_k=10, min_similarity=0.99)
    assert len(results) == 0


def test_retrieve_exclude_same_ticker() -> None:
    mem = AnalogMemory()
    same_ticker = _make_record("evt_001", ticker="AAPL")
    diff_ticker = _make_record("evt_002", ticker="MSFT")
    mem.add_batch([same_ticker, diff_ticker])

    query = _make_record("q", ticker="AAPL")
    results = mem.retrieve(query, top_k=10, exclude_same_ticker=True)
    tickers = [r.record.ticker for r in results]
    assert "AAPL" not in tickers


def test_retrieve_empty_memory() -> None:
    mem = AnalogMemory()
    query = _make_record("q")
    results = mem.retrieve(query)
    assert results == []


def test_analog_result_has_breakdown() -> None:
    mem = AnalogMemory()
    mem.add(_make_record("evt_001", ticker="MSFT"))
    query = _make_record("q", ticker="QUERY")
    results = mem.retrieve(query, top_k=1)
    assert len(results) == 1
    breakdown = results[0].similarity_breakdown
    assert "semantic" in breakdown
    assert "sector" in breakdown
    assert "volatility_regime" in breakdown


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

def test_calibrate_no_analogs() -> None:
    mem = AnalogMemory()
    result = mem.calibrate_prediction(0.03, analogs=[], horizon="intraday")
    assert result["calibrated_return"] == pytest.approx(0.03)
    assert result["analog_count"] == 0
    assert result["blend_weight"] == pytest.approx(0.0)


def test_calibrate_analogs_without_outcomes() -> None:
    mem = AnalogMemory()
    r = _make_record("evt_001", observed_return_intraday=None)
    mem.add(r)
    query = _make_record("q", ticker="QUERY")
    analogs = mem.retrieve(query)
    result = mem.calibrate_prediction(0.03, analogs, horizon="intraday")
    assert result["calibrated_return"] == pytest.approx(0.03)
    assert result["analog_count"] == 0


def test_calibrate_blends_toward_analog_mean() -> None:
    mem = AnalogMemory()
    # Two analogs with known outcomes
    r1 = _make_record("evt_001", ticker="MSFT", observed_return_intraday=0.10)
    r2 = _make_record("evt_002", ticker="GOOG", observed_return_intraday=0.10)
    mem.add_batch([r1, r2])

    query = _make_record("q", ticker="QUERY")
    analogs = mem.retrieve(query)
    result = mem.calibrate_prediction(0.00, analogs, horizon="intraday", blend_weight=0.5)
    # With blend_weight=0.5: calibrated = 0.5*0.0 + 0.5*0.1 = 0.05
    assert 0.0 < result["calibrated_return"] < 0.10
    assert result["analog_count"] == 2
    assert result["analog_mean"] == pytest.approx(0.10)


def test_calibrate_shortterm_horizon() -> None:
    mem = AnalogMemory()
    r = _make_record("evt_001", ticker="MSFT", observed_return_shortterm=0.05)
    mem.add(r)
    query = _make_record("q", ticker="QUERY")
    analogs = mem.retrieve(query)
    result = mem.calibrate_prediction(0.01, analogs, horizon="shortterm", blend_weight=1.0)
    assert result["calibrated_return"] == pytest.approx(0.05)


def test_calibrate_medium_horizon() -> None:
    mem = AnalogMemory()
    r = _make_record("evt_001", ticker="MSFT", observed_return_medium=0.08)
    mem.add(r)
    query = _make_record("q", ticker="QUERY")
    analogs = mem.retrieve(query)
    result = mem.calibrate_prediction(0.01, analogs, horizon="medium", blend_weight=1.0)
    assert result["calibrated_return"] == pytest.approx(0.08)


def test_calibrate_returns_std() -> None:
    mem = AnalogMemory()
    r1 = _make_record("evt_001", ticker="MSFT", observed_return_intraday=0.02)
    r2 = _make_record("evt_002", ticker="GOOG", observed_return_intraday=0.08)
    mem.add_batch([r1, r2])
    query = _make_record("q", ticker="QUERY")
    analogs = mem.retrieve(query)
    result = mem.calibrate_prediction(0.0, analogs, horizon="intraday")
    assert result["analog_std"] >= 0.0


def test_custom_weights_respected() -> None:
    weights = {
        "semantic": 1.0,
        "sector": 0.0,
        "volatility_regime": 0.0,
        "liquidity_state": 0.0,
        "trend_regime": 0.0,
    }
    mem = AnalogMemory(weights=weights)
    q = _make_record("q", event_text="earnings beat")
    c = _make_record("c", event_text="earnings beat", ticker="MSFT")
    mem.add(c)
    results = mem.retrieve(q)
    assert len(results) == 1
    # When semantic weight=1.0 and all others=0, composite == semantic_sim
    assert results[0].similarity == pytest.approx(
        results[0].similarity_breakdown["semantic"]
    )
