from __future__ import annotations

import logging

from spectraquant.cli.main import _log_universe_resolution


def test_universe_logging_compact_by_default(monkeypatch, caplog):
    monkeypatch.delenv("SPECTRAQUANT_VERBOSE", raising=False)
    tickers = ["INFY.NS", "TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", "SBIN.NS"]
    meta = {"source": "universe.tickers", "raw_count": 5}

    with caplog.at_level(logging.INFO):
        _log_universe_resolution(tickers, meta, context="Download")

    assert "Download universe loaded: 5 symbols" in caplog.text
    assert "Download universe preview: INFY.NS, TCS.NS, HDFCBANK.NS" in caplog.text
    assert "full universe tickers" not in caplog.text


def test_universe_logging_full_list_only_in_verbose(monkeypatch, caplog):
    monkeypatch.setenv("SPECTRAQUANT_VERBOSE", "true")
    tickers = ["INFY.NS", "TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", "SBIN.NS"]
    meta = {"source": "universe.tickers", "raw_count": 5}

    with caplog.at_level(logging.DEBUG):
        _log_universe_resolution(tickers, meta, context="Download")

    assert "Download full universe tickers: INFY.NS, TCS.NS, HDFCBANK.NS, RELIANCE.NS, SBIN.NS" in caplog.text
