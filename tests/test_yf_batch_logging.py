from __future__ import annotations

import logging

import pandas as pd

from spectraquant.data import retention, yf_batch


def _sample_price_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC", name="date")
    return pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1],
            "close": [1.1, 1.2, 1.3],
            "volume": [100, 110, 120],
        },
        index=idx,
    )


def test_retention_per_symbol_log_suppressed_at_info(caplog):
    df = _sample_price_frame()

    with caplog.at_level(logging.INFO):
        retention.prune_dataframe_to_last_n_years(df, years=5)

    assert "Retention (dataframe): kept=" not in caplog.text


def test_retention_per_symbol_log_emitted_at_debug(caplog):
    df = _sample_price_frame()

    with caplog.at_level(logging.DEBUG):
        retention.prune_dataframe_to_last_n_years(df, years=5)

    assert "Retention (dataframe): kept=" in caplog.text


def test_batch_saved_rows_log_suppressed_at_info_and_summary_present(monkeypatch, caplog):
    class DummyProvider:
        def __init__(self, config=None):
            self.config = config

        def fetch_daily(self, ticker, period="5y", interval="1d"):
            return _sample_price_frame()

    monkeypatch.setattr(yf_batch, "get_provider", lambda _name: DummyProvider)
    monkeypatch.setattr(yf_batch, "_safe_write_price", lambda ticker, df: None)
    monkeypatch.setattr(yf_batch.time, "sleep", lambda _seconds: None)

    with caplog.at_level(logging.INFO):
        yf_batch.fetch_history_batched(["AAA", "BBB"], batch_size=2, sleep_seconds=0)

    assert "Saved 3 rows for AAA after retention." not in caplog.text
    assert "Saved 3 rows for BBB after retention." not in caplog.text
    assert "Download complete: 2 symbols processed (6 rows written)" in caplog.text


def test_batch_saved_rows_log_emitted_at_debug(monkeypatch, caplog):
    class DummyProvider:
        def __init__(self, config=None):
            self.config = config

        def fetch_daily(self, ticker, period="5y", interval="1d"):
            return _sample_price_frame()

    monkeypatch.setattr(yf_batch, "get_provider", lambda _name: DummyProvider)
    monkeypatch.setattr(yf_batch, "_safe_write_price", lambda ticker, df: None)
    monkeypatch.setattr(yf_batch.time, "sleep", lambda _seconds: None)

    with caplog.at_level(logging.DEBUG):
        yf_batch.fetch_history_batched(["AAA"], batch_size=1, sleep_seconds=0)

    assert "Saved 3 rows for AAA after retention." in caplog.text
