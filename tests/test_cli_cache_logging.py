from __future__ import annotations

import logging

import pandas as pd

from spectraquant.cli import main


def test_load_price_history_cache_log_is_debug(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(main, "PRICES_DIR", tmp_path)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"), "close": [1, 2, 3]})
    (tmp_path / "INFY.NS.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    with caplog.at_level(logging.INFO):
        loaded = main._load_price_history("INFY.NS")

    assert loaded is not None
    assert "Using cached yfinance data" not in caplog.text
