from __future__ import annotations

# replaced hardcoded ticker
from pathlib import Path

import pandas as pd
import pytest

from spectraquant.core.providers.mock import MockProvider
from spectraquant.core.providers import yfinance as yf_provider
from spectraquant.qa.quality_gates import QualityGateError, run_quality_gates_price_frame
from spectraquant.sentiment.newsapi_provider import fetch_news_items


def test_mock_provider_returns_explicit_date() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            "close": [100.0, 101.0],
        }
    )
    provider = MockProvider({"TICKER1.NS": df})
    out = provider.fetch_daily("TICKER1.NS", period="5y", interval="1d")
    assert "date" in out.columns
    assert out["date"].dt.tz is not None


def test_mock_provider_duplicate_bars_fail_quality_gate() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "close": [100.0, 100.0],
        }
    )
    provider = MockProvider({"TICKER1.NS": df})
    out = provider.fetch_daily("TICKER1.NS", period="1y", interval="1d")
    cfg = {"qa": {"stale_tolerance_minutes": 10000000}}
    with pytest.raises(QualityGateError):
        run_quality_gates_price_frame(out, ticker="TICKER1.NS", exchange="NSE", interval="1d", cfg=cfg)


def test_yfinance_provider_cooldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        return pd.DataFrame({"Date": ["2024-01-01"], "Close": [100.0]})

    times = iter([1000.0, 1000.0, 1100.0])
    monkeypatch.setattr(yf_provider, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(yf_provider.yf, "download", fake_download)
    monkeypatch.setattr(yf_provider.time, "time", lambda: next(times))
    provider = yf_provider.YfinanceProvider(config={"data": {"cooldown_seconds": 500}})
    provider.fetch_daily("TICKER1.NS", period="1y", interval="1d")
    provider.fetch_daily("TICKER1.NS", period="1y", interval="1d")
    assert calls["count"] == 1


def test_yfinance_provider_force_download_ignores_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        return pd.DataFrame({"Date": ["2024-01-01"], "Close": [100.0]})

    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"downloads": {"yfinance": {"tickers": {"TICKER1.NS": {"last_fetch": 999.0}}}}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(yf_provider, "STATE_PATH", state_path)
    monkeypatch.setattr(yf_provider.yf, "download", fake_download)
    monkeypatch.setattr(yf_provider.time, "time", lambda: 1000.0)
    provider = yf_provider.YfinanceProvider(config={"data": {"cooldown_seconds": 3600, "force_download": True}})
    provider.fetch_daily("TICKER1.NS", period="1y", interval="1d")
    assert calls["count"] == 1


def test_yfinance_provider_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("rate limit")
        return pd.DataFrame({"Date": ["2024-01-01"], "Close": [100.0]})

    monkeypatch.setattr(yf_provider, "MAX_RETRIES", 2)
    monkeypatch.setattr(yf_provider, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(yf_provider.yf, "download", fake_download)
    monkeypatch.setattr(yf_provider.time, "sleep", lambda *_: None)
    provider = yf_provider.YfinanceProvider()
    out = provider.fetch_daily("TICKER1.NS", period="1y", interval="1d")
    assert not out.empty


def test_yfinance_provider_zero_cooldown_allows_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        return pd.DataFrame({"Date": ["2024-01-01"], "Close": [100.0]})

    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"downloads": {"yfinance": {"tickers": {"TICKER1.NS": {"last_fetch": 999.0}}}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(yf_provider, "STATE_PATH", state_path)
    monkeypatch.setattr(yf_provider.yf, "download", fake_download)
    monkeypatch.setattr(yf_provider.time, "time", lambda: 1000.0)
    provider = yf_provider.YfinanceProvider(config={"data": {"cooldown_seconds": 0}})
    provider.fetch_daily("TICKER1.NS", period="1y", interval="1d")
    assert calls["count"] == 1


def test_newsapi_provider_requires_key_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    config = {"sentiment": {"enabled": True, "use_news": True}}
    with pytest.raises(ValueError, match="NEWSAPI_KEY"):
        fetch_news_items("TICKER1.NS", "2024-01-01", "2024-01-02", config)
