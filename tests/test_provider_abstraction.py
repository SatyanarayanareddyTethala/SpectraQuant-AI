"""Unit tests for provider abstraction layer."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spectraquant.providers.interfaces import NewsProvider, PriceProvider
from spectraquant.providers.newsapi_adapter import NewsAPIProvider
from spectraquant.providers.router import MultiProviderRouter
from spectraquant.providers.yfinance_adapter import YFinancePriceProvider


class MockPriceProvider(PriceProvider):
    """Mock price provider for testing."""

    def __init__(self, name: str, healthy: bool = True, data: pd.DataFrame | None = None) -> None:
        self._name = name
        self._healthy = healthy
        self._data = data if data is not None else pd.DataFrame()
        self.daily_calls = 0
        self.intraday_calls = 0

    def fetch_daily(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        self.daily_calls += 1
        if not self._healthy:
            raise RuntimeError(f"{self._name} is unavailable")
        return self._data.copy()

    def fetch_intraday(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        self.intraday_calls += 1
        if not self._healthy:
            raise RuntimeError(f"{self._name} is unavailable")
        return self._data.copy()

    def is_healthy(self) -> bool:
        return self._healthy

    def get_name(self) -> str:
        return self._name


class MockNewsProvider(NewsProvider):
    """Mock news provider for testing."""

    def __init__(self, name: str, healthy: bool = True, articles: list[dict[str, Any]] | None = None) -> None:
        self._name = name
        self._healthy = healthy
        self._articles = articles if articles is not None else []
        self.fetch_calls = 0

    def fetch_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_articles: int = 50,
    ) -> list[dict[str, Any]]:
        self.fetch_calls += 1
        if not self._healthy:
            raise RuntimeError(f"{self._name} is unavailable")
        return list(self._articles)

    def is_healthy(self) -> bool:
        return self._healthy

    def get_name(self) -> str:
        return self._name


def test_multi_provider_router_price_fallback() -> None:
    """Test that router falls back to secondary provider when primary fails."""
    primary_data = pd.DataFrame()
    secondary_data = pd.DataFrame({
        "date": ["2024-01-01"],
        "close": [100.0],
    })

    primary = MockPriceProvider("primary", healthy=True, data=primary_data)
    secondary = MockPriceProvider("secondary", healthy=True, data=secondary_data)

    router = MultiProviderRouter(price_providers=[primary, secondary])
    result = router.fetch_daily("AAPL", "1y", "1d")

    assert primary.daily_calls == 1
    assert secondary.daily_calls == 1
    assert not result.empty
    assert result["close"].iloc[0] == 100.0


def test_multi_provider_router_price_outage_simulation() -> None:
    """Test graceful degradation when provider has outage."""
    working_data = pd.DataFrame({
        "date": ["2024-01-01"],
        "close": [150.0],
    })

    failed_provider = MockPriceProvider("failed", healthy=False)
    working_provider = MockPriceProvider("working", healthy=True, data=working_data)

    router = MultiProviderRouter(price_providers=[failed_provider, working_provider])
    result = router.fetch_daily("AAPL", "1y", "1d")

    assert failed_provider.daily_calls == 0
    assert working_provider.daily_calls == 1
    assert not result.empty
    assert result["close"].iloc[0] == 150.0


def test_multi_provider_router_all_providers_fail() -> None:
    """Test behavior when all providers fail."""
    failed1 = MockPriceProvider("failed1", healthy=False)
    failed2 = MockPriceProvider("failed2", healthy=False)

    router = MultiProviderRouter(price_providers=[failed1, failed2])
    result = router.fetch_daily("AAPL", "1y", "1d")

    assert result.empty
    stats = router.get_failure_stats()
    assert stats["price"]["failed1"] >= 1
    assert stats["price"]["failed2"] >= 1


def test_multi_provider_router_news_fallback() -> None:
    """Test that news router falls back correctly."""
    primary_articles: list[dict[str, Any]] = []
    secondary_articles = [
        {"date": "2024-01-01", "text": "News 1", "title": "Title 1", "source": "test", "url": ""},
    ]

    primary = MockNewsProvider("primary", healthy=True, articles=primary_articles)
    secondary = MockNewsProvider("secondary", healthy=True, articles=secondary_articles)

    router = MultiProviderRouter(news_providers=[primary, secondary])
    result = router.fetch_news("AAPL", "2024-01-01", "2024-01-02")

    assert primary.fetch_calls == 1
    assert secondary.fetch_calls == 1
    assert len(result) == 1
    assert result[0]["text"] == "News 1"


def test_multi_provider_router_news_outage_simulation() -> None:
    """Test news provider graceful degradation."""
    working_articles = [
        {"date": "2024-01-01", "text": "News", "title": "Title", "source": "test", "url": ""},
    ]

    failed_provider = MockNewsProvider("failed", healthy=False)
    working_provider = MockNewsProvider("working", healthy=True, articles=working_articles)

    router = MultiProviderRouter(news_providers=[failed_provider, working_provider])
    result = router.fetch_news("AAPL", "2024-01-01", "2024-01-02")

    assert failed_provider.fetch_calls == 0
    assert working_provider.fetch_calls == 1
    assert len(result) == 1


def test_multi_provider_router_reset_stats() -> None:
    """Test that failure stats can be reset."""
    failed = MockPriceProvider("failed", healthy=False)
    router = MultiProviderRouter(price_providers=[failed])

    router.fetch_daily("AAPL", "1y", "1d")
    stats = router.get_failure_stats()
    assert "failed" in stats["price"]

    router.reset_failure_stats()
    stats = router.get_failure_stats()
    assert len(stats["price"]) == 0


def test_yfinance_adapter_fetch_daily() -> None:
    """Test YFinance adapter daily fetch."""
    mock_df = pd.DataFrame({
        "date": ["2024-01-01"],
        "close": [100.0],
    })

    with patch("spectraquant.providers.yfinance_adapter.YfinanceProvider") as mock_provider:
        mock_instance = MagicMock()
        mock_instance.fetch_daily.return_value = mock_df
        mock_provider.return_value = mock_instance

        adapter = YFinancePriceProvider()
        result = adapter.fetch_daily("AAPL", "1y", "1d")

        assert not result.empty
        assert result["close"].iloc[0] == 100.0


def test_yfinance_adapter_handles_exceptions() -> None:
    """Test that YFinance adapter handles exceptions gracefully."""
    with patch("spectraquant.providers.yfinance_adapter.YfinanceProvider") as mock_provider:
        mock_instance = MagicMock()
        mock_instance.fetch_daily.side_effect = RuntimeError("API error")
        mock_provider.return_value = mock_instance

        adapter = YFinancePriceProvider()
        result = adapter.fetch_daily("AAPL", "1y", "1d")

        assert result.empty


def test_newsapi_adapter_fetch_news(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test NewsAPI adapter news fetch."""
    mock_items = [
        {"date": "2024-01-01T00:00:00Z", "text": "Test article"},
    ]

    def mock_fetch_news_items(ticker, start_date, end_date, config):
        return mock_items

    monkeypatch.setattr(
        "spectraquant.providers.newsapi_adapter.fetch_news_items",
        mock_fetch_news_items,
    )

    adapter = NewsAPIProvider()
    result = adapter.fetch_news("AAPL", "2024-01-01", "2024-01-02")

    assert len(result) == 1
    assert result[0]["text"] == "Test article"
    assert result[0]["source"] == "newsapi"


def test_newsapi_adapter_handles_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that NewsAPI adapter handles exceptions gracefully."""
    def mock_fetch_news_items(ticker, start_date, end_date, config):
        raise RuntimeError("API error")

    monkeypatch.setattr(
        "spectraquant.providers.newsapi_adapter.fetch_news_items",
        mock_fetch_news_items,
    )

    adapter = NewsAPIProvider()
    result = adapter.fetch_news("AAPL", "2024-01-01", "2024-01-02")

    assert len(result) == 0


def test_newsapi_adapter_health_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test NewsAPI health check."""
    monkeypatch.setenv("NEWSAPI_KEY", "test_key")
    
    adapter = NewsAPIProvider()
    assert adapter.is_healthy()

    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    adapter = NewsAPIProvider()
    assert not adapter.is_healthy()


def test_router_add_providers() -> None:
    """Test adding providers to router dynamically."""
    router = MultiProviderRouter()
    
    price_provider = MockPriceProvider("test_price", healthy=True)
    news_provider = MockNewsProvider("test_news", healthy=True)

    router.add_price_provider(price_provider)
    router.add_news_provider(news_provider)

    assert len(router._price_providers) == 1
    assert len(router._news_providers) == 1
