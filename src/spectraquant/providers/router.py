"""Multi-provider fallback router with graceful degradation."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from spectraquant.providers.interfaces import NewsProvider, PriceProvider

logger = logging.getLogger(__name__)


class MultiProviderRouter:
    """Router that provides fallback across multiple providers with graceful degradation."""

    def __init__(
        self,
        price_providers: list[PriceProvider] | None = None,
        news_providers: list[NewsProvider] | None = None,
    ) -> None:
        """Initialize the multi-provider router.
        
        Args:
            price_providers: List of price providers in priority order
            news_providers: List of news providers in priority order
        """
        self._price_providers = price_providers or []
        self._news_providers = news_providers or []
        self._price_failures: dict[str, int] = {}
        self._news_failures: dict[str, int] = {}

    def add_price_provider(self, provider: PriceProvider) -> None:
        """Add a price provider to the router.
        
        Args:
            provider: Price provider instance to add
        """
        self._price_providers.append(provider)

    def add_news_provider(self, provider: NewsProvider) -> None:
        """Add a news provider to the router.
        
        Args:
            provider: News provider instance to add
        """
        self._news_providers.append(provider)

    def fetch_daily(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch daily price data with fallback across providers.
        
        Args:
            ticker: Stock symbol to fetch
            period: Time period (e.g., '1y', '5y')
            interval: Data interval (e.g., '1d')
            
        Returns:
            DataFrame with price data, or empty DataFrame if all providers fail
        """
        for provider in self._price_providers:
            provider_name = provider.get_name()
            
            if not provider.is_healthy():
                logger.warning(
                    "Provider %s is unhealthy, skipping for ticker %s",
                    provider_name,
                    ticker,
                )
                self._price_failures[provider_name] = (
                    self._price_failures.get(provider_name, 0) + 1
                )
                continue

            try:
                df = provider.fetch_daily(ticker, period, interval)
                if df is not None and not df.empty:
                    logger.info(
                        "Successfully fetched daily data for %s from %s",
                        ticker,
                        provider_name,
                    )
                    return df
                
                logger.warning(
                    "Provider %s returned empty data for %s",
                    provider_name,
                    ticker,
                )
            except Exception as exc:
                logger.error(
                    "Provider %s failed for %s: %s",
                    provider_name,
                    ticker,
                    exc,
                )
                self._price_failures[provider_name] = (
                    self._price_failures.get(provider_name, 0) + 1
                )

        logger.error("All price providers failed for ticker %s", ticker)
        return pd.DataFrame()

    def fetch_intraday(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch intraday price data with fallback across providers.
        
        Args:
            ticker: Stock symbol to fetch
            period: Time period (e.g., '1d', '5d')
            interval: Data interval (e.g., '1m', '5m', '1h')
            
        Returns:
            DataFrame with price data, or empty DataFrame if all providers fail
        """
        for provider in self._price_providers:
            provider_name = provider.get_name()
            
            if not provider.is_healthy():
                logger.warning(
                    "Provider %s is unhealthy, skipping for ticker %s",
                    provider_name,
                    ticker,
                )
                self._price_failures[provider_name] = (
                    self._price_failures.get(provider_name, 0) + 1
                )
                continue

            try:
                df = provider.fetch_intraday(ticker, period, interval)
                if df is not None and not df.empty:
                    logger.info(
                        "Successfully fetched intraday data for %s from %s",
                        ticker,
                        provider_name,
                    )
                    return df
                
                logger.warning(
                    "Provider %s returned empty data for %s",
                    provider_name,
                    ticker,
                )
            except Exception as exc:
                logger.error(
                    "Provider %s failed for %s: %s",
                    provider_name,
                    ticker,
                    exc,
                )
                self._price_failures[provider_name] = (
                    self._price_failures.get(provider_name, 0) + 1
                )

        logger.error("All price providers failed for ticker %s", ticker)
        return pd.DataFrame()

    def fetch_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_articles: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch news articles with fallback across providers.
        
        Args:
            ticker: Stock symbol or company name to search
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            max_articles: Maximum number of articles to return
            
        Returns:
            List of news articles, or empty list if all providers fail
        """
        for provider in self._news_providers:
            provider_name = provider.get_name()
            
            if not provider.is_healthy():
                logger.warning(
                    "Provider %s is unhealthy, skipping for ticker %s",
                    provider_name,
                    ticker,
                )
                self._news_failures[provider_name] = (
                    self._news_failures.get(provider_name, 0) + 1
                )
                continue

            try:
                articles = provider.fetch_news(ticker, start_date, end_date, max_articles)
                if articles:
                    logger.info(
                        "Successfully fetched %d articles for %s from %s",
                        len(articles),
                        ticker,
                        provider_name,
                    )
                    return articles
                
                logger.warning(
                    "Provider %s returned no articles for %s",
                    provider_name,
                    ticker,
                )
            except Exception as exc:
                logger.error(
                    "Provider %s failed for %s: %s",
                    provider_name,
                    ticker,
                    exc,
                )
                self._news_failures[provider_name] = (
                    self._news_failures.get(provider_name, 0) + 1
                )

        logger.error("All news providers failed for ticker %s", ticker)
        return []

    def get_failure_stats(self) -> dict[str, dict[str, int]]:
        """Get failure statistics for all providers.
        
        Returns:
            Dictionary with failure counts per provider type
        """
        return {
            "price": dict(self._price_failures),
            "news": dict(self._news_failures),
        }

    def reset_failure_stats(self) -> None:
        """Reset failure statistics."""
        self._price_failures.clear()
        self._news_failures.clear()
