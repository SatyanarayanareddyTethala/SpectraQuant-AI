"""Interface definitions for market data and news providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class PriceProvider(ABC):
    """Abstract interface for market data price providers."""

    @abstractmethod
    def fetch_daily(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch daily price data for a ticker.
        
        Args:
            ticker: Stock symbol to fetch
            period: Time period (e.g., '1y', '5y')
            interval: Data interval (e.g., '1d')
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_intraday(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch intraday price data for a ticker.
        
        Args:
            ticker: Stock symbol to fetch
            period: Time period (e.g., '1d', '5d')
            interval: Data interval (e.g., '1m', '5m', '1h')
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        raise NotImplementedError

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the provider is healthy and responsive.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this provider.
        
        Returns:
            Provider name string
        """
        raise NotImplementedError


class NewsProvider(ABC):
    """Abstract interface for news data providers."""

    @abstractmethod
    def fetch_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_articles: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch news articles for a ticker within a date range.
        
        Args:
            ticker: Stock symbol or company name to search
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            max_articles: Maximum number of articles to return
            
        Returns:
            List of dicts with keys: date, text, title, source, url
        """
        raise NotImplementedError

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the provider is healthy and responsive.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this provider.
        
        Returns:
            Provider name string
        """
        raise NotImplementedError
