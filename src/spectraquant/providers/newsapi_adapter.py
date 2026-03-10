"""NewsAPI provider adapter."""
from __future__ import annotations

import logging
from typing import Any

from spectraquant.providers.interfaces import NewsProvider
from spectraquant.sentiment.newsapi_provider import fetch_news_items

logger = logging.getLogger(__name__)


class NewsAPIProvider(NewsProvider):
    """Adapter for NewsAPI data provider."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize NewsAPI provider adapter.
        
        Args:
            config: Configuration dictionary for the provider
        """
        self._config = config or {}
        self._name = "newsapi"

    def fetch_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_articles: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch news articles for a ticker within a date range."""
        try:
            config = dict(self._config)
            if "sentiment" not in config:
                config["sentiment"] = {}
            config["sentiment"]["max_articles_per_ticker"] = max_articles
            
            items = fetch_news_items(ticker, start_date, end_date, config)
            
            result = []
            for item in items:
                result.append({
                    "date": item.get("date", ""),
                    "text": item.get("text", ""),
                    "title": item.get("text", "").split(".")[0] if item.get("text") else "",
                    "source": "newsapi",
                    "url": "",
                })
            
            return result
        except Exception as exc:
            logger.error("NewsAPI fetch failed for %s: %s", ticker, exc)
            return []

    def is_healthy(self) -> bool:
        """Check if NewsAPI provider is healthy."""
        import os
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        
        api_key = os.getenv("NEWSAPI_KEY")
        return api_key is not None and len(api_key) > 0

    def get_name(self) -> str:
        """Get the name of this provider."""
        return self._name
