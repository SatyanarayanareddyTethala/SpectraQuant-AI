"""RSS news feed provider for SpectraQuant-AI-V3 equity ingestion.

Fetches news headlines from Yahoo Finance and Google Finance RSS feeds and
normalises each item into a flat dict.  News is treated as optional — any
failure (import error, network error, malformed feed) silently returns an
empty list rather than raising.

This module must NEVER import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default RSS feed URL templates.  The ticker is interpolated at call time.
_DEFAULT_FEED_TEMPLATES: tuple[str, ...] = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://www.google.com/finance/news?q={ticker}&output=rss",
)


class RSSProvider:
    """Fetches RSS news items for equity tickers.

    Args:
        _feedparser_module: Optional feedparser module injected for testing.
            When ``None`` (the default) the real ``feedparser`` package is
            imported lazily on first use.  If the import fails entirely, all
            calls return empty lists without raising.
    """

    def __init__(self, _feedparser_module: Any = None) -> None:
        self._feedparser = _feedparser_module

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_feedparser(self) -> Any | None:
        """Return the feedparser module, or ``None`` if unavailable."""
        if self._feedparser is not None:
            return self._feedparser
        try:
            import feedparser  # noqa: PLC0415
            return feedparser
        except ImportError:
            logger.warning(
                "RSSProvider: feedparser is not installed; news fetching is disabled. "
                "Install it with: pip install feedparser"
            )
            return None

    @staticmethod
    def _normalize_entry(entry: Any, feed_url: str) -> dict[str, str]:
        """Convert a single feedparser entry into a normalised dict."""
        published_at: str = (
            getattr(entry, "published", None)
            or getattr(entry, "updated", None)
            or ""
        )
        return {
            "title": getattr(entry, "title", "") or "",
            "url": getattr(entry, "link", "") or "",
            "published_at": published_at,
            "source": feed_url,
            "provider": "rss",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_news(
        self,
        ticker: str,
        sources: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Fetch and normalise news items for *ticker* from RSS feeds.

        Args:
            ticker:  Canonical equity ticker, e.g. ``"INFY.NS"``.
            sources: List of RSS feed URLs.  ``{ticker}`` in each URL is
                     replaced with the value of *ticker*.  Defaults to Yahoo
                     Finance and Google Finance RSS feeds.

        Returns:
            List of normalised news item dicts, each with keys:
            ``title``, ``url``, ``published_at``, ``source``, ``provider``.
            Returns an empty list on any error so news failures are non-fatal.
        """
        feedparser = self._get_feedparser()
        if feedparser is None:
            return []

        feed_urls: list[str] = [
            url.format(ticker=ticker)
            for url in (sources if sources is not None else _DEFAULT_FEED_TEMPLATES)
        ]

        items: list[dict[str, str]] = []
        for feed_url in feed_urls:
            try:
                feed = feedparser.parse(feed_url)
                for entry in getattr(feed, "entries", []):
                    items.append(self._normalize_entry(entry, feed_url))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RSSProvider: failed to fetch/parse feed '%s' for '%s': %s",
                    feed_url,
                    ticker,
                    exc,
                )

        logger.debug(
            "RSSProvider: fetched %d news items for '%s' across %d feeds",
            len(items),
            ticker,
            len(feed_urls),
        )
        return items
