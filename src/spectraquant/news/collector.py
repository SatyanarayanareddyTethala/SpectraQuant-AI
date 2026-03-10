"""News article collector from RSS feeds and APIs.

Gathers articles from configurable sources, normalizes them, and returns
a list of raw article dicts ready for deduplication and scoring.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def collect_rss(feed_urls: list[str] | None = None) -> list[dict[str, Any]]:
    """Fetch articles from RSS feeds.

    Parameters
    ----------
    feed_urls : list of str, optional
        RSS feed URLs.  When *None*, uses built-in defaults for crypto news.

    Returns
    -------
    list of dict
        Raw articles with keys: title, summary, url, published_utc, source.
    """
    try:
        import feedparser  # noqa: WPS433
    except ImportError:
        logger.warning("feedparser not installed — skipping RSS collection")
        return []

    if feed_urls is None:
        feed_urls = _DEFAULT_FEEDS

    articles: list[dict[str, Any]] = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                pub = entry.get("published_parsed")
                if pub:
                    pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
                else:
                    pub_dt = datetime.now(timezone.utc)
                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "url": entry.get("link", ""),
                    "published_utc": pub_dt,
                    "source": feed.feed.get("title", url),
                })
        except Exception:
            logger.exception("Failed to fetch RSS feed: %s", url)
    logger.info("Collected %d articles from %d feeds", len(articles), len(feed_urls))
    return articles


def collect_newsapi(
    api_key: str,
    query: str = "crypto OR bitcoin OR ethereum",
    lookback_hours: int = 24,
) -> list[dict[str, Any]]:
    """Fetch articles from NewsAPI.

    Parameters
    ----------
    api_key : str
        NewsAPI API key.
    query : str
        Search query string.
    lookback_hours : int
        How far back to search.

    Returns
    -------
    list of dict
        Raw articles with standard keys.
    """
    try:
        import aiohttp  # noqa: WPS433 — validates install
    except ImportError:
        pass

    import urllib.request
    import json
    from datetime import timedelta

    from_date = (
        datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&from={from_date}&sortBy=publishedAt&apiKey={api_key}"
    )

    articles: list[dict[str, Any]] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SpectraQuant/0.5"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        for art in data.get("articles", []):
            pub_str = art.get("publishedAt", "")
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pub_dt = datetime.now(timezone.utc)
            articles.append({
                "title": art.get("title", ""),
                "summary": art.get("description", ""),
                "url": art.get("url", ""),
                "published_utc": pub_dt,
                "source": art.get("source", {}).get("name", ""),
            })
    except Exception:
        logger.exception("NewsAPI request failed")
    logger.info("Collected %d articles from NewsAPI", len(articles))
    return articles


def article_id(article: dict[str, Any]) -> str:
    """Compute a deterministic ID from title + URL."""
    raw = (article.get("title", "") + article.get("url", "")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


_DEFAULT_FEEDS: list[str] = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss/full/",
]
