"""Hourly news fetch and risk-tag enrichment."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def hourly_news(config: Optional[Any] = None) -> Dict[str, Any]:
    """Fetch and process news for the current hour.

    Parameters
    ----------
    config : IntelligenceConfig, optional
        Loaded automatically when *None*.

    Returns
    -------
    dict
        News summary with keys: ``as_of``, ``articles``, ``risk_tags``,
        ``status``.
    """
    if config is None:
        from spectraquant.intelligence.config import load_config
        config = load_config()

    # AS-OF timestamp — never look ahead
    as_of = datetime.now(tz=timezone.utc)

    logger.info("Fetching hourly news as-of %s", as_of.isoformat())

    news_cfg = config.news
    articles: List[Dict[str, Any]] = []
    risk_tags: List[str] = []

    # Simulation / stub path — real providers plugged in via news_cfg.provider
    if news_cfg.provider == "rss":
        logger.debug("RSS provider selected — returning stub articles")
        articles.append(
            {
                "headline": "Market steady ahead of Fed decision",
                "source": "stub",
                "published_at": as_of.isoformat(),
                "risk_score": 0.3,
            }
        )
    else:
        logger.info("Provider '%s' — no articles fetched (stub)", news_cfg.provider)

    # Tag extraction
    for article in articles:
        score = article.get("risk_score", 0.0)
        if score >= news_cfg.risk_score_threshold:
            risk_tags.append(article.get("headline", "unknown"))

    result: Dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "hour": as_of.strftime("%Y-%m-%d %H:00"),
        "article_count": len(articles),
        "articles": articles,
        "risk_tags": risk_tags,
        "status": "ok",
    }

    logger.info("Hourly news: %d articles, %d risk tags", len(articles), len(risk_tags))
    return result
