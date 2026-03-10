"""Score the market impact of news articles.

Combines recency, source credibility, sentiment polarity, and event type
to produce a scalar impact score per article per symbol.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Source credibility tiers (0-1 scale)
_SOURCE_RANK: dict[str, float] = {
    "reuters": 0.95,
    "bloomberg": 0.95,
    "coindesk": 0.85,
    "cointelegraph": 0.80,
    "decrypt": 0.75,
    "the block": 0.80,
    "bitcoin magazine": 0.70,
}

_DEFAULT_SOURCE_RANK = 0.40


def score_article(
    article: dict[str, Any],
    now_utc: datetime | None = None,
    half_life_hours: float = 6.0,
) -> dict[str, Any]:
    """Compute impact score for a single article.

    Parameters
    ----------
    article : dict
        Must have: title, summary, published_utc, source, symbols.
        Optionally: sentiment_score (float, -1 to 1).
    now_utc : datetime, optional
        Reference time for recency calculation.
    half_life_hours : float
        Exponential decay half-life for recency weighting.

    Returns
    -------
    dict
        Article enriched with ``impact_score``, ``recency_weight``,
        ``source_credibility``, and ``sentiment_score``.
    """
    now = now_utc or datetime.now(timezone.utc)

    # Recency weight
    pub = article.get("published_utc")
    if isinstance(pub, str):
        try:
            pub = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pub = now
    if pub is None:
        pub = now
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)

    age_hours = max((now - pub).total_seconds() / 3600, 0)
    decay = math.log(2) / max(half_life_hours, 0.1)
    recency_weight = math.exp(-decay * age_hours)

    # Source credibility
    source = article.get("source", "").lower()
    credibility = _SOURCE_RANK.get(source, _DEFAULT_SOURCE_RANK)

    # Sentiment (fallback to 0 = neutral)
    sentiment = float(article.get("sentiment_score", 0.0))
    sentiment_magnitude = abs(sentiment)

    # Composite score
    impact = recency_weight * credibility * (0.3 + 0.7 * sentiment_magnitude)
    impact = round(float(np.clip(impact, 0.0, 1.0)), 4)

    article["impact_score"] = impact
    article["recency_weight"] = round(recency_weight, 4)
    article["source_credibility"] = credibility
    article["sentiment_score"] = sentiment
    return article


def build_news_features(
    articles: list[dict[str, Any]],
    now_utc: datetime | None = None,
    half_life_hours: float = 6.0,
) -> pd.DataFrame:
    """Build a features table from scored articles.

    Returns a DataFrame indexed by ``(asof_utc, symbol)`` with columns:
    ``news_impact_mean``, ``news_impact_max``, ``news_sentiment_mean``,
    ``news_article_count``, ``news_recency_best``.

    Parameters
    ----------
    articles : list of dict
        Articles with ``symbols`` list and optional ``sentiment_score``.
    now_utc : datetime, optional
        Reference time.
    half_life_hours : float
        Decay half-life for recency.

    Returns
    -------
    pd.DataFrame
        News features keyed by (asof_utc, symbol).
    """
    now = now_utc or datetime.now(timezone.utc)

    rows: list[dict[str, Any]] = []
    for art in articles:
        scored = score_article(art, now_utc=now, half_life_hours=half_life_hours)
        for sym in scored.get("symbols", []):
            rows.append({
                "asof_utc": now,
                "symbol": sym,
                "impact_score": scored["impact_score"],
                "sentiment_score": scored["sentiment_score"],
                "recency_weight": scored["recency_weight"],
                "source_credibility": scored["source_credibility"],
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "asof_utc",
                "symbol",
                "news_impact_mean",
                "news_impact_max",
                "news_sentiment_mean",
                "news_article_count",
                "news_recency_best",
            ]
        )

    df = pd.DataFrame(rows)
    grouped = df.groupby(["asof_utc", "symbol"]).agg(
        news_impact_mean=("impact_score", "mean"),
        news_impact_max=("impact_score", "max"),
        news_sentiment_mean=("sentiment_score", "mean"),
        news_article_count=("impact_score", "count"),
        news_recency_best=("recency_weight", "max"),
    ).reset_index()

    grouped["asof_utc"] = pd.to_datetime(grouped["asof_utc"], utc=True)
    return grouped
