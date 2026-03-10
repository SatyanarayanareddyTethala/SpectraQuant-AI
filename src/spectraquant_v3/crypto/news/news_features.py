from __future__ import annotations

import datetime as dt
from typing import Any


def build_news_features(articles: list[dict[str, Any]], now: dt.datetime | None = None) -> list[dict[str, Any]]:
    now = now or dt.datetime.now(tz=dt.timezone.utc)
    features: list[dict[str, Any]] = []
    for article in articles:
        published = dt.datetime.fromisoformat(str(article["published_at"]).replace("Z", "+00:00"))
        age_hours = max(0.0, (now - published).total_seconds() / 3600.0)
        features.append(
            {
                "article_id": article["article_id"],
                "sentiment_score": float(article.get("sentiment_score", 0.0)),
                "relevance_score": float(article.get("relevance_score", 0.0)),
                "symbol_count": len(article.get("mentioned_symbols") or []),
                "title_length": len(article.get("title", "")),
                "age_hours": age_hours,
                "event_type": article.get("event_type", "general"),
            }
        )
    return features
