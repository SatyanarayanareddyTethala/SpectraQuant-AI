from __future__ import annotations

import datetime as dt
import hashlib
import re
from typing import Any

CANONICAL_ARTICLE_FIELDS: tuple[str, ...] = (
    "article_id",
    "published_at",
    "title",
    "source",
    "url",
    "mentioned_symbols",
    "sentiment_score",
    "event_type",
    "relevance_score",
)

_SYMBOL_RE = re.compile(r"\b[A-Z]{2,10}\b")
_EVENT_KEYWORDS = {
    "hack": "security_incident",
    "exploit": "security_incident",
    "etf": "etf",
    "listing": "listing",
    "delist": "delisting",
    "upgrade": "protocol_upgrade",
    "fork": "protocol_upgrade",
    "lawsuit": "regulation",
    "sec": "regulation",
}


def _parse_timestamp(value: str | None) -> str:
    if not value:
        return dt.datetime.now(tz=dt.timezone.utc).isoformat()
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = dt.datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=dt.timezone.utc
            )
        except ValueError:
            parsed = dt.datetime.now(tz=dt.timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).isoformat()


def _extract_symbols(title: str, payload: dict[str, Any]) -> list[str]:
    from_title = set(_SYMBOL_RE.findall(title.upper()))
    from_payload: set[str] = set()
    for item in payload.get("currencies") or payload.get("mentioned_symbols") or []:
        if isinstance(item, dict) and item.get("code"):
            from_payload.add(str(item["code"]).upper())
        elif isinstance(item, str):
            from_payload.add(item.upper())
    symbols = sorted(s for s in (from_title | from_payload) if 2 <= len(s) <= 10)
    return symbols


def _derive_event_type(title: str) -> str:
    lowered = title.lower()
    for token, label in _EVENT_KEYWORDS.items():
        if token in lowered:
            return label
    return "general"


def build_article_id(payload: dict[str, Any]) -> str:
    native_id = payload.get("id") or payload.get("guid")
    if native_id:
        return str(native_id)
    digest = hashlib.sha256(f"{payload.get('url','')}|{payload.get('title','')}".encode("utf-8"))
    return digest.hexdigest()[:24]


def normalize_article_payload(
    payload: dict[str, Any],
    source_name: str,
    sentiment_score: float = 0.0,
    relevance_score: float = 0.5,
) -> dict[str, Any]:
    title = str(payload.get("title") or "").strip()
    article = {
        "article_id": build_article_id(payload),
        "published_at": _parse_timestamp(payload.get("published_at") or payload.get("pubDate")),
        "title": title,
        "source": source_name,
        "url": str(payload.get("url") or payload.get("link") or "").strip(),
        "mentioned_symbols": _extract_symbols(title, payload),
        "sentiment_score": float(sentiment_score),
        "event_type": _derive_event_type(title),
        "relevance_score": float(max(0.0, min(1.0, relevance_score))),
    }
    validate_article_schema(article)
    return article


def validate_article_schema(article: dict[str, Any]) -> None:
    missing = [field for field in CANONICAL_ARTICLE_FIELDS if field not in article]
    if missing:
        raise ValueError(f"Article missing canonical fields: {missing}")
    if not isinstance(article["mentioned_symbols"], list):
        raise ValueError("mentioned_symbols must be a list")
    if not isinstance(article["title"], str) or not article["title"].strip():
        raise ValueError("title must be a non-empty string")
    if not isinstance(article["url"], str) or not article["url"].strip():
        raise ValueError("url must be a non-empty string")
