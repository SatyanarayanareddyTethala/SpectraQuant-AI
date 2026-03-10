"""Canonical news article schema for SpectraQuant.

All news providers must normalise their output into this schema before the
article is processed by any downstream pipeline stage.

Canonical fields (reference: NewsAPI article shape):
    title           – headline text
    description     – short summary
    content         – full article body
    source_name     – provider/publication name
    published_at_utc – ISO-8601 UTC timestamp string
    url             – article URL (optional)

Enriched fields (populated by event_classifier / entity_linker):
    event_type      – ontology event type string (e.g. "earnings")
    entities        – list of entity strings extracted from the article
    competitors     – list of competitor ticker strings
    magnitude_score – estimated impact magnitude [0.0, 1.0]
    source_rank     – credibility score for the source [0.0, 1.0]
"""
from __future__ import annotations

import unicodedata
from typing import List, TypedDict


class CanonicalArticle(TypedDict):
    """Canonical internal representation of a news article.

    All fields are strings; missing / empty values use the documented
    sentinel rather than None so downstream code can always ``str()``
    without a None-check.
    """

    title: str
    description: str
    content: str
    source_name: str
    published_at_utc: str
    url: str


class EnrichedArticle(CanonicalArticle, total=False):
    """Canonical article extended with NLP-derived enrichment fields.

    These fields are populated downstream by
    :mod:`spectraquant.news.event_classifier` and
    :mod:`spectraquant.news.entity_linker`.  They are **optional** so that
    all existing code using :class:`CanonicalArticle` continues to work
    without modification.
    """

    event_type: str
    entities: List[str]
    competitors: List[str]
    magnitude_score: float
    source_rank: float


def normalize_article(item: dict) -> CanonicalArticle:
    """Normalise a raw provider item into :class:`CanonicalArticle`.

    The function is a *pure* mapping with explicit fallbacks so that it is
    easy to unit-test and does not raise on missing keys.

    Provider schema variants handled:

    * NewsAPI native – has ``title``, ``description``, ``content``,
      ``publishedAt``, ``source.name``, ``url``
    * Legacy text-only – has ``date`` and ``text`` (from
      ``spectraquant.sentiment.newsapi_provider``)
    * Mix – any combination of the above fields
    """
    # --- text fallback chain ------------------------------------------------
    text = str(item.get("text") or "").strip()
    raw_title = str(item.get("title") or "").strip()
    raw_desc = str(item.get("description") or "").strip()
    raw_content = str(item.get("content") or "").strip()

    # title: prefer explicit field; else derive from text; else sentinel
    if raw_title:
        title = raw_title[:140]
    elif text:
        title = text.split("\n")[0][:120]
    else:
        title = "untitled"

    # description
    description = raw_desc or text or ""

    # content
    content = raw_content or text or ""

    # published_at_utc: prefer explicit UTC field; else legacy "date"; else ""
    published_at_utc = (
        str(item.get("publishedAt") or item.get("published_at_utc") or item.get("date") or "").strip()
    )

    # source_name: direct field or nested source dict (NewsAPI native)
    source_obj = item.get("source")
    if isinstance(source_obj, dict):
        raw_source = str(source_obj.get("name") or "").strip()
    else:
        raw_source = ""
    source_name = (
        str(item.get("source_name") or "").strip()
        or raw_source
        or "unknown"
    )

    url = str(item.get("url") or "").strip()

    return CanonicalArticle(
        title=_canon_str(title),
        description=_canon_str(description),
        content=_canon_str(content),
        source_name=_canon_str(source_name),
        published_at_utc=published_at_utc,
        url=url,
    )


def _canon_str(s: str) -> str:
    """Canonicalise a string: strip whitespace and normalise Unicode NFC."""
    return unicodedata.normalize("NFC", s).strip()


def dedupe_key(article: CanonicalArticle) -> str:
    """Return a stable deduplication key for a canonical article.

    The key is based on title + published_at_utc + source_name + url so
    that exact duplicates (same story from same source at same time) are
    collapsed while different stories are preserved.

    Strings are lowercased and stripped before hashing so minor formatting
    differences do not produce false duplicates.
    """
    import hashlib

    title_norm = article["title"].lower().strip()
    ts_norm = article["published_at_utc"][:19] if article["published_at_utc"] else ""
    source_norm = article["source_name"].lower().strip()
    url_norm = article["url"].strip()
    raw = f"{title_norm}|{ts_norm}|{source_norm}|{url_norm}"
    return hashlib.sha1(raw.encode()).hexdigest()  # noqa: S324 – not used for security
