"""Article deduplication via content hashing and similarity."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def dedupe_articles(
    articles: list[dict[str, Any]],
    sim_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Remove near-duplicate articles.

    Uses exact URL matching first, then title-based Jaccard similarity
    as a lightweight fallback when embedding models are unavailable.

    Parameters
    ----------
    articles : list of dict
        Raw articles (must have ``title`` and ``url`` keys).
    sim_threshold : float
        Jaccard similarity above which two articles are considered duplicates.

    Returns
    -------
    list of dict
        Deduplicated articles.
    """
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    result: list[dict[str, Any]] = []

    for art in articles:
        url = art.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        title_hash = _title_hash(art.get("title", ""))
        if title_hash in seen_hashes:
            continue

        # Jaccard check against existing
        is_dup = False
        title_tokens = _tokenize(art.get("title", ""))
        for existing in result:
            existing_tokens = _tokenize(existing.get("title", ""))
            if _jaccard(title_tokens, existing_tokens) >= sim_threshold:
                is_dup = True
                break

        if not is_dup:
            seen_hashes.add(title_hash)
            result.append(art)

    dropped = len(articles) - len(result)
    if dropped:
        logger.info("Deduplicated %d → %d articles (dropped %d)", len(articles), len(result), dropped)
    return result


def _title_hash(title: str) -> str:
    """Normalize and hash a title for exact-match dedup."""
    normalized = title.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _tokenize(text: str) -> set[str]:
    """Simple whitespace tokenizer with lowercasing."""
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
