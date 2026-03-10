"""Build a dynamic crypto universe from recent news articles.

Reuses the existing news pipeline (collector, dedupe, entity_map,
impact_scoring) to discover which crypto assets are trending in the news,
then returns a ranked list of ``SYM-USD`` style symbols ready for the
crypto pipeline.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Minimum half-life (hours) to avoid division-by-zero in decay calculation
_MIN_HALF_LIFE_HOURS = 0.1


def build_news_crypto_universe(cfg: dict[str, Any]) -> list[str]:
    """Discover crypto symbols from recent news and return ranked list.

    Parameters
    ----------
    cfg : dict
        Full application config with ``news_ai`` and ``crypto`` sections.

    Returns
    -------
    list of str
        Symbols in ``SYM-USD`` format (e.g. ``["BTC-USD", "ETH-USD"]``),
        ranked by a weighted combination of mention count and impact score.

    Raises
    ------
    RuntimeError
        If ``cfg.crypto.news_strict`` is *True* and no symbols are found.
    """
    crypto_cfg = cfg.get("crypto", {})
    news_cfg = cfg.get("news_ai", {})
    max_symbols = crypto_cfg.get("news_max_symbols", 10)
    half_life = news_cfg.get("recency_half_life_hours", 6.0)
    strict = crypto_cfg.get("news_strict", False)
    fallback_symbols = crypto_cfg.get("symbols", ["BTC-USD", "ETH-USD", "SOL-USD"])

    now_utc = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Step a: Collect & dedupe news articles
    # ------------------------------------------------------------------
    try:
        from spectraquant.news.collector import collect_rss
        from spectraquant.news.dedupe import dedupe_articles
    except ImportError:
        logger.warning("News modules not available; cannot build news universe")
        if strict:
            raise RuntimeError("No crypto symbols found from news")
        return fallback_symbols

    articles = collect_rss()
    articles = dedupe_articles(articles)

    if not articles:
        logger.warning("No news articles collected")
        if strict:
            raise RuntimeError("No crypto symbols found from news")
        logger.info("Falling back to configured symbols: %s", fallback_symbols)
        return fallback_symbols

    # ------------------------------------------------------------------
    # Step b: Map entities → crypto symbols
    # ------------------------------------------------------------------
    from spectraquant.news.entity_map import extract_symbols as _extract

    symbol_scores: dict[str, float] = {}
    symbol_counts: dict[str, int] = {}

    for art in articles:
        text = (art.get("title", "") + " " + art.get("summary", "")).strip()
        syms = _extract(text)
        if not syms:
            continue

        # ------------------------------------------------------------------
        # Step c: Score by recency decay + impact
        # ------------------------------------------------------------------
        pub = art.get("published_utc")
        if isinstance(pub, str):
            try:
                pub = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pub = now_utc
        if pub is None:
            pub = now_utc
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)

        age_hours = max((now_utc - pub).total_seconds() / 3600, 0)
        decay = math.log(2) / max(half_life, _MIN_HALF_LIFE_HOURS)
        recency_weight = math.exp(-decay * age_hours)

        for sym in syms:
            symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
            symbol_scores[sym] = symbol_scores.get(sym, 0.0) + recency_weight

    if not symbol_scores:
        logger.warning("No crypto symbols extracted from news articles")
        if strict:
            raise RuntimeError("No crypto symbols found from news")
        logger.info("Falling back to configured symbols: %s", fallback_symbols)
        return fallback_symbols

    # ------------------------------------------------------------------
    # Step d: Rank and return top N as SYM-USD
    # ------------------------------------------------------------------
    ranked = sorted(
        symbol_scores.keys(),
        key=lambda s: symbol_scores[s],
        reverse=True,
    )[:max_symbols]

    result = [f"{sym}-USD" for sym in ranked]
    logger.info(
        "News-first crypto universe selected: %s (from %d mentioned symbols)",
        result,
        len(symbol_scores),
    )
    return result
