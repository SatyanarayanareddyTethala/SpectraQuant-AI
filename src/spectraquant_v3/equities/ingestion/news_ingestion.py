"""Equity news ingestion for SpectraQuant-AI-V3.

Fetches and normalises news headlines for canonical equity tickers from RSS
feeds (via :class:`~spectraquant_v3.equities.ingestion.providers.rss_provider.RSSProvider`)
and writes results to the parquet cache.

Equity news failures are **NON-FATAL** — a missing or empty news feed returns
an :class:`~spectraquant_v3.core.ingestion_result.IngestionResult` with
``success=False`` rather than raising.

RunMode enforcement:
- NORMAL  : cache-first; fetch on miss.
- TEST    : cache-only; returns cached data if present; returns error result
            on cache miss (network calls forbidden, but news is non-fatal).
- REFRESH : always re-fetch; overwrites existing cache entry.

This module must NEVER import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

from spectraquant_v3.core.enums import RunMode
from spectraquant_v3.core.ingestion_result import IngestionResult, make_error_result

if TYPE_CHECKING:
    from spectraquant_v3.core.cache import CacheManager
    from spectraquant_v3.equities.ingestion.providers.rss_provider import RSSProvider
    from spectraquant_v3.equities.symbols.mapper import EquitySymbolMapper

logger = logging.getLogger(__name__)

_ASSET_CLASS = "equity"
_PROVIDER_NAME = "rss"
_CACHE_KEY_PREFIX = "news__"


# ---------------------------------------------------------------------------
# Single-symbol ingestion
# ---------------------------------------------------------------------------


def ingest_news_for_symbol(
    symbol: str,
    cache: "CacheManager",
    mapper: "EquitySymbolMapper",
    run_mode: RunMode = RunMode.NORMAL,
    provider: "RSSProvider | None" = None,
    sources: list[str] | None = None,
) -> IngestionResult:
    """Ingest RSS news for a single canonical equity ticker.

    Args:
        symbol:   Canonical equity ticker, e.g. ``"INFY.NS"``.
        cache:    :class:`~spectraquant_v3.core.cache.CacheManager` wired to
                  the equity cache directory.
        mapper:   :class:`~spectraquant_v3.equities.symbols.mapper.EquitySymbolMapper`
                  for asset-class validation.
        run_mode: Cache behaviour (NORMAL / TEST / REFRESH).
        provider: Optional :class:`~spectraquant_v3.equities.ingestion.providers.rss_provider.RSSProvider`
                  instance.  When ``None`` a default instance is created.
        sources:  List of RSS feed URL templates.  ``{ticker}`` is substituted
                  with *symbol*.  Defaults to the provider's built-in feeds.

    Returns:
        :class:`~spectraquant_v3.core.ingestion_result.IngestionResult` with
        ``success=True`` when at least one item was fetched or loaded from
        cache.  Returns ``success=False`` on any failure — news is optional.
    """
    import pandas as pd  # noqa: PLC0415

    from spectraquant_v3.equities.ingestion.providers.rss_provider import (  # noqa: PLC0415
        RSSProvider,
    )

    # Asset-class guard: reject crypto symbols silently through mapper.
    try:
        mapper._assert_not_crypto(symbol)
    except Exception as exc:  # noqa: BLE001
        return make_error_result(
            canonical_symbol=symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            error_code=type(exc).__name__,
            error_message=str(exc),
        )

    if provider is None:
        provider = RSSProvider()

    cache_key = f"{_CACHE_KEY_PREFIX}{symbol}"
    cache_path = str(cache.get_path(cache_key))

    # ------------------------------------------------------------------
    # Cache probe (NORMAL/TEST; REFRESH always re-fetches)
    # ------------------------------------------------------------------
    if run_mode != RunMode.REFRESH and cache.exists(cache_key):
        try:
            cached_df: pd.DataFrame = cache.read_parquet(cache_key)
            min_ts, max_ts = _news_timestamp_range(cached_df)
            logger.debug(
                "ingest_news_for_symbol: cache hit for '%s' (%d rows)",
                symbol,
                len(cached_df),
            )
            return IngestionResult(
                canonical_symbol=symbol,
                asset_class=_ASSET_CLASS,
                provider=_PROVIDER_NAME,
                success=True,
                rows_loaded=len(cached_df),
                cache_hit=True,
                cache_path=cache_path,
                min_timestamp=min_ts,
                max_timestamp=max_ts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ingest_news_for_symbol: cache read failed for '%s': %s",
                symbol,
                exc,
            )

    # In TEST mode, do not attempt network calls for news (non-fatal).
    if run_mode == RunMode.TEST:
        logger.info(
            "ingest_news_for_symbol: TEST mode cache miss for '%s'; "
            "returning empty success=False (news is non-fatal).",
            symbol,
        )
        return make_error_result(
            canonical_symbol=symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            error_code="CacheOnlyViolation",
            error_message=(
                f"TEST mode: no cached news for '{symbol}'. "
                "Network calls are forbidden in test mode."
            ),
        )

    # ------------------------------------------------------------------
    # Fetch from provider
    # ------------------------------------------------------------------
    logger.info("ingest_news_for_symbol: fetching news for '%s'", symbol)

    try:
        raw_items: list[dict[str, str]] = provider.get_news(
            ticker=symbol, sources=sources
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ingest_news_for_symbol: provider.get_news failed for '%s': %s",
            symbol,
            exc,
        )
        return make_error_result(
            canonical_symbol=symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            error_code=type(exc).__name__,
            error_message=str(exc),
        )

    if not raw_items:
        logger.info(
            "ingest_news_for_symbol: no news items returned for '%s'", symbol
        )
        return make_error_result(
            canonical_symbol=symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            error_code="EmptyNewsResult",
            error_message=f"No news items returned for '{symbol}'.",
        )

    # ------------------------------------------------------------------
    # Normalise to DataFrame
    # ------------------------------------------------------------------
    ingested_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    rows = []
    for item in raw_items:
        rows.append(
            {
                "timestamp": _parse_published_at(item.get("published_at", "")),
                "canonical_symbol": symbol,
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "provider": item.get("provider", _PROVIDER_NAME),
                "ingested_at": ingested_at,
            }
        )

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Write to cache
    # ------------------------------------------------------------------
    try:
        cache.write_parquet(cache_key, df)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ingest_news_for_symbol: cache write failed for '%s': %s",
            symbol,
            exc,
        )
        # Still return success — the data was fetched; cache write is best-effort.

    min_ts, max_ts = _news_timestamp_range(df)
    logger.info(
        "ingest_news_for_symbol: ingested %d news items for '%s'",
        len(df),
        symbol,
    )

    return IngestionResult(
        canonical_symbol=symbol,
        asset_class=_ASSET_CLASS,
        provider=_PROVIDER_NAME,
        success=True,
        rows_loaded=len(df),
        cache_hit=False,
        cache_path=cache_path,
        min_timestamp=min_ts,
        max_timestamp=max_ts,
    )


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------


def ingest_news_for_many(
    symbols: list[str],
    cache: "CacheManager",
    mapper: "EquitySymbolMapper",
    run_mode: RunMode = RunMode.NORMAL,
    provider: "RSSProvider | None" = None,
    sources: list[str] | None = None,
) -> dict[str, IngestionResult]:
    """Ingest news for multiple canonical equity tickers.

    Args:
        symbols:  List of canonical equity tickers.
        cache:    :class:`~spectraquant_v3.core.cache.CacheManager`.
        mapper:   :class:`~spectraquant_v3.equities.symbols.mapper.EquitySymbolMapper`.
        run_mode: Cache behaviour (NORMAL / TEST / REFRESH).
        provider: Optional :class:`~spectraquant_v3.equities.ingestion.providers.rss_provider.RSSProvider`.
        sources:  Optional list of RSS feed URL templates.

    Returns:
        Dict mapping canonical symbol → :class:`~spectraquant_v3.core.ingestion_result.IngestionResult`.
        Every input symbol is represented, even on failure.  Failures are
        non-fatal and never abort the batch.
    """
    results: dict[str, IngestionResult] = {}

    for sym in symbols:
        try:
            results[sym] = ingest_news_for_symbol(
                symbol=sym,
                cache=cache,
                mapper=mapper,
                run_mode=run_mode,
                provider=provider,
                sources=sources,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ingest_news_for_many: unexpected error for '%s': %s",
                sym,
                exc,
            )
            results[sym] = make_error_result(
                canonical_symbol=sym,
                asset_class=_ASSET_CLASS,
                provider=_PROVIDER_NAME,
                error_code=type(exc).__name__,
                error_message=str(exc),
            )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_published_at(value: str) -> datetime.datetime:
    """Parse an RSS ``published_at`` string into a :class:`datetime.datetime`.

    Falls back to ``datetime.datetime.now(utc)`` when parsing fails so that
    every news row always has a valid timestamp.
    """
    if not value:
        return datetime.datetime.now(tz=datetime.timezone.utc)

    try:
        from dateutil import parser as dateutil_parser  # noqa: PLC0415

        return dateutil_parser.parse(value)
    except Exception:  # noqa: BLE001
        return datetime.datetime.now(tz=datetime.timezone.utc)


def _news_timestamp_range(df: "pd.DataFrame") -> tuple[str, str]:
    """Return (min_timestamp, max_timestamp) strings from a news DataFrame."""
    try:
        if "timestamp" in df.columns and not df.empty:
            return str(df["timestamp"].min()), str(df["timestamp"].max())
        return "", ""
    except Exception:  # noqa: BLE001
        return "", ""
