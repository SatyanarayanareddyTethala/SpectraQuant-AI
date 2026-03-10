"""Compatibility wrapper for crypto news ingestion.

This module preserves the legacy ingestion entrypoints while delegating
fetch/normalization/sentiment/persistence responsibilities to the
``spectraquant_v3.crypto.news`` package.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any

from spectraquant_v3.core.cache import CacheManager
from spectraquant_v3.core.enums import RunMode
from spectraquant_v3.core.ingestion_result import IngestionResult, make_error_result
from spectraquant_v3.crypto.ingestion.providers.cryptopanic_provider import CryptoPanicProvider
from spectraquant_v3.crypto.news.news_fetcher import CryptoPanicAdapter, fetch_articles
from spectraquant_v3.crypto.news.news_normalizer import normalize_article_payload
from spectraquant_v3.crypto.news.news_sentiment import DeterministicSentimentScorer
from spectraquant_v3.crypto.news.news_store import NewsStore
from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper

logger = logging.getLogger(__name__)

_PROVIDER_NAME = "cryptopanic"
_ASSET_CLASS = "crypto"


def _make_success_result(
    df: Any,
    canonical_symbol: str,
    cache_path: str,
    cache_hit: bool,
) -> IngestionResult:
    if df.empty or "timestamp" not in df.columns:
        min_ts = ""
        max_ts = ""
    else:
        min_ts = str(df["timestamp"].min())
        max_ts = str(df["timestamp"].max())
    return IngestionResult(
        canonical_symbol=canonical_symbol,
        asset_class=_ASSET_CLASS,
        provider=_PROVIDER_NAME,
        success=True,
        rows_loaded=len(df),
        cache_hit=cache_hit,
        cache_path=cache_path,
        min_timestamp=min_ts,
        max_timestamp=max_ts,
    )


def ingest_news_for_symbol(
    symbol: str,
    cache: CacheManager,
    mapper: CryptoSymbolMapper,
    run_mode: RunMode,
    provider: CryptoPanicProvider,
    currencies_param: list[str] | None = None,
) -> IngestionResult:
    canonical_symbol = symbol.upper()
    cache_key = f"news__{canonical_symbol}"
    cache_path = str(cache.get_path(cache_key))

    try:
        mapper._assert_not_equity(symbol)
    except Exception as exc:  # noqa: BLE001
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            error_code="ASSET_CLASS_LEAK",
            error_message=str(exc),
        )

    if run_mode in (RunMode.TEST, RunMode.NORMAL) and cache.exists(cache_key):
        try:
            df = cache.read_parquet(cache_key)
            return _make_success_result(df, canonical_symbol, cache_path, cache_hit=True)
        except Exception:  # noqa: BLE001
            if run_mode == RunMode.TEST:
                return make_error_result(
                    canonical_symbol=canonical_symbol,
                    asset_class=_ASSET_CLASS,
                    provider=_PROVIDER_NAME,
                    error_code="CACHE_READ_ERROR",
                    error_message="Failed to read cached news in TEST mode.",
                )

    if run_mode == RunMode.TEST:
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            error_code="CACHE_MISS_TEST_MODE",
            error_message=f"TEST mode: no cached news for '{cache_key}'.",
        )

    sentiment = DeterministicSentimentScorer()
    adapter = CryptoPanicAdapter(provider=provider)

    currencies = currencies_param if currencies_param is not None else [canonical_symbol]

    def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
        score = sentiment.score(str(raw.get("title", "")))
        norm = normalize_article_payload(raw, source_name="cryptopanic", sentiment_score=score)
        if currencies:
            symbols_set = set(norm["mentioned_symbols"]) | {c.upper() for c in currencies}
            norm["mentioned_symbols"] = sorted(symbols_set)
        return norm

    try:
        articles = fetch_articles(adapter, max_pages=3, page_size=50, normalizer=_normalize)
    except Exception as exc:  # noqa: BLE001
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            error_code="PROVIDER_ERROR",
            error_message=str(exc),
        )

    if not articles:
        return IngestionResult(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            success=True,
            rows_loaded=0,
            cache_hit=False,
            cache_path=cache_path,
            min_timestamp="",
            max_timestamp="",
            warning_codes=["NO_NEWS_AVAILABLE"],
        )

    # Persist canonical records to side-car store and legacy frame to cache.
    store = NewsStore(cache.cache_dir / "news_store")
    store.write_jsonl(canonical_symbol, articles)
    store.write_parquet(canonical_symbol, articles)

    import pandas as pd

    ingested_at = datetime.datetime.now(tz=datetime.timezone.utc)
    rows = [
        {
            "timestamp": a["published_at"],
            "canonical_symbol": canonical_symbol,
            "source": a["source"],
            "title": a["title"],
            "url": a["url"],
            "tags": json.dumps(a["mentioned_symbols"]),
            "sentiment_votes": json.dumps({"score": a["sentiment_score"]}),
            "provider": _PROVIDER_NAME,
            "ingested_at": ingested_at,
        }
        for a in articles
    ]
    df = pd.DataFrame(rows)
    written = cache.write_parquet(cache_key, df)
    return _make_success_result(df, canonical_symbol, str(written), cache_hit=False)


def ingest_news_for_many(
    symbols: list[str],
    cache: CacheManager,
    mapper: CryptoSymbolMapper,
    run_mode: RunMode,
    provider: CryptoPanicProvider,
) -> dict[str, IngestionResult]:
    return {
        symbol.upper(): ingest_news_for_symbol(
            symbol=symbol,
            cache=cache,
            mapper=mapper,
            run_mode=run_mode,
            provider=provider,
        )
        for symbol in symbols
    }
