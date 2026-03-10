"""Crypto OHLCV price downloader for SpectraQuant-AI-V3.

Orchestrates cache-first fetching of OHLCV candlestick data for canonical
crypto symbols.  Primary provider is CCXT; CryptoCompare and CoinGecko are
used as ordered fallbacks when CCXT fails.

Run-mode semantics
------------------
NORMAL  – check cache first; re-download only when coverage is insufficient.
TEST    – cache-only; raises :exc:`CacheOnlyViolationError` on any cache miss
          or insufficient coverage.  No network calls permitted.
REFRESH – always re-download; overwrites the cache.

Empty OHLCV is **never** accepted as a success.  :exc:`EmptyPriceDataError`
is raised when all providers return nothing.

Async support
-------------
:func:`async_download_symbol_ohlcv` and :func:`async_download_many_ohlcv`
provide asyncio-based equivalents that delegate to the synchronous
:func:`download_symbol_ohlcv` via :func:`asyncio.get_running_loop().run_in_executor`
so that I/O-bound provider calls do not block the event loop.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any

from spectraquant_v3.core.cache import CacheManager
from spectraquant_v3.core.enums import RunMode
from spectraquant_v3.core.errors import (
    AssetClassLeakError,
    CacheOnlyViolationError,
    EmptyPriceDataError,
    SpectraQuantError,
    SymbolResolutionError,
)
from spectraquant_v3.core.ingestion_result import IngestionResult, make_error_result
from spectraquant_v3.core.schema import validate_ohlcv_dataframe
from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider
from spectraquant_v3.crypto.ingestion.providers.coingecko_provider import CoinGeckoProvider
from spectraquant_v3.crypto.ingestion.providers.cryptocompare_provider import CryptoCompareProvider
from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal normalizers
# ---------------------------------------------------------------------------


def _normalize_ccxt_ohlcv(
    raw: list[list],
    canonical_symbol: str,
    exchange_id: str,
    timeframe: str,
    ingested_at: datetime.datetime,
) -> Any:
    """Convert a raw CCXT OHLCV response to a normalized DataFrame."""
    import pandas as pd

    rows = []
    for candle in raw:
        rows.append(
            {
                "timestamp": datetime.datetime.fromtimestamp(int(candle[0]) / 1000, tz=datetime.timezone.utc),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "canonical_symbol": canonical_symbol,
                "provider": f"ccxt/{exchange_id}",
                "exchange_id": exchange_id,
                "timeframe": timeframe,
                "ingested_at": ingested_at,
            }
        )
    return pd.DataFrame(rows)


def _normalize_cryptocompare_ohlcv(
    raw: list[dict],
    canonical_symbol: str,
    ingested_at: datetime.datetime,
) -> Any:
    """Convert CryptoCompare daily OHLCV records to a normalized DataFrame."""
    import pandas as pd

    rows = []
    for candle in raw:
        rows.append(
            {
                "timestamp": datetime.datetime.fromtimestamp(int(candle["time"]), tz=datetime.timezone.utc),
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle["volumefrom"]),
                "canonical_symbol": canonical_symbol,
                "provider": "cryptocompare",
                "exchange_id": "",
                "timeframe": "1d",
                "ingested_at": ingested_at,
            }
        )
    return pd.DataFrame(rows)


def _normalize_coingecko_ohlcv(
    raw: list[list],
    canonical_symbol: str,
    ingested_at: datetime.datetime,
) -> Any:
    """Convert CoinGecko OHLC records to a normalized DataFrame.

    CoinGecko's OHLC endpoint does not return a volume column; ``volume`` is
    set to ``0.0`` to satisfy the OHLCV schema while making the absence clear.
    """
    import pandas as pd

    rows = []
    for candle in raw:
        rows.append(
            {
                "timestamp": datetime.datetime.fromtimestamp(int(candle[0]) / 1000, tz=datetime.timezone.utc),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": 0.0,
                "canonical_symbol": canonical_symbol,
                "provider": "coingecko",
                "exchange_id": "",
                "timeframe": "1d",
                "ingested_at": ingested_at,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_ts_range(df: Any) -> tuple[str, str]:
    """Return (min_timestamp_iso, max_timestamp_iso) from *df*."""
    if "timestamp" not in df.columns or df.empty:
        return "", ""
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    min_str = ts_min.isoformat() if hasattr(ts_min, "isoformat") else str(ts_min)
    max_str = ts_max.isoformat() if hasattr(ts_max, "isoformat") else str(ts_max)
    return min_str, max_str


def _make_cache_hit_result(
    df: Any,
    canonical_symbol: str,
    cache_path: str,
) -> IngestionResult:
    """Build a successful cache-hit :class:`IngestionResult` from *df*."""
    min_ts, max_ts = _extract_ts_range(df)
    provider = str(df["provider"].iloc[0]) if "provider" in df.columns and not df.empty else "cache"
    return IngestionResult(
        canonical_symbol=canonical_symbol,
        asset_class="crypto",
        provider=provider,
        success=True,
        rows_loaded=len(df),
        cache_hit=True,
        cache_path=cache_path,
        min_timestamp=min_ts,
        max_timestamp=max_ts,
    )


def _make_download_result(
    df: Any,
    canonical_symbol: str,
    provider_name: str,
    cache_path: str,
) -> IngestionResult:
    """Build a successful download :class:`IngestionResult` from *df*."""
    min_ts, max_ts = _extract_ts_range(df)
    return IngestionResult(
        canonical_symbol=canonical_symbol,
        asset_class="crypto",
        provider=provider_name,
        success=True,
        rows_loaded=len(df),
        cache_hit=False,
        cache_path=cache_path,
        min_timestamp=min_ts,
        max_timestamp=max_ts,
    )


def _try_fallback_providers(
    symbol: str,
    mapper: CryptoSymbolMapper,
    fallback_providers: list,
    lookback_days: int,
    ingested_at: datetime.datetime,
) -> tuple[Any, str]:
    """Try each fallback provider in order; return (df, provider_name).

    Raises :exc:`EmptyPriceDataError` when all fallback providers fail.
    """
    last_exc: Exception | None = None
    for provider in fallback_providers:
        try:
            if isinstance(provider, CryptoCompareProvider):
                logger.info(
                    "Trying CryptoCompare fallback for '%s' (limit=%d)",
                    symbol,
                    lookback_days,
                )
                raw = provider.get_daily_ohlcv(symbol, limit=lookback_days)
                df = _normalize_cryptocompare_ohlcv(raw, symbol, ingested_at)
                return df, "cryptocompare"

            if isinstance(provider, CoinGeckoProvider):
                coingecko_id = mapper.to_coingecko_id(symbol)
                logger.info(
                    "Trying CoinGecko fallback for '%s' (id=%s, days=%d)",
                    symbol,
                    coingecko_id,
                    lookback_days,
                )
                raw = provider.get_ohlcv(coingecko_id, days=lookback_days)
                df = _normalize_coingecko_ohlcv(raw, symbol, ingested_at)
                return df, "coingecko"

            logger.warning(
                "Unknown fallback provider type %s – skipping.", type(provider).__name__
            )
        except Exception as exc:  # noqa: BLE001
            # Providers wrap errors in SpectraQuantError; catching Exception
            # here also handles test mocks that raise bare exceptions.
            logger.warning("Fallback provider %s failed for '%s': %s", type(provider).__name__, symbol, exc)
            last_exc = exc

    raise EmptyPriceDataError(
        f"All providers (CCXT + fallbacks) failed for '{symbol}'. "
        f"Last error: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_symbol_ohlcv(
    symbol: str,
    cache: CacheManager,
    mapper: CryptoSymbolMapper,
    run_mode: RunMode,
    ccxt_provider: CcxtProvider,
    fallback_providers: list | None = None,
    timeframe: str = "1d",
    lookback_days: int = 365,
    exchange_id: str = "binance",
) -> IngestionResult:
    """Download OHLCV data for a single canonical crypto symbol.

    Cache-first in NORMAL mode; strictly cache-only in TEST mode;
    always re-downloads in REFRESH mode.

    Args:
        symbol:            Canonical crypto ticker, e.g. ``"BTC"``.
        cache:             :class:`~spectraquant_v3.core.cache.CacheManager` instance.
        mapper:            :class:`~spectraquant_v3.crypto.symbols.mapper.CryptoSymbolMapper`.
        run_mode:          Controls cache vs. network behaviour.
        ccxt_provider:     Primary CCXT data provider.
        fallback_providers: Ordered list of fallback providers tried when CCXT
            fails.  Supported types: ``CryptoCompareProvider``,
            ``CoinGeckoProvider``.
        timeframe:         CCXT candle timeframe string, e.g. ``"1d"``.
        lookback_days:     Minimum number of candles required; also used as the
            ``limit`` argument for providers.
        exchange_id:       CCXT exchange identifier, e.g. ``"binance"``.

    Returns:
        :class:`~spectraquant_v3.core.ingestion_result.IngestionResult`

    Raises:
        AssetClassLeakError:     If *symbol* is detected as an equity ticker.
        SymbolResolutionError:   If *symbol* cannot be resolved via the mapper.
        CacheOnlyViolationError: In TEST mode when cache is absent or coverage
            is insufficient.
        EmptyPriceDataError:     When all providers return no data.
        DataSchemaError:         When the OHLCV DataFrame fails schema validation.
    """
    if fallback_providers is None:
        fallback_providers = []

    # --- 1. Asset-class guard ---
    mapper._assert_not_equity(symbol)

    canonical_symbol = symbol.upper()
    cache_key = canonical_symbol
    cache_path = str(cache.get_path(cache_key))
    ingested_at = datetime.datetime.now(tz=datetime.timezone.utc)

    # --- 2. TEST mode: cache-only, fail loudly on miss or insufficient coverage ---
    if run_mode == RunMode.TEST:
        cache.assert_network_allowed(cache_key)  # raises CacheOnlyViolationError if absent
        df = cache.read_parquet(cache_key)
        if len(df) < lookback_days:
            raise CacheOnlyViolationError(
                f"TEST mode: cached data for '{cache_key}' has {len(df)} rows "
                f"but {lookback_days} are required. "
                "Re-populate the cache or switch to run_mode=normal."
            )
        logger.debug("Cache hit (TEST mode) for '%s': %d rows.", canonical_symbol, len(df))
        return _make_cache_hit_result(df, canonical_symbol, cache_path)

    # --- 3. NORMAL mode: return cache when coverage is sufficient ---
    if run_mode == RunMode.NORMAL and cache.exists(cache_key):
        try:
            df = cache.read_parquet(cache_key)
            if len(df) >= lookback_days:
                logger.debug(
                    "Cache hit (NORMAL mode) for '%s': %d rows (required %d).",
                    canonical_symbol, len(df), lookback_days,
                )
                return _make_cache_hit_result(df, canonical_symbol, cache_path)
            logger.info(
                "Cache coverage insufficient for '%s' (%d/%d rows); re-downloading.",
                canonical_symbol, len(df), lookback_days,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cache read failed for '%s', re-downloading: %s", canonical_symbol, exc
            )

    # --- 4. REFRESH mode or cache miss/insufficient: download from providers ---
    provider_sym = mapper.to_provider_symbol(canonical_symbol)
    df: Any = None
    provider_used: str = ""

    try:
        logger.info(
            "Fetching OHLCV via CCXT for '%s' on '%s' (timeframe=%s, limit=%d).",
            provider_sym, exchange_id, timeframe, lookback_days,
        )
        raw = ccxt_provider.fetch_ohlcv(
            provider_sym,
            timeframe=timeframe,
            limit=lookback_days,
            exchange_id=exchange_id,
        )
        df = _normalize_ccxt_ohlcv(raw, canonical_symbol, exchange_id, timeframe, ingested_at)
        provider_used = f"ccxt/{exchange_id}"
    except Exception as exc:  # noqa: BLE001
        # CcxtProvider wraps all errors in SpectraQuantError; catching Exception
        # here also handles test mocks that raise bare exceptions.
        logger.warning(
            "CCXT fetch failed for '%s' on '%s': %s. Trying fallbacks.",
            canonical_symbol, exchange_id, exc,
        )
        if not fallback_providers:
            raise EmptyPriceDataError(
                f"CCXT failed for '{canonical_symbol}' and no fallback providers configured. "
                f"Underlying error: {exc}"
            ) from exc
        df, provider_used = _try_fallback_providers(
            canonical_symbol, mapper, fallback_providers, lookback_days, ingested_at
        )

    # --- 5. Schema validation ---
    validate_ohlcv_dataframe(df, symbol=canonical_symbol, min_rows=1)

    # --- 6. Atomic cache write ---
    written_path = cache.write_parquet(cache_key, df)
    logger.info(
        "Cached %d OHLCV rows for '%s' (provider=%s) at %s.",
        len(df), canonical_symbol, provider_used, written_path,
    )

    return _make_download_result(df, canonical_symbol, provider_used, str(written_path))


def download_many_ohlcv(
    symbols: list[str],
    cache: CacheManager,
    mapper: CryptoSymbolMapper,
    run_mode: RunMode,
    ccxt_provider: CcxtProvider,
    fallback_providers: list | None = None,
    timeframe: str = "1d",
    lookback_days: int = 365,
    exchange_id: str = "binance",
    fail_fast: bool = False,
) -> dict[str, IngestionResult]:
    """Download OHLCV data for multiple canonical crypto symbols.

    Args:
        symbols:           List of canonical crypto tickers.
        cache:             :class:`~spectraquant_v3.core.cache.CacheManager`.
        mapper:            :class:`~spectraquant_v3.crypto.symbols.mapper.CryptoSymbolMapper`.
        run_mode:          Controls cache vs. network behaviour.
        ccxt_provider:     Primary CCXT data provider.
        fallback_providers: Fallback providers passed to
            :func:`download_symbol_ohlcv`.
        timeframe:         CCXT candle timeframe string.
        lookback_days:     Minimum candles required per symbol.
        exchange_id:       CCXT exchange identifier.
        fail_fast:         When ``True``, stop after the first failed symbol
            and return partial results.

    Returns:
        Dict mapping ``canonical_symbol → IngestionResult``.
    """
    if fallback_providers is None:
        fallback_providers = []

    results: dict[str, IngestionResult] = {}

    for symbol in symbols:
        canonical = symbol.upper()
        try:
            result = download_symbol_ohlcv(
                symbol=symbol,
                cache=cache,
                mapper=mapper,
                run_mode=run_mode,
                ccxt_provider=ccxt_provider,
                fallback_providers=fallback_providers,
                timeframe=timeframe,
                lookback_days=lookback_days,
                exchange_id=exchange_id,
            )
            results[canonical] = result
        except (AssetClassLeakError, SymbolResolutionError) as exc:
            logger.error("Symbol resolution error for '%s': %s", canonical, exc)
            results[canonical] = make_error_result(
                canonical_symbol=canonical,
                asset_class="crypto",
                provider="",
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
            if fail_fast:
                logger.warning("fail_fast=True: stopping after error on '%s'.", canonical)
                break
        except SpectraQuantError as exc:
            logger.error("Ingestion failed for '%s': %s", canonical, exc)
            results[canonical] = make_error_result(
                canonical_symbol=canonical,
                asset_class="crypto",
                provider="",
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
            if fail_fast:
                logger.warning("fail_fast=True: stopping after error on '%s'.", canonical)
                break
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error for '%s': %s", canonical, exc)
            results[canonical] = make_error_result(
                canonical_symbol=canonical,
                asset_class="crypto",
                provider="",
                error_code="UNEXPECTED_ERROR",
                error_message=str(exc),
            )
            if fail_fast:
                logger.warning("fail_fast=True: stopping after error on '%s'.", canonical)
                break

    return results


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------


async def async_download_symbol_ohlcv(
    symbol: str,
    cache: CacheManager,
    mapper: CryptoSymbolMapper,
    run_mode: RunMode,
    ccxt_provider: CcxtProvider,
    fallback_providers: list | None = None,
    timeframe: str = "1d",
    lookback_days: int = 365,
    exchange_id: str = "binance",
) -> IngestionResult:
    """Async wrapper around :func:`download_symbol_ohlcv`.

    Runs the synchronous download in a thread-pool executor so that
    blocking I/O does not stall the event loop.

    Args:
        symbol:            Canonical crypto ticker, e.g. ``"BTC"``.
        cache:             :class:`~spectraquant_v3.core.cache.CacheManager`.
        mapper:            :class:`~spectraquant_v3.crypto.symbols.mapper.CryptoSymbolMapper`.
        run_mode:          Controls cache vs. network behaviour.
        ccxt_provider:     Primary CCXT data provider.
        fallback_providers: Ordered fallback providers.
        timeframe:         CCXT candle timeframe string.
        lookback_days:     Minimum candles required.
        exchange_id:       CCXT exchange identifier.

    Returns:
        :class:`~spectraquant_v3.core.ingestion_result.IngestionResult`

    Raises:
        Same exceptions as :func:`download_symbol_ohlcv`.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: download_symbol_ohlcv(
            symbol=symbol,
            cache=cache,
            mapper=mapper,
            run_mode=run_mode,
            ccxt_provider=ccxt_provider,
            fallback_providers=fallback_providers,
            timeframe=timeframe,
            lookback_days=lookback_days,
            exchange_id=exchange_id,
        ),
    )


async def async_download_many_ohlcv(
    symbols: list[str],
    cache: CacheManager,
    mapper: CryptoSymbolMapper,
    run_mode: RunMode,
    ccxt_provider: CcxtProvider,
    fallback_providers: list | None = None,
    timeframe: str = "1d",
    lookback_days: int = 365,
    exchange_id: str = "binance",
    concurrency: int = 10,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> dict[str, IngestionResult]:
    """Download OHLCV data for many crypto symbols concurrently.

    Uses the :mod:`~spectraquant_v3.core.async_engine` to bound concurrency
    and retry failing symbols with exponential back-off.

    A single failing symbol is captured as a failure
    :class:`~spectraquant_v3.core.ingestion_result.IngestionResult` and does
    not crash the batch.

    Args:
        symbols:           Canonical crypto tickers.
        cache:             :class:`~spectraquant_v3.core.cache.CacheManager`.
        mapper:            :class:`~spectraquant_v3.crypto.symbols.mapper.CryptoSymbolMapper`.
        run_mode:          Controls cache vs. network behaviour.
        ccxt_provider:     Primary CCXT data provider.
        fallback_providers: Ordered fallback providers.
        timeframe:         CCXT candle timeframe string.
        lookback_days:     Minimum candles required per symbol.
        exchange_id:       CCXT exchange identifier.
        concurrency:       Maximum parallel downloads (semaphore bound).
        max_retries:       Per-symbol retry limit.
        base_delay:        Initial retry wait in seconds.
        max_delay:         Maximum retry wait cap in seconds.
        backoff_factor:    Exponential back-off multiplier.

    Returns:
        Dict mapping ``canonical_symbol → IngestionResult``.
        Every input symbol is present; failures carry ``success=False``.
    """
    from spectraquant_v3.core.async_engine import ingest_many_symbols  # noqa: PLC0415

    async def _fetch(symbol: str) -> IngestionResult:
        return await async_download_symbol_ohlcv(
            symbol=symbol,
            cache=cache,
            mapper=mapper,
            run_mode=run_mode,
            ccxt_provider=ccxt_provider,
            fallback_providers=fallback_providers,
            timeframe=timeframe,
            lookback_days=lookback_days,
            exchange_id=exchange_id,
        )

    summary = await ingest_many_symbols(
        symbols=symbols,
        ingestion_func=_fetch,
        concurrency=concurrency,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
    )

    results: dict[str, IngestionResult] = {}
    for symbol, raw in summary.results.items():
        if isinstance(raw, IngestionResult):
            results[symbol] = raw
        else:
            # AsyncIngestionError – convert to a failure IngestionResult
            results[symbol] = make_error_result(
                canonical_symbol=symbol,
                asset_class="crypto",
                provider="",
                error_code=raw.error_type,
                error_message=raw.error_message,
            )

    return results
