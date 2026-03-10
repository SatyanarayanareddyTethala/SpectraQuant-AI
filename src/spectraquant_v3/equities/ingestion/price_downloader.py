"""Equity OHLCV price downloader for SpectraQuant-AI-V3.

Downloads historical OHLCV data for canonical equity tickers, enforces
run-mode cache semantics, validates schema, and persists to the parquet cache.

Equity OHLCV failures are **FATAL** — a missing or empty price series raises
:exc:`~spectraquant_v3.core.errors.EmptyPriceDataError` rather than returning
a partial or empty result.

RunMode enforcement:
- NORMAL  : cache-first with coverage check; download on miss or insufficient rows.
- TEST    : cache-only; raises :exc:`~spectraquant_v3.core.errors.CacheOnlyViolationError`
            on cache miss (network calls forbidden).
- REFRESH : always re-download; overwrites existing cache entry.

Async support
-------------
:func:`async_download_symbol_ohlcv` and :func:`async_download_many_ohlcv`
are asyncio-based equivalents that delegate to the synchronous counterparts
via :func:`asyncio.get_running_loop().run_in_executor`.

This module must NEVER import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import TYPE_CHECKING

from spectraquant_v3.core.enums import RunMode
from spectraquant_v3.core.errors import CacheOnlyViolationError, EmptyPriceDataError
from spectraquant_v3.core.ingestion_result import IngestionResult, make_error_result
from spectraquant_v3.core.schema import validate_ohlcv_dataframe

if TYPE_CHECKING:
    import pandas as pd

    from spectraquant_v3.core.cache import CacheManager
    from spectraquant_v3.equities.ingestion.providers.yfinance_provider import YFinanceProvider
    from spectraquant_v3.equities.symbols.mapper import EquitySymbolMapper

logger = logging.getLogger(__name__)

_ASSET_CLASS = "equity"
_PROVIDER_NAME = "yfinance"


# ---------------------------------------------------------------------------
# Single-symbol downloader
# ---------------------------------------------------------------------------


def download_symbol_ohlcv(
    symbol: str,
    cache: "CacheManager",
    mapper: "EquitySymbolMapper",
    run_mode: RunMode = RunMode.NORMAL,
    provider: "YFinanceProvider | None" = None,
    period: str = "5y",
    interval: str = "1d",
    lookback_days: int = 0,
) -> IngestionResult:
    """Download OHLCV data for a single canonical equity ticker.

    Enforces run-mode cache semantics and returns a fully-populated
    :class:`~spectraquant_v3.core.ingestion_result.IngestionResult`.

    Args:
        symbol:       Canonical equity ticker, e.g. ``"INFY.NS"``.
        cache:        :class:`~spectraquant_v3.core.cache.CacheManager` wired
                      to the equity cache directory.
        mapper:       :class:`~spectraquant_v3.equities.symbols.mapper.EquitySymbolMapper`
                      for provider-symbol translation and asset-class validation.
        run_mode:     Cache behaviour (NORMAL / TEST / REFRESH).
        provider:     :class:`~spectraquant_v3.equities.ingestion.providers.yfinance_provider.YFinanceProvider`
                      instance.  When ``None`` a default instance is created.
        period:       yfinance history period, e.g. ``"5y"``.
        interval:     yfinance data interval, e.g. ``"1d"``.
        lookback_days: Minimum number of rows required for a cached result to
                       be considered sufficient.  0 means any non-empty cache
                       is accepted.

    Returns:
        :class:`~spectraquant_v3.core.ingestion_result.IngestionResult` with
        ``success=True`` on success.

    Raises:
        AssetClassLeakError:    If *symbol* is detected as a crypto pair.
        CacheOnlyViolationError: In TEST mode when the cache is absent.
        EmptyPriceDataError:    When the provider returns no data.
    """
    import pandas as pd  # noqa: PLC0415

    from spectraquant_v3.equities.ingestion.providers.yfinance_provider import (  # noqa: PLC0415
        YFinanceProvider,
    )

    # Enforce equity-only asset class.
    mapper._assert_not_crypto(symbol)

    if provider is None:
        provider = YFinanceProvider()

    yf_symbol = mapper.to_yfinance_symbol(symbol)
    cache_key = symbol
    cache_path = str(cache.get_path(cache_key))

    # In TEST mode ensure the cache file exists before any logic branch.
    if run_mode == RunMode.TEST:
        cache.assert_network_allowed(cache_key)

    # ------------------------------------------------------------------
    # Cache probe (NORMAL mode only; REFRESH always skips)
    # ------------------------------------------------------------------
    if run_mode != RunMode.REFRESH and cache.exists(cache_key):
        try:
            cached_df: pd.DataFrame = cache.read_parquet(cache_key)
            coverage_ok = lookback_days == 0 or len(cached_df) >= lookback_days
            if coverage_ok:
                min_ts, max_ts = _timestamp_range(cached_df)
                logger.debug(
                    "download_symbol_ohlcv: cache hit for '%s' (%d rows)",
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
            # Insufficient coverage — fall through to download unless TEST mode.
            logger.info(
                "download_symbol_ohlcv: cache for '%s' has %d rows "
                "(need %d); will re-download.",
                symbol,
                len(cached_df),
                lookback_days,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "download_symbol_ohlcv: cache read failed for '%s': %s; "
                "will attempt download.",
                symbol,
                exc,
            )
            if run_mode == RunMode.TEST:
                raise

    # In TEST mode with a cache miss we must never attempt a network call.
    if run_mode == RunMode.TEST:
        raise CacheOnlyViolationError(
            f"TEST mode: cache miss (or insufficient coverage) for '{symbol}' "
            f"at {cache_path}. "
            "Network calls are forbidden in test mode. "
            "Pre-populate the cache or switch to run_mode=normal."
        )

    # ------------------------------------------------------------------
    # Download from provider
    # ------------------------------------------------------------------
    logger.info(
        "download_symbol_ohlcv: downloading '%s' (yf='%s') period=%s interval=%s",
        symbol,
        yf_symbol,
        period,
        interval,
    )

    df: pd.DataFrame = provider.download_ohlcv(
        yf_symbol, period=period, interval=interval
    )

    # provider.download_ohlcv raises on empty/error; double-check defensively.
    if df is None or df.empty:
        raise EmptyPriceDataError(
            f"Provider returned empty OHLCV data for equity '{symbol}'. "
            "Equity OHLCV is required — this is a fatal error."
        )

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------
    ingested_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    df = df.copy()
    df["canonical_symbol"] = symbol
    df["provider"] = _PROVIDER_NAME
    df["timeframe"] = interval
    df["ingested_at"] = ingested_at

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------
    validate_ohlcv_dataframe(df, symbol=symbol)

    # ------------------------------------------------------------------
    # Write to cache
    # ------------------------------------------------------------------
    cache.write_parquet(cache_key, df)

    min_ts, max_ts = _timestamp_range(df)
    logger.info(
        "download_symbol_ohlcv: cached '%s' (%d rows, %s→%s)",
        symbol,
        len(df),
        min_ts,
        max_ts,
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
# Batch downloader
# ---------------------------------------------------------------------------


def download_many_ohlcv(
    symbols: list[str],
    cache: "CacheManager",
    mapper: "EquitySymbolMapper",
    run_mode: RunMode = RunMode.NORMAL,
    provider: "YFinanceProvider | None" = None,
    period: str = "5y",
    interval: str = "1d",
    lookback_days: int = 0,
    fail_fast: bool = False,
) -> dict[str, IngestionResult]:
    """Download OHLCV data for multiple canonical equity tickers.

    Args:
        symbols:      List of canonical equity tickers.
        cache:        :class:`~spectraquant_v3.core.cache.CacheManager`.
        mapper:       :class:`~spectraquant_v3.equities.symbols.mapper.EquitySymbolMapper`.
        run_mode:     Cache behaviour (NORMAL / TEST / REFRESH).
        provider:     Optional :class:`~spectraquant_v3.equities.ingestion.providers.yfinance_provider.YFinanceProvider`.
        period:       yfinance history period.
        interval:     yfinance data interval.
        lookback_days: Minimum rows for cache coverage check.
        fail_fast:    When ``True``, re-raise the first error instead of
                      collecting error results and continuing.

    Returns:
        Dict mapping canonical symbol → :class:`~spectraquant_v3.core.ingestion_result.IngestionResult`.
        Every input symbol is represented, even on failure.
    """
    results: dict[str, IngestionResult] = {}

    for sym in symbols:
        try:
            results[sym] = download_symbol_ohlcv(
                symbol=sym,
                cache=cache,
                mapper=mapper,
                run_mode=run_mode,
                provider=provider,
                period=period,
                interval=interval,
                lookback_days=lookback_days,
            )
        except Exception as exc:  # noqa: BLE001
            if fail_fast:
                raise
            logger.error(
                "download_many_ohlcv: failed for '%s': %s",
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


def _timestamp_range(df: "pd.DataFrame") -> tuple[str, str]:
    """Return (min_timestamp, max_timestamp) strings from *df*.

    Prefers the ``"timestamp"`` column; falls back to the DataFrame index.
    Returns empty strings if neither is available.
    """
    try:
        if "timestamp" in df.columns:
            return str(df["timestamp"].min()), str(df["timestamp"].max())
        return str(df.index.min()), str(df.index.max())
    except Exception:  # noqa: BLE001
        return "", ""


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------


async def async_download_symbol_ohlcv(
    symbol: str,
    cache: "CacheManager",
    mapper: "EquitySymbolMapper",
    run_mode: RunMode = RunMode.NORMAL,
    provider: "YFinanceProvider | None" = None,
    period: str = "5y",
    interval: str = "1d",
    lookback_days: int = 0,
) -> IngestionResult:
    """Async wrapper around :func:`download_symbol_ohlcv`.

    Runs the synchronous download in a thread-pool executor so that
    blocking I/O does not stall the event loop.

    Args:
        symbol:       Canonical equity ticker, e.g. ``"INFY.NS"``.
        cache:        :class:`~spectraquant_v3.core.cache.CacheManager`.
        mapper:       :class:`~spectraquant_v3.equities.symbols.mapper.EquitySymbolMapper`.
        run_mode:     Cache behaviour (NORMAL / TEST / REFRESH).
        provider:     Optional :class:`~spectraquant_v3.equities.ingestion.providers.yfinance_provider.YFinanceProvider`.
        period:       yfinance history period.
        interval:     yfinance data interval.
        lookback_days: Minimum rows for cache coverage check.

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
            provider=provider,
            period=period,
            interval=interval,
            lookback_days=lookback_days,
        ),
    )


async def async_download_many_ohlcv(
    symbols: list[str],
    cache: "CacheManager",
    mapper: "EquitySymbolMapper",
    run_mode: RunMode = RunMode.NORMAL,
    provider: "YFinanceProvider | None" = None,
    period: str = "5y",
    interval: str = "1d",
    lookback_days: int = 0,
    concurrency: int = 10,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> dict[str, IngestionResult]:
    """Download OHLCV data for many equity tickers concurrently.

    Uses the :mod:`~spectraquant_v3.core.async_engine` to bound concurrency
    and retry failing symbols with exponential back-off.

    A single failing symbol is captured as a failure
    :class:`~spectraquant_v3.core.ingestion_result.IngestionResult` and does
    not crash the batch.

    Args:
        symbols:      Canonical equity tickers.
        cache:        :class:`~spectraquant_v3.core.cache.CacheManager`.
        mapper:       :class:`~spectraquant_v3.equities.symbols.mapper.EquitySymbolMapper`.
        run_mode:     Cache behaviour (NORMAL / TEST / REFRESH).
        provider:     Optional yfinance provider.
        period:       yfinance history period.
        interval:     yfinance data interval.
        lookback_days: Minimum rows for cache coverage check.
        concurrency:  Maximum parallel downloads (semaphore bound).
        max_retries:  Per-symbol retry limit.
        base_delay:   Initial retry wait in seconds.
        max_delay:    Maximum retry wait cap in seconds.
        backoff_factor: Exponential back-off multiplier.

    Returns:
        Dict mapping canonical symbol → :class:`~spectraquant_v3.core.ingestion_result.IngestionResult`.
        Every input symbol is present; failures carry ``success=False``.
    """
    from spectraquant_v3.core.async_engine import ingest_many_symbols  # noqa: PLC0415

    async def _fetch(sym: str) -> IngestionResult:
        return await async_download_symbol_ohlcv(
            symbol=sym,
            cache=cache,
            mapper=mapper,
            run_mode=run_mode,
            provider=provider,
            period=period,
            interval=interval,
            lookback_days=lookback_days,
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
    for sym, raw in summary.results.items():
        if isinstance(raw, IngestionResult):
            results[sym] = raw
        else:
            # AsyncIngestionError – convert to a failure IngestionResult
            results[sym] = make_error_result(
                canonical_symbol=sym,
                asset_class=_ASSET_CLASS,
                provider=_PROVIDER_NAME,
                error_code=raw.error_type,
                error_message=raw.error_message,
            )

    return results
