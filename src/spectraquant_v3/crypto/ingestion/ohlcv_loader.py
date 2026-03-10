"""Crypto OHLCV data loader for SpectraQuant-AI-V3.

Loads OHLCV data for canonical crypto tickers from the parquet cache
(via :class:`~spectraquant_v3.core.cache.CacheManager`) or downloads it
via the configured provider (yfinance as the default free provider).

RunMode enforcement:
- NORMAL  : cache-first; download missing data then write to cache.
- TEST    : cache-only; raises :exc:`CacheOnlyViolationError` on cache miss.
- REFRESH : always re-download; overwrites the existing cache entry.

Async support
-------------
:meth:`CryptoOHLCVLoader.load_many_async` runs symbol loads concurrently using
the :mod:`~spectraquant_v3.core.async_engine` for bounded concurrency and
per-symbol retry.  :meth:`CryptoOHLCVLoader.load_many` remains as a
synchronous sequential fallback.

This module must never import from ``spectraquant_v3.equities``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd

from spectraquant_v3.core.cache import CacheManager
from spectraquant_v3.core.enums import RunMode
from spectraquant_v3.core.schema import validate_ohlcv_dataframe
from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper
from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider helper
# ---------------------------------------------------------------------------

_YFINANCE_SUFFIX = "-USD"


def _yfinance_symbol_for(canonical_symbol: str) -> str:
    """Return the yfinance-compatible symbol for a canonical crypto ticker.

    For most tickers the mapping is ``<TICKER>-USD`` (e.g. ``BTC`` → ``BTC-USD``).
    If the canonical symbol already contains a hyphen it is returned unchanged.
    """
    upper = canonical_symbol.upper()
    return upper if "-" in upper else f"{upper}{_YFINANCE_SUFFIX}"


def _download_ohlcv_yfinance(
    yf_symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame | None:
    """Download OHLCV for *yf_symbol* via yfinance.

    Args:
        yf_symbol: yfinance-compatible symbol, e.g. ``"BTC-USD"``.
        period:    yfinance period string, e.g. ``"1y"``, ``"6mo"``.
        interval:  yfinance interval string, e.g. ``"1d"``, ``"1h"``.

    Returns:
        OHLCV DataFrame with lower-case columns, or ``None`` when unavailable.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        logger.warning(
            "yfinance is not installed; cannot download OHLCV for %s. "
            "Install it with: pip install yfinance",
            yf_symbol,
        )
        return None

    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance download failed for %s: %s", yf_symbol, exc)
        return None

    if df.empty:
        logger.warning("yfinance returned empty DataFrame for %s", yf_symbol)
        return None

    df.columns = [c.lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


class CryptoOHLCVLoader:
    """Cache-managed OHLCV loader for the crypto pipeline.

    Loads OHLCV DataFrames from the parquet cache or downloads them via
    yfinance when the cache is absent (or stale in REFRESH mode).

    Args:
        cache:    :class:`~spectraquant_v3.core.cache.CacheManager` wired to
                  the crypto cache directory.
        mapper:   :class:`~spectraquant_v3.crypto.symbols.mapper.CryptoSymbolMapper`
                  used for provider-symbol translation.
        run_mode: Cache behaviour (NORMAL / TEST / REFRESH).
        period:   yfinance history period (e.g. ``"1y"``).
        interval: yfinance data interval (e.g. ``"1d"``).
    """

    def __init__(
        self,
        cache: CacheManager,
        mapper: CryptoSymbolMapper,
        run_mode: RunMode = RunMode.NORMAL,
        period: str = "1y",
        interval: str = "1d",
    ) -> None:
        self._cache = cache
        self._mapper = mapper
        self._run_mode = run_mode
        self._period = period
        self._interval = interval

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        cache: CacheManager,
        registry: CryptoSymbolRegistry,
        run_mode: RunMode = RunMode.NORMAL,
    ) -> "CryptoOHLCVLoader":
        """Build from merged crypto config.

        Args:
            cfg:      Merged crypto configuration dict.
            cache:    Pre-wired :class:`~spectraquant_v3.core.cache.CacheManager`.
            registry: Pre-populated :class:`~spectraquant_v3.crypto.symbols.registry.CryptoSymbolRegistry`.
            run_mode: Cache behaviour.

        Returns:
            A fully-configured :class:`CryptoOHLCVLoader`.
        """
        ingestion_cfg = cfg.get("crypto", {}).get("ingestion", {})
        return cls(
            cache=cache,
            mapper=CryptoSymbolMapper(registry=registry),
            run_mode=run_mode,
            period=ingestion_cfg.get("period", "1y"),
            interval=ingestion_cfg.get("interval", "1d"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, canonical_symbol: str) -> pd.DataFrame | None:
        """Load OHLCV data for a single canonical crypto ticker.

        Cache-first behaviour (respects :attr:`run_mode`):

        - **NORMAL**: return cached data if present; otherwise download,
          validate, cache, and return.
        - **TEST**: return cached data if present; raise
          :exc:`~spectraquant_v3.core.errors.CacheOnlyViolationError`
          on cache miss (network is forbidden).
        - **REFRESH**: always download, validate, and overwrite the cache.

        Args:
            canonical_symbol: Upper-case coin ticker, e.g. ``"BTC"``.

        Returns:
            Validated OHLCV DataFrame, or ``None`` when data is unavailable.

        Raises:
            CacheOnlyViolationError: In TEST mode when the cache is absent.
        """
        cache_key = canonical_symbol.upper()

        # In TEST mode, assert cache is present before any operation.
        if self._run_mode == RunMode.TEST:
            self._cache.assert_network_allowed(cache_key)

        # Return cached data unless REFRESH is requested.
        if self._run_mode != RunMode.REFRESH and self._cache.exists(cache_key):
            try:
                df = self._cache.read_parquet(cache_key)
                logger.debug(
                    "CryptoOHLCVLoader: loaded %s from cache (%d rows)",
                    canonical_symbol,
                    len(df),
                )
                return df
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CryptoOHLCVLoader: cache read failed for %s: %s; "
                    "will attempt re-download.",
                    canonical_symbol,
                    exc,
                )
                if self._run_mode == RunMode.TEST:
                    raise

        # Download via yfinance.
        yf_sym = _yfinance_symbol_for(canonical_symbol)
        logger.info(
            "CryptoOHLCVLoader: downloading %s (%s) via yfinance …",
            canonical_symbol,
            yf_sym,
        )
        df = _download_ohlcv_yfinance(yf_sym, period=self._period, interval=self._interval)
        if df is None or df.empty:
            logger.warning(
                "CryptoOHLCVLoader: no OHLCV data returned for %s", canonical_symbol
            )
            return None

        # Validate schema before caching.
        try:
            validate_ohlcv_dataframe(df, symbol=canonical_symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CryptoOHLCVLoader: schema validation failed for %s: %s",
                canonical_symbol,
                exc,
            )
            return None

        # Persist to cache.
        try:
            self._cache.write_parquet(cache_key, df)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CryptoOHLCVLoader: cache write failed for %s: %s", canonical_symbol, exc
            )

        return df

    def load_many(
        self,
        canonical_symbols: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Load OHLCV data for multiple canonical crypto tickers (sequential).

        Symbols that fail to load (cache miss + download failure, schema error,
        etc.) are omitted from the result without aborting the batch.

        Args:
            canonical_symbols: List of upper-case coin tickers.

        Returns:
            Dict mapping canonical symbol → validated OHLCV DataFrame.
            Only symbols with data are included.
        """
        result: dict[str, pd.DataFrame] = {}
        for sym in canonical_symbols:
            df = self.load(sym)
            if df is not None:
                result[sym] = df
        return result

    async def load_many_async(
        self,
        canonical_symbols: list[str],
        concurrency: int = 10,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
    ) -> dict[str, pd.DataFrame]:
        """Load OHLCV data for multiple canonical crypto tickers concurrently.

        Uses :func:`~spectraquant_v3.core.async_engine.ingest_many_symbols` to
        bound the number of simultaneous downloads and to retry transient
        failures with exponential back-off.

        A symbol that cannot be loaded is silently omitted from the result
        (matching the behaviour of :meth:`load_many`); the failure is logged.

        Args:
            canonical_symbols: List of upper-case coin tickers.
            concurrency:       Maximum parallel loads (semaphore bound).
            max_retries:       Per-symbol retry limit before giving up.
            base_delay:        Initial retry wait in seconds.
            max_delay:         Maximum retry wait cap in seconds.
            backoff_factor:    Exponential back-off multiplier.

        Returns:
            Dict mapping canonical symbol → validated OHLCV DataFrame.
            Only successfully loaded symbols are included.
        """
        from spectraquant_v3.core.async_engine import ingest_many_symbols  # noqa: PLC0415

        loop = asyncio.get_running_loop()

        async def _load_one(sym: str) -> pd.DataFrame:
            df = await loop.run_in_executor(None, lambda: self.load(sym))
            if df is None or df.empty:
                raise ValueError(f"No OHLCV data available for '{sym}'")
            return df

        summary = await ingest_many_symbols(
            symbols=canonical_symbols,
            ingestion_func=_load_one,
            concurrency=concurrency,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
        )

        result: dict[str, pd.DataFrame] = {}
        for sym, item in summary.results.items():
            if isinstance(item, pd.DataFrame):
                result[sym] = item
            else:
                logger.warning(
                    "CryptoOHLCVLoader.load_many_async: '%s' failed – %s",
                    sym,
                    item.error_message if hasattr(item, "error_message") else item,
                )
        return result

    def load_many_async_run(
        self,
        canonical_symbols: list[str],
        concurrency: int = 10,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
    ) -> dict[str, pd.DataFrame]:
        """Synchronous wrapper around :meth:`load_many_async`.

        Calls :func:`asyncio.run` internally.  Do **not** call from inside
        an already-running event loop; use ``await load_many_async(...)``
        directly instead.

        Args:
            canonical_symbols: List of upper-case coin tickers.
            concurrency:       Maximum parallel loads.
            max_retries:       Per-symbol retry limit.
            base_delay:        Initial retry wait in seconds.
            max_delay:         Maximum retry wait cap in seconds.
            backoff_factor:    Exponential back-off multiplier.

        Returns:
            Dict mapping canonical symbol → validated OHLCV DataFrame.
        """
        return asyncio.run(
            self.load_many_async(
                canonical_symbols,
                concurrency=concurrency,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
            )
        )
