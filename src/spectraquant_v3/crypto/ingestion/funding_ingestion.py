"""Funding rate and open interest ingestion for SpectraQuant-AI-V3.

Fetches perpetual futures funding rates and open interest data from Binance
USD-M Futures or Bybit, then caches the combined result as parquet.  Funding
ingestion is **non-fatal**: provider failures produce an error
:class:`~spectraquant_v3.core.ingestion_result.IngestionResult` rather than
raising an exception.

Normalized DataFrame columns
-----------------------------
``timestamp``, ``canonical_symbol``, ``provider``, ``exchange_id``,
``funding_rate``, ``open_interest``, ``ingested_at``

Provider-specific key mapping
------------------------------
*BinanceFutures*:
  - ``fundingTime`` (int, milliseconds) → ``timestamp``
  - ``fundingRate`` (str) → ``funding_rate``
  - ``openInterest`` (str) from the snapshot endpoint → ``open_interest``
    for the most-recent funding-rate row; all others default to ``0.0``.

*Bybit*:
  - ``fundingRateTimestamp`` (str, milliseconds) → ``timestamp``
  - ``fundingRate`` (str) → ``funding_rate``
  - ``openInterest`` (str, from OI history) → ``open_interest`` merged by
    nearest-second timestamp; unmatched rows default to ``0.0``.

Run-mode semantics
------------------
NORMAL  – cache-first; fetch from provider when cache is absent.
TEST    – cache-only; returns an error result when cache is absent.
REFRESH – always re-fetch; overwrites the cache.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from spectraquant_v3.core.cache import CacheManager
from spectraquant_v3.core.enums import RunMode
from spectraquant_v3.core.errors import SpectraQuantError
from spectraquant_v3.core.ingestion_result import IngestionResult, make_error_result
from spectraquant_v3.crypto.ingestion.providers.binance_futures_provider import BinanceFuturesProvider
from spectraquant_v3.crypto.ingestion.providers.bybit_provider import BybitProvider
from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper

logger = logging.getLogger(__name__)

_ASSET_CLASS = "crypto"


# ---------------------------------------------------------------------------
# Internal normalizers
# ---------------------------------------------------------------------------


def _futures_symbol(canonical_symbol: str, quote: str = "USDT") -> str:
    """Construct the futures contract symbol from the canonical ticker."""
    return f"{canonical_symbol.upper()}{quote.upper()}"


def _normalize_binance_funding(
    funding_rows: list[dict],
    oi_snapshot: dict | None,
    canonical_symbol: str,
    exchange_id: str,
    ingested_at: datetime.datetime,
) -> list[dict]:
    """Convert BinanceFutures funding rate + OI snapshot to schema rows.

    The OI value from the point-in-time snapshot is applied to the most-recent
    funding-rate row.  All other rows receive ``open_interest=0.0``.
    """
    rows = []
    oi_value = 0.0
    if oi_snapshot:
        try:
            oi_value = float(oi_snapshot.get("openInterest", 0.0))
        except (TypeError, ValueError):
            oi_value = 0.0

    last_idx = len(funding_rows) - 1
    for i, row in enumerate(funding_rows):
        try:
            ts = datetime.datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=datetime.timezone.utc)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Could not parse fundingTime from row %d: %s", i, exc)
            ts = ingested_at

        try:
            funding_rate = float(row["fundingRate"])
        except (KeyError, TypeError, ValueError):
            funding_rate = 0.0

        rows.append(
            {
                "timestamp": ts,
                "canonical_symbol": canonical_symbol,
                "provider": "binance_futures",
                "exchange_id": exchange_id,
                "funding_rate": funding_rate,
                "open_interest": oi_value if i == last_idx else 0.0,
                "ingested_at": ingested_at,
            }
        )
    return rows


def _normalize_bybit_funding(
    funding_rows: list[dict],
    oi_rows: list[dict],
    canonical_symbol: str,
    exchange_id: str,
    ingested_at: datetime.datetime,
) -> list[dict]:
    """Convert Bybit funding rate + OI history to schema rows.

    OI values are matched to funding-rate rows by nearest second; unmatched
    rows receive ``open_interest=0.0``.
    """
    # Build OI lookup: timestamp_seconds → float(openInterest)
    oi_map: dict[int, float] = {}
    for entry in oi_rows:
        try:
            ts_sec = int(str(entry.get("timestamp", 0))) // 1000
            oi_map[ts_sec] = float(entry.get("openInterest", 0.0))
        except (TypeError, ValueError):
            pass

    rows = []
    for i, row in enumerate(funding_rows):
        try:
            ts_ms = int(str(row["fundingRateTimestamp"]))
            ts = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
            ts_sec = ts_ms // 1000
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Could not parse fundingRateTimestamp from row %d: %s", i, exc)
            ts = ingested_at
            ts_sec = 0

        try:
            funding_rate = float(row["fundingRate"])
        except (KeyError, TypeError, ValueError):
            funding_rate = 0.0

        # Exact match first, then nearest second within a ±4-hour window.
        oi_value = oi_map.get(ts_sec, 0.0)

        rows.append(
            {
                "timestamp": ts,
                "canonical_symbol": canonical_symbol,
                "provider": "bybit",
                "exchange_id": exchange_id,
                "funding_rate": funding_rate,
                "open_interest": oi_value,
                "ingested_at": ingested_at,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_ts_range(df: Any) -> tuple[str, str]:
    """Return (min_ts_iso, max_ts_iso) from the 'timestamp' column of *df*."""
    if "timestamp" not in df.columns or df.empty:
        return "", ""
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    min_str = ts_min.isoformat() if hasattr(ts_min, "isoformat") else str(ts_min)
    max_str = ts_max.isoformat() if hasattr(ts_max, "isoformat") else str(ts_max)
    return min_str, max_str


def _make_success_result(
    df: Any,
    canonical_symbol: str,
    provider_name: str,
    cache_path: str,
    cache_hit: bool,
) -> IngestionResult:
    min_ts, max_ts = _extract_ts_range(df)
    return IngestionResult(
        canonical_symbol=canonical_symbol,
        asset_class=_ASSET_CLASS,
        provider=provider_name,
        success=True,
        rows_loaded=len(df),
        cache_hit=cache_hit,
        cache_path=cache_path,
        min_timestamp=min_ts,
        max_timestamp=max_ts,
    )


def _fetch_and_normalize(
    primary_provider: BinanceFuturesProvider | BybitProvider,
    canonical_symbol: str,
    exchange_id: str,
    limit: int,
    ingested_at: datetime.datetime,
) -> tuple[list[dict], str]:
    """Fetch and normalize funding + OI rows from the given provider.

    Returns:
        ``(rows, provider_name)`` where *rows* is a list of schema dicts.

    Raises:
        :exc:`SpectraQuantError` on provider failures (caller decides whether
        to try the fallback or return an error result).
    """
    futures_sym = _futures_symbol(canonical_symbol)

    if isinstance(primary_provider, BinanceFuturesProvider):
        logger.info(
            "Fetching Binance funding rate for '%s' (limit=%d).", futures_sym, limit
        )
        funding_rows = primary_provider.get_funding_rate(futures_sym, limit=limit)

        oi_snapshot: dict | None = None
        try:
            oi_snapshot = primary_provider.get_open_interest(futures_sym)
        except SpectraQuantError as exc:
            logger.warning(
                "Binance open-interest fetch failed for '%s': %s; OI set to 0.0.",
                futures_sym, exc,
            )

        rows = _normalize_binance_funding(
            funding_rows, oi_snapshot, canonical_symbol, exchange_id, ingested_at
        )
        return rows, "binance_futures"

    if isinstance(primary_provider, BybitProvider):
        logger.info(
            "Fetching Bybit funding rate for '%s' (limit=%d).", futures_sym, limit
        )
        funding_rows = primary_provider.get_funding_rate(futures_sym, limit=limit)

        oi_rows: list[dict] = []
        try:
            oi_rows = primary_provider.get_open_interest(futures_sym, limit=limit)
        except SpectraQuantError as exc:
            logger.warning(
                "Bybit open-interest fetch failed for '%s': %s; OI set to 0.0.",
                futures_sym, exc,
            )

        rows = _normalize_bybit_funding(
            funding_rows, oi_rows, canonical_symbol, exchange_id, ingested_at
        )
        return rows, "bybit"

    raise SpectraQuantError(
        f"Unsupported funding provider type: {type(primary_provider).__name__}. "
        "Supported: BinanceFuturesProvider, BybitProvider."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_funding_for_symbol(
    symbol: str,
    cache: CacheManager,
    mapper: CryptoSymbolMapper,
    run_mode: RunMode,
    primary_provider: BinanceFuturesProvider | BybitProvider,
    fallback_provider: BinanceFuturesProvider | BybitProvider | None = None,
    exchange_id: str = "binance",
    limit: int = 100,
) -> IngestionResult:
    """Ingest funding rates and open interest for a single crypto symbol.

    Normalises all records to a DataFrame with columns:
    ``timestamp``, ``canonical_symbol``, ``provider``, ``exchange_id``,
    ``funding_rate``, ``open_interest``, ``ingested_at``.

    Args:
        symbol:           Canonical crypto ticker, e.g. ``"BTC"``.
        cache:            Cache manager instance.
        mapper:           Crypto symbol mapper (used for equity guard).
        run_mode:         Controls cache vs. network behaviour.
        primary_provider: Primary funding data provider (Binance or Bybit).
        fallback_provider: Optional fallback tried when the primary fails.
        exchange_id:      Exchange identifier used in cache key and DataFrame.
        limit:            Number of funding rate records to request.

    Returns:
        :class:`~spectraquant_v3.core.ingestion_result.IngestionResult`.
        Never raises – failures produce a ``success=False`` result.
    """
    canonical_symbol = symbol.upper()
    cache_key = f"funding__{canonical_symbol}__{exchange_id}"
    cache_path = str(cache.get_path(cache_key))

    try:
        mapper._assert_not_equity(symbol)
    except Exception as exc:  # noqa: BLE001
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider="",
            error_code="ASSET_CLASS_LEAK",
            error_message=str(exc),
        )

    # --- TEST mode: cache-only ---
    if run_mode == RunMode.TEST:
        if cache.exists(cache_key):
            try:
                df = cache.read_parquet(cache_key)
                logger.debug("Funding cache hit (TEST mode) for '%s'.", canonical_symbol)
                provider_name = (
                    str(df["provider"].iloc[0])
                    if "provider" in df.columns and not df.empty
                    else "cache"
                )
                return _make_success_result(df, canonical_symbol, provider_name, cache_path, cache_hit=True)
            except Exception as exc:  # noqa: BLE001
                return make_error_result(
                    canonical_symbol=canonical_symbol,
                    asset_class=_ASSET_CLASS,
                    provider="",
                    error_code="CACHE_READ_ERROR",
                    error_message=str(exc),
                )
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider="",
            error_code="CACHE_MISS_TEST_MODE",
            error_message=(
                f"TEST mode: no cached funding data for '{cache_key}'. "
                "Funding data is optional; populate the cache or use run_mode=normal."
            ),
        )

    # --- NORMAL mode: cache-first ---
    if run_mode == RunMode.NORMAL and cache.exists(cache_key):
        try:
            df = cache.read_parquet(cache_key)
            logger.debug("Funding cache hit (NORMAL mode) for '%s'.", canonical_symbol)
            provider_name = (
                str(df["provider"].iloc[0])
                if "provider" in df.columns and not df.empty
                else "cache"
            )
            return _make_success_result(df, canonical_symbol, provider_name, cache_path, cache_hit=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Funding cache read failed for '%s', re-fetching: %s", canonical_symbol, exc
            )

    # --- Fetch from primary provider ---
    ingested_at = datetime.datetime.now(tz=datetime.timezone.utc)
    rows: list[dict] = []
    provider_name = ""

    try:
        rows, provider_name = _fetch_and_normalize(
            primary_provider, canonical_symbol, exchange_id, limit, ingested_at
        )
    except SpectraQuantError as exc:
        logger.warning(
            "Primary funding provider failed for '%s': %s. %s",
            canonical_symbol,
            exc,
            "Trying fallback." if fallback_provider else "No fallback configured.",
        )
        if fallback_provider is None:
            return make_error_result(
                canonical_symbol=canonical_symbol,
                asset_class=_ASSET_CLASS,
                provider="",
                error_code="PROVIDER_ERROR",
                error_message=str(exc),
            )
        try:
            rows, provider_name = _fetch_and_normalize(
                fallback_provider, canonical_symbol, exchange_id, limit, ingested_at
            )
        except SpectraQuantError as fb_exc:
            return make_error_result(
                canonical_symbol=canonical_symbol,
                asset_class=_ASSET_CLASS,
                provider="",
                error_code="ALL_PROVIDERS_FAILED",
                error_message=(
                    f"Primary error: {exc}. Fallback error: {fb_exc}"
                ),
            )
        except Exception as fb_exc:  # noqa: BLE001
            return make_error_result(
                canonical_symbol=canonical_symbol,
                asset_class=_ASSET_CLASS,
                provider="",
                error_code="FALLBACK_UNEXPECTED_ERROR",
                error_message=str(fb_exc),
            )
    except Exception as exc:  # noqa: BLE001
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider="",
            error_code="UNEXPECTED_ERROR",
            error_message=str(exc),
        )

    if not rows:
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider=provider_name,
            error_code="EMPTY_FUNDING_DATA",
            error_message=f"Provider '{provider_name}' returned no funding rows for '{canonical_symbol}'.",
        )

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider=provider_name,
            error_code="NORMALIZATION_ERROR",
            error_message=str(exc),
        )

    try:
        written_path = cache.write_parquet(cache_key, df)
        logger.info(
            "Cached %d funding rows for '%s' (provider=%s) at %s.",
            len(df), canonical_symbol, provider_name, written_path,
        )
        cache_path = str(written_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to cache funding data for '%s': %s", canonical_symbol, exc)

    return _make_success_result(df, canonical_symbol, provider_name, cache_path, cache_hit=False)


def ingest_funding_for_many(
    symbols: list[str],
    cache: CacheManager,
    mapper: CryptoSymbolMapper,
    run_mode: RunMode,
    primary_provider: BinanceFuturesProvider | BybitProvider,
    fallback_provider: BinanceFuturesProvider | BybitProvider | None = None,
    exchange_id: str = "binance",
) -> dict[str, IngestionResult]:
    """Ingest funding rates and open interest for multiple crypto symbols.

    Args:
        symbols:          List of canonical crypto tickers.
        cache:            Cache manager instance.
        mapper:           Crypto symbol mapper.
        run_mode:         Controls cache vs. network behaviour.
        primary_provider: Primary funding data provider.
        fallback_provider: Optional fallback provider.
        exchange_id:      Exchange identifier.

    Returns:
        Dict mapping ``canonical_symbol → IngestionResult``.  Never raises.
    """
    results: dict[str, IngestionResult] = {}
    for symbol in symbols:
        canonical = symbol.upper()
        results[canonical] = ingest_funding_for_symbol(
            symbol=symbol,
            cache=cache,
            mapper=mapper,
            run_mode=run_mode,
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            exchange_id=exchange_id,
        )
    return results
