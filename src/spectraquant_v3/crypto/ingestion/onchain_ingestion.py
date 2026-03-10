"""On-chain metric ingestion for SpectraQuant-AI-V3.

Fetches on-chain metrics from the Glassnode API and caches them as parquet.
On-chain ingestion is **non-fatal**: provider failures produce an error
:class:`~spectraquant_v3.core.ingestion_result.IngestionResult` rather than
raising an exception, because downstream stages can proceed without on-chain
data.

Supported metrics (``metric_name → Glassnode metric path``):

.. code-block:: python

    METRIC_MAP = {
        "active_addresses": "addresses/active_count",
        "transaction_count": "transactions/count",
        "fees":              "fees/volume_sum",
        "exchange_inflow":   "transactions/transfers_volume_to_exchanges_sum",
    }

Run-mode semantics
------------------
NORMAL  – cache-first; fetch from Glassnode when cache is absent.
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
from spectraquant_v3.crypto.ingestion.providers.glassnode_provider import GlassnodeProvider
from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper

logger = logging.getLogger(__name__)

_PROVIDER_NAME = "glassnode"
_ASSET_CLASS = "crypto"

# Mapping from friendly metric name to the Glassnode REST API path.
METRIC_MAP: dict[str, str] = {
    "active_addresses": "addresses/active_count",
    "transaction_count": "transactions/count",
    "fees": "fees/volume_sum",
    "exchange_inflow": "transactions/transfers_volume_to_exchanges_sum",
}

_DEFAULT_METRICS: list[str] = ["active_addresses", "transaction_count", "fees"]


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
    cache_path: str,
    cache_hit: bool,
    warning_codes: list[str] | None = None,
) -> IngestionResult:
    min_ts, max_ts = _extract_ts_range(df)
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
        warning_codes=warning_codes or [],
    )


def _normalize_metric_rows(
    raw: list[dict],
    canonical_symbol: str,
    metric_name: str,
    ingested_at: datetime.datetime,
) -> list[dict]:
    """Convert Glassnode ``[{"t": unix_sec, "v": value}]`` to schema rows."""
    rows = []
    for entry in raw:
        rows.append(
            {
                "timestamp": datetime.datetime.fromtimestamp(int(entry["t"]), tz=datetime.timezone.utc),
                "canonical_symbol": canonical_symbol,
                "metric_name": metric_name,
                "value": float(entry["v"]) if entry["v"] is not None else None,
                "provider": _PROVIDER_NAME,
                "ingested_at": ingested_at,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_onchain_for_symbol(
    symbol: str,
    cache: CacheManager,
    mapper: CryptoSymbolMapper,
    run_mode: RunMode,
    provider: GlassnodeProvider,
    metrics: list[str] | None = None,
    since: int | None = None,
    until: int | None = None,
) -> IngestionResult:
    """Ingest on-chain metrics for a single canonical crypto symbol.

    Normalises each Glassnode metric series to rows with columns:
    ``timestamp``, ``canonical_symbol``, ``metric_name``, ``value``,
    ``provider``, ``ingested_at``.  All requested metrics are combined into a
    single parquet file keyed by ``onchain__{SYMBOL}``.

    When individual metrics fail, a ``warning_codes`` entry is added but the
    function still returns ``success=True`` if at least one metric succeeds.

    Args:
        symbol:   Canonical crypto ticker, e.g. ``"BTC"``.
        cache:    Cache manager instance.
        mapper:   Crypto symbol mapper (used for equity guard).
        run_mode: Controls cache vs. network behaviour.
        provider: :class:`~...glassnode_provider.GlassnodeProvider`.
        metrics:  List of metric names from :data:`METRIC_MAP`.  Defaults to
            ``["active_addresses", "transaction_count", "fees"]``.
        since:    Start time as Unix timestamp (seconds).
        until:    End time as Unix timestamp (seconds).

    Returns:
        :class:`~spectraquant_v3.core.ingestion_result.IngestionResult`.
        Never raises – failures produce a ``success=False`` result.
    """
    canonical_symbol = symbol.upper()
    cache_key = f"onchain__{canonical_symbol}"
    cache_path = str(cache.get_path(cache_key))
    requested_metrics = metrics if metrics is not None else list(_DEFAULT_METRICS)

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

    # --- TEST mode: cache-only ---
    if run_mode == RunMode.TEST:
        if cache.exists(cache_key):
            try:
                df = cache.read_parquet(cache_key)
                logger.debug("On-chain cache hit (TEST mode) for '%s'.", canonical_symbol)
                return _make_success_result(df, canonical_symbol, cache_path, cache_hit=True)
            except Exception as exc:  # noqa: BLE001
                return make_error_result(
                    canonical_symbol=canonical_symbol,
                    asset_class=_ASSET_CLASS,
                    provider=_PROVIDER_NAME,
                    error_code="CACHE_READ_ERROR",
                    error_message=str(exc),
                )
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            error_code="CACHE_MISS_TEST_MODE",
            error_message=(
                f"TEST mode: no cached on-chain data for '{cache_key}'. "
                "On-chain data is optional; populate the cache or use run_mode=normal."
            ),
        )

    # --- NORMAL mode: cache-first ---
    if run_mode == RunMode.NORMAL and cache.exists(cache_key):
        try:
            df = cache.read_parquet(cache_key)
            logger.debug("On-chain cache hit (NORMAL mode) for '%s'.", canonical_symbol)
            return _make_success_result(df, canonical_symbol, cache_path, cache_hit=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "On-chain cache read failed for '%s', re-fetching: %s", canonical_symbol, exc
            )

    # --- Fetch each requested metric from Glassnode ---
    ingested_at = datetime.datetime.now(tz=datetime.timezone.utc)
    all_rows: list[dict] = []
    warning_codes: list[str] = []

    for metric_name in requested_metrics:
        metric_path = METRIC_MAP.get(metric_name)
        if metric_path is None:
            logger.warning(
                "Unknown on-chain metric '%s' for '%s'; skipping. "
                "Valid metrics: %s",
                metric_name, canonical_symbol, list(METRIC_MAP.keys()),
            )
            warning_codes.append(f"UNKNOWN_METRIC:{metric_name}")
            continue

        try:
            logger.info(
                "Fetching Glassnode metric '%s' (%s) for '%s'.",
                metric_name, metric_path, canonical_symbol,
            )
            raw = provider.get_metric(
                asset=canonical_symbol,
                metric_path=metric_path,
                since=since,
                until=until,
            )
            rows = _normalize_metric_rows(raw, canonical_symbol, metric_name, ingested_at)
            all_rows.extend(rows)
            logger.debug(
                "Fetched %d rows for metric '%s' (%s).",
                len(rows), metric_name, canonical_symbol,
            )
        except SpectraQuantError as exc:
            logger.warning(
                "Glassnode metric '%s' failed for '%s': %s", metric_name, canonical_symbol, exc
            )
            warning_codes.append(f"METRIC_FAILED:{metric_name}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unexpected error fetching metric '%s' for '%s': %s",
                metric_name, canonical_symbol, exc,
            )
            warning_codes.append(f"METRIC_ERROR:{metric_name}")

    if not all_rows:
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            error_code="ALL_METRICS_FAILED",
            error_message=(
                f"All on-chain metrics failed for '{canonical_symbol}'. "
                f"Attempted: {requested_metrics}. Warnings: {warning_codes}"
            ),
        )

    try:
        import pandas as pd

        df = pd.DataFrame(all_rows)
    except Exception as exc:  # noqa: BLE001
        return make_error_result(
            canonical_symbol=canonical_symbol,
            asset_class=_ASSET_CLASS,
            provider=_PROVIDER_NAME,
            error_code="NORMALIZATION_ERROR",
            error_message=str(exc),
        )

    try:
        written_path = cache.write_parquet(cache_key, df)
        logger.info(
            "Cached %d on-chain rows for '%s' at %s.",
            len(df), canonical_symbol, written_path,
        )
        cache_path = str(written_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to cache on-chain data for '%s': %s", canonical_symbol, exc)

    return _make_success_result(
        df, canonical_symbol, cache_path, cache_hit=False, warning_codes=warning_codes
    )


def ingest_onchain_for_many(
    symbols: list[str],
    cache: CacheManager,
    mapper: CryptoSymbolMapper,
    run_mode: RunMode,
    provider: GlassnodeProvider,
) -> dict[str, IngestionResult]:
    """Ingest on-chain metrics for multiple canonical crypto symbols.

    Args:
        symbols:   List of canonical crypto tickers.
        cache:     Cache manager instance.
        mapper:    Crypto symbol mapper.
        run_mode:  Controls cache vs. network behaviour.
        provider:  :class:`~...glassnode_provider.GlassnodeProvider`.

    Returns:
        Dict mapping ``canonical_symbol → IngestionResult``.  Never raises.
    """
    results: dict[str, IngestionResult] = {}
    for symbol in symbols:
        canonical = symbol.upper()
        results[canonical] = ingest_onchain_for_symbol(
            symbol=symbol,
            cache=cache,
            mapper=mapper,
            run_mode=run_mode,
            provider=provider,
        )
    return results
