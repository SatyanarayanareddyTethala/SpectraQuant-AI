"""Orderbook ingestion placeholder for SpectraQuant-AI-V3.

This module defines the **full public interface** for live orderbook snapshot
ingestion so that QA/report integration can reference stable types now, while
the actual websocket implementation is delivered in a future sprint.

Current status
--------------
:func:`ingest_orderbook_snapshot` raises :exc:`NotImplementedError` with the
message ``"Live orderbook websocket not yet implemented"``.  All surrounding
types (:class:`OrderBookSnapshot`, :class:`OrderBookIngestionResult`,
:func:`get_orderbook_cache_key`) are fully implemented and production-ready.

Planned implementation
----------------------
The real implementation will subscribe to exchange websocket feeds (e.g.,
Binance ``depth`` stream, Bybit ``orderbook.200``), capture a point-in-time
L2 orderbook snapshot, persist it to the parquet cache, and return a populated
:class:`OrderBookIngestionResult`.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

from spectraquant_v3.core.ingestion_result import IngestionResult
from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper

logger = logging.getLogger(__name__)

_ASSET_CLASS = "crypto"
_PROVIDER_PLACEHOLDER = "orderbook_ws"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OrderBookSnapshot:
    """Point-in-time L2 orderbook snapshot for a single symbol on one exchange.

    Attributes:
        canonical_symbol: Upper-case canonical ticker, e.g. ``"BTC"``.
        exchange_id:      Exchange identifier, e.g. ``"binance"``.
        timestamp:        UTC datetime when the snapshot was captured.
        bids:             List of ``[price, quantity]`` bid levels, best first.
        asks:             List of ``[price, quantity]`` ask levels, best first.
        provider:         Provider / feed name, e.g. ``"binance_ws"``.
        ingested_at:      UTC datetime when the snapshot was persisted.
    """

    canonical_symbol: str
    exchange_id: str
    timestamp: datetime.datetime
    bids: list = field(default_factory=list)
    asks: list = field(default_factory=list)
    provider: str = _PROVIDER_PLACEHOLDER
    ingested_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc)
    )


@dataclass
class OrderBookIngestionResult(IngestionResult):
    """Ingestion result for a single orderbook snapshot.

    Extends :class:`~spectraquant_v3.core.ingestion_result.IngestionResult`
    with an optional reference to the captured snapshot.

    Attributes:
        orderbook_snapshot: The captured snapshot, or ``None`` when ingestion
            was not attempted (e.g., placeholder / not-implemented state).
    """

    orderbook_snapshot: OrderBookSnapshot | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_orderbook_cache_key(symbol: str, exchange_id: str) -> str:
    """Return the canonical parquet cache key for an orderbook snapshot.

    Args:
        symbol:      Canonical crypto ticker, e.g. ``"BTC"``.
        exchange_id: Exchange identifier, e.g. ``"binance"``.

    Returns:
        Cache key string, e.g. ``"orderbook__BTC__binance"``.
    """
    return f"orderbook__{symbol.upper()}__{exchange_id}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_orderbook_snapshot(
    symbol: str,
    mapper: CryptoSymbolMapper,
    exchange_id: str,
    provider: Any,
) -> OrderBookIngestionResult:
    """Capture a live L2 orderbook snapshot for *symbol* on *exchange_id*.

    .. note::
        **Placeholder implementation.**  This function currently raises
        :exc:`NotImplementedError`.  It will be implemented when the
        websocket feed infrastructure is available.  The interface is
        intentionally stable so that QA matrices, pipeline reporters, and
        any code that references :class:`OrderBookIngestionResult` or
        :class:`OrderBookSnapshot` can be written against the final API today.

    Args:
        symbol:      Canonical crypto ticker, e.g. ``"BTC"``.
        mapper:      :class:`~...mapper.CryptoSymbolMapper` for symbol
            validation and exchange-symbol translation.
        exchange_id: Target exchange identifier, e.g. ``"binance"``.
        provider:    Websocket / REST provider object (interface TBD).

    Returns:
        :class:`OrderBookIngestionResult` (future implementation).

    Raises:
        NotImplementedError: Always raised in this placeholder release.
    """
    raise NotImplementedError("Live orderbook websocket not yet implemented")
