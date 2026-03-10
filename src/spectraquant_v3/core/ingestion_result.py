"""Shared ingestion result dataclass for SpectraQuant-AI-V3.

Every ingestion provider returns an :class:`IngestionResult` so that pipeline
stages have a uniform, strongly-typed summary of each load attempt.  Callers
must never treat an empty or error result as a silent success.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IngestionResult:
    """Outcome of a single ingestion attempt for one canonical symbol.

    Attributes:
        canonical_symbol: Upper-case canonical ticker, e.g. ``"BTC"``.
        asset_class:      Asset-class string, e.g. ``"crypto"`` or ``"equity"``.
        provider:         Human-readable provider name, e.g. ``"ccxt/binance"``.
        success:          ``True`` only when data was loaded without errors.
        rows_loaded:      Number of rows in the resulting dataset.
        cache_hit:        ``True`` when data came from the local cache.
        cache_path:       Absolute path of the cache file used/written.
        min_timestamp:    ISO-8601 timestamp of the earliest row.
        max_timestamp:    ISO-8601 timestamp of the most recent row.
        warning_codes:    Non-fatal warning codes emitted during ingestion.
        error_code:       Short machine-readable code when ``success=False``.
        error_message:    Human-readable description when ``success=False``.
    """

    canonical_symbol: str
    asset_class: str
    provider: str
    success: bool
    rows_loaded: int
    cache_hit: bool
    cache_path: str
    min_timestamp: str
    max_timestamp: str
    warning_codes: list[str] = field(default_factory=list)
    error_code: str = ""
    error_message: str = ""


def make_error_result(
    canonical_symbol: str,
    asset_class: str,
    provider: str,
    error_code: str,
    error_message: str,
) -> IngestionResult:
    """Return a pre-filled failure :class:`IngestionResult`.

    Args:
        canonical_symbol: Upper-case canonical ticker.
        asset_class:      Asset-class string.
        provider:         Provider name.
        error_code:       Short machine-readable error code.
        error_message:    Human-readable error description.

    Returns:
        An :class:`IngestionResult` with ``success=False`` and all numeric /
        timestamp fields set to sentinel values.
    """
    return IngestionResult(
        canonical_symbol=canonical_symbol,
        asset_class=asset_class,
        provider=provider,
        success=False,
        rows_loaded=0,
        cache_hit=False,
        cache_path="",
        min_timestamp="",
        max_timestamp="",
        error_code=error_code,
        error_message=error_message,
    )
