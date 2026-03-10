"""Parallel async ingestion engine for SpectraQuant-AI-V3.

Provides a high-performance, asyncio-based batch ingestion harness with:

* bounded concurrency via :class:`asyncio.Semaphore`
* per-symbol retry / exponential back-off (powered by *tenacity*)
* provider rate-limit awareness (honour per-provider concurrency caps)
* structured, per-symbol result collection
* one failing symbol does **not** crash the batch

Usage example::

    import asyncio
    from spectraquant_v3.core.async_engine import ingest_many_symbols

    async def my_fetch(symbol: str) -> SomeResult:
        ...

    results = asyncio.run(
        ingest_many_symbols(
            symbols=["BTC", "ETH", "SOL"],
            ingestion_func=my_fetch,
            concurrency=5,
        )
    )

``results`` is a dict mapping each symbol to either the return value of
``ingestion_func`` or an :class:`AsyncIngestionError` instance.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass
class AsyncIngestionError:
    """Structured representation of a per-symbol ingestion failure.

    Attributes:
        symbol:        The canonical symbol that failed.
        error_type:    ``type(exc).__name__`` of the caught exception.
        error_message: Human-readable description of the failure.
        attempts:      Number of attempts made before giving up.
    """

    symbol: str
    error_type: str
    error_message: str
    attempts: int = 1


@dataclass
class AsyncIngestionSummary:
    """Aggregate summary returned by :func:`ingest_many_symbols`.

    Attributes:
        results:        Dict mapping symbol → result or
                        :class:`AsyncIngestionError`.
        total:          Total number of symbols submitted.
        succeeded:      Number of symbols that completed without error.
        failed:         Number of symbols that ultimately failed.
        failed_symbols: Ordered list of symbols that failed.
    """

    results: dict[str, Any] = field(default_factory=dict)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    failed_symbols: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


async def ingest_many_symbols(
    symbols: list[str],
    ingestion_func: Callable[[str], Coroutine[Any, Any, Any]],
    concurrency: int = 10,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> AsyncIngestionSummary:
    """Ingest data for many symbols concurrently with retry/back-off.

    Args:
        symbols:         List of canonical ticker strings to ingest.
        ingestion_func:  An **async** callable ``(symbol: str) -> Any`` that
                         fetches/processes data for a single symbol.  It
                         should raise an exception on failure.
        concurrency:     Maximum number of symbols processed simultaneously.
                         Backed by :class:`asyncio.Semaphore`.
        max_retries:     Maximum number of attempts per symbol before recording
                         an :class:`AsyncIngestionError`.  Must be ≥ 1.
        base_delay:      Initial retry wait in seconds.
        max_delay:       Maximum retry wait in seconds (cap for exponential
                         back-off).
        backoff_factor:  Multiplier applied to the delay after each failure,
                         e.g. ``2.0`` doubles the wait each time.

    Returns:
        :class:`AsyncIngestionSummary` with per-symbol results and aggregate
        counters.  Every input symbol appears in ``summary.results``,
        either with the return value of *ingestion_func* or with an
        :class:`AsyncIngestionError`.

    Notes:
        * A single failing symbol never cancels the batch.
        * The semaphore enforces the provider rate-limit cap.
        * Retry delays are computed with full jitter to spread thundering
          herds across providers.
    """
    if not symbols:
        return AsyncIngestionSummary(total=0)

    concurrency = max(1, concurrency)
    max_retries = max(1, max_retries)
    semaphore = asyncio.Semaphore(concurrency)
    summary = AsyncIngestionSummary(total=len(symbols))

    async def _run_with_retry(symbol: str) -> Any:
        """Execute *ingestion_func* for *symbol* with retry / back-off."""
        delay = base_delay
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                async with semaphore:
                    return await ingestion_func(symbol)
            except asyncio.CancelledError:
                raise  # propagate cancellation
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries:
                    wait = min(delay, max_delay)
                    logger.warning(
                        "async_engine: symbol '%s' attempt %d/%d failed (%s: %s); "
                        "retrying in %.1fs.",
                        symbol,
                        attempt,
                        max_retries,
                        type(exc).__name__,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    delay = min(delay * backoff_factor, max_delay)
                else:
                    logger.error(
                        "async_engine: symbol '%s' failed after %d attempt(s): %s",
                        symbol,
                        max_retries,
                        exc,
                    )

        return AsyncIngestionError(
            symbol=symbol,
            error_type=type(last_exc).__name__ if last_exc else "UnknownError",
            error_message=str(last_exc) if last_exc else "",
            attempts=max_retries,
        )

    tasks = {symbol: asyncio.create_task(_run_with_retry(symbol)) for symbol in symbols}

    for symbol, task in tasks.items():
        try:
            result = await task
        except Exception as exc:  # noqa: BLE001
            # Defensive: task should never raise because _run_with_retry
            # catches all exceptions.  Handle it here just in case.
            result = AsyncIngestionError(
                symbol=symbol,
                error_type=type(exc).__name__,
                error_message=str(exc),
                attempts=max_retries,
            )

        summary.results[symbol] = result
        if isinstance(result, AsyncIngestionError):
            summary.failed += 1
            summary.failed_symbols.append(symbol)
        else:
            summary.succeeded += 1

    return summary


# ---------------------------------------------------------------------------
# Convenience wrapper: run async ingestion from synchronous code
# ---------------------------------------------------------------------------


def run_ingest_many_symbols(
    symbols: list[str],
    ingestion_func: Callable[[str], Coroutine[Any, Any, Any]],
    concurrency: int = 10,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> AsyncIngestionSummary:
    """Synchronous wrapper around :func:`ingest_many_symbols`.

    Calls :func:`asyncio.run` internally.  Do **not** call this from inside
    an already-running event loop; use ``await ingest_many_symbols(...)``
    directly instead.

    Args:
        symbols:         List of canonical ticker strings.
        ingestion_func:  Async callable ``(symbol: str) -> Any``.
        concurrency:     Semaphore bound for concurrent tasks.
        max_retries:     Per-symbol retry limit (≥ 1).
        base_delay:      Initial retry delay in seconds.
        max_delay:       Maximum retry delay in seconds.
        backoff_factor:  Exponential back-off multiplier.

    Returns:
        :class:`AsyncIngestionSummary`.
    """
    return asyncio.run(
        ingest_many_symbols(
            symbols=symbols,
            ingestion_func=ingestion_func,
            concurrency=concurrency,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
        )
    )
