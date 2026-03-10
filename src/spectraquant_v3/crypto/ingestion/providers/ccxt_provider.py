"""CCXT-based exchange data provider for SpectraQuant-AI-V3.

Wraps the `ccxt <https://github.com/ccxt/ccxt>`_ library to fetch OHLCV data
and market metadata from centralised crypto exchanges.  The provider is fully
mockable: pass ``exchange_overrides`` to inject pre-configured exchange objects
during testing without making real network calls.

Supported default exchange IDs: ``binance``, ``coinbase``, ``kraken``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from spectraquant_v3.core.errors import (
    DataSchemaError,
    EmptyPriceDataError,
    SpectraQuantError,
    SymbolResolutionError,
)

logger = logging.getLogger(__name__)

_SUPPORTED_EXCHANGE_IDS: frozenset[str] = frozenset({"binance", "coinbase", "kraken"})


def _import_ccxt() -> Any:
    """Lazily import ccxt and raise a clear error when it is not installed."""
    try:
        import ccxt  # noqa: PLC0415
        return ccxt
    except ImportError as exc:
        raise SpectraQuantError(
            "ccxt is not installed. Install it with: pip install ccxt"
        ) from exc


class CcxtProvider:
    """Fetch OHLCV data and market information via ccxt.

    Args:
        exchange_overrides: Optional mapping of ``exchange_id`` → pre-built
            exchange object.  When an entry is present for the requested
            ``exchange_id`` the provider uses that object instead of
            constructing one via ccxt.  Pass this in tests to inject mocks.
        exchange_factory:   Optional callable ``(exchange_id: str) -> exchange``
            used to build exchange objects for IDs not covered by
            ``exchange_overrides``.  Defaults to the ccxt class lookup.
    """

    def __init__(
        self,
        exchange_overrides: dict[str, Any] | None = None,
        exchange_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._overrides: dict[str, Any] = exchange_overrides or {}
        self._factory: Callable[[str], Any] | None = exchange_factory
        self._exchange_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_exchange(self, exchange_id: str) -> Any:
        """Return a cached or newly constructed exchange object."""
        if exchange_id in self._overrides:
            return self._overrides[exchange_id]

        if exchange_id not in self._exchange_cache:
            if self._factory is not None:
                exchange = self._factory(exchange_id)
            else:
                ccxt = _import_ccxt()
                if not hasattr(ccxt, exchange_id):
                    raise SymbolResolutionError(
                        f"ccxt does not support exchange '{exchange_id}'. "
                        f"Supported by this provider: {sorted(_SUPPORTED_EXCHANGE_IDS)}."
                    )
                exchange_cls = getattr(ccxt, exchange_id)
                exchange = exchange_cls()
            self._exchange_cache[exchange_id] = exchange

        return self._exchange_cache[exchange_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: int | None = None,
        limit: int | None = None,
        exchange_id: str = "binance",
    ) -> list[list]:
        """Fetch OHLCV candles for *symbol* from *exchange_id*.

        Args:
            symbol:      ccxt market symbol, e.g. ``"BTC/USDT"``.
            timeframe:   Candle timeframe, e.g. ``"1d"``, ``"1h"``.
            since:       Start time as a Unix timestamp in **milliseconds**.
            limit:       Maximum number of candles to return.
            exchange_id: Exchange to query (``"binance"``, ``"coinbase"``,
                         ``"kraken"``).

        Returns:
            List of OHLCV candles, each a ``[timestamp_ms, open, high, low,
            close, volume]`` list.

        Raises:
            SymbolResolutionError: When the market does not exist.
            EmptyPriceDataError:   When the exchange returns no candles.
            DataSchemaError:       When the response has an unexpected shape.
            SpectraQuantError:     For other ccxt errors.
        """
        exchange = self._get_exchange(exchange_id)
        kwargs: dict[str, Any] = {}
        if since is not None:
            kwargs["since"] = since
        if limit is not None:
            kwargs["limit"] = limit

        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, **kwargs)
        except Exception as exc:
            exc_name = type(exc).__name__
            # Translate well-known ccxt exceptions to SpectraQuantError subclasses.
            if "BadSymbol" in exc_name or "ExchangeError" in exc_name:
                raise SymbolResolutionError(
                    f"Market '{symbol}' not found on exchange '{exchange_id}': {exc}"
                ) from exc
            raise SpectraQuantError(
                f"ccxt fetch_ohlcv failed for '{symbol}' on '{exchange_id}': {exc}"
            ) from exc

        if not ohlcv:
            raise EmptyPriceDataError(
                f"ccxt returned empty OHLCV for '{symbol}' on '{exchange_id}' "
                f"(timeframe={timeframe}). An empty result is never a valid success."
            )

        if not isinstance(ohlcv[0], (list, tuple)) or len(ohlcv[0]) < 6:
            raise DataSchemaError(
                f"Unexpected OHLCV row shape from '{exchange_id}' for '{symbol}': "
                f"expected [ts, O, H, L, C, V], got {ohlcv[0]!r}."
            )

        logger.debug(
            "CcxtProvider: fetched %d candles for %s on %s",
            len(ohlcv),
            symbol,
            exchange_id,
        )
        return ohlcv  # type: ignore[return-value]

    def validate_market_exists(
        self,
        symbol: str,
        exchange_id: str = "binance",
    ) -> bool:
        """Return ``True`` if *symbol* exists as an active market on *exchange_id*.

        Args:
            symbol:      ccxt market symbol, e.g. ``"BTC/USDT"``.
            exchange_id: Exchange to query.

        Returns:
            ``True`` when the market is listed, ``False`` otherwise.

        Raises:
            SpectraQuantError: On exchange connectivity errors.
        """
        markets = self.load_markets(exchange_id)
        return symbol in markets

    def load_markets(self, exchange_id: str = "binance") -> dict:
        """Load and return the full markets dictionary for *exchange_id*.

        Args:
            exchange_id: Exchange to query.

        Returns:
            ccxt markets dict keyed by symbol string.

        Raises:
            SpectraQuantError: On connectivity or exchange errors.
        """
        exchange = self._get_exchange(exchange_id)
        try:
            markets: dict = exchange.load_markets()
        except Exception as exc:
            raise SpectraQuantError(
                f"ccxt load_markets failed for '{exchange_id}': {exc}"
            ) from exc

        logger.debug(
            "CcxtProvider: loaded %d markets from %s", len(markets), exchange_id
        )
        return markets
