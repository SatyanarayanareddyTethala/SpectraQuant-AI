"""Multi-exchange execution wrapper using the *ccxt* library.

This module is **optional** — it gracefully degrades when *ccxt* is not
installed.  All public methods raise :class:`ImportError` with a helpful
message if the dependency is missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import ccxt as _ccxt  # type: ignore[import-untyped]

    _HAS_CCXT = True
except ImportError:
    _ccxt = None  # type: ignore[assignment]
    _HAS_CCXT = False


def _require_ccxt() -> None:
    """Raise if ccxt is not installed."""
    if not _HAS_CCXT:
        raise ImportError(
            "The 'ccxt' package is required for CCXTExecutor.  "
            "Install it with:  pip install ccxt"
        )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class CCXTOrderResult:
    """Normalised order result returned by :class:`CCXTExecutor`."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    status: str
    qty: float
    filled_qty: float = 0.0
    price: float | None = None
    avg_fill_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CCXTBalance:
    """Currency balance from a ccxt exchange."""

    currency: str
    free: float
    used: float
    total: float


@dataclass
class OHLCVBar:
    """A single OHLCV bar."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class CCXTExecutor:
    """Unified order execution and data retrieval across exchanges via ccxt.

    Args:
        exchange_id: Lower-case ccxt exchange identifier
            (e.g. ``"binance"``, ``"kraken"``, ``"coinbasepro"``).
        api_key: Exchange API key.
        api_secret: Exchange API secret.
        password: Exchange passphrase (required by some exchanges).
        sandbox: If ``True``, connect to the exchange sandbox/testnet.
        extra_config: Additional kwargs forwarded to the ccxt exchange
            constructor.

    Example::

        ex = CCXTExecutor("binance", api_key="…", api_secret="…")
        balances = ex.get_balances()
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        api_secret: str = "",
        password: str = "",
        sandbox: bool = False,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        _require_ccxt()

        exchange_class = getattr(_ccxt, exchange_id, None)
        if exchange_class is None:
            supported = ", ".join(sorted(_ccxt.exchanges[:10])) + " …"
            raise ValueError(
                f"Unknown exchange '{exchange_id}'.  "
                f"Supported (sample): {supported}"
            )

        config: dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        }
        if password:
            config["password"] = password
        if extra_config:
            config.update(extra_config)

        self._exchange: Any = exchange_class(config)
        if sandbox:
            self._exchange.set_sandbox_mode(True)

        self._exchange_id = exchange_id
        logger.info("CCXTExecutor initialised for %s (sandbox=%s)", exchange_id, sandbox)

    # -- orders -------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        price: float | None = None,
    ) -> CCXTOrderResult:
        """Place an order on the configured exchange.

        Args:
            symbol: Market symbol, e.g. ``"BTC/USDT"``.
            side: ``"buy"`` or ``"sell"``.
            qty: Amount in base currency.
            order_type: ``"market"`` or ``"limit"``.
            price: Required for limit orders.

        Returns:
            A :class:`CCXTOrderResult`.
        """
        side = side.lower()
        order_type = order_type.lower()

        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
        if order_type == "limit" and price is None:
            raise ValueError("price is required for limit orders")

        logger.info(
            "Placing %s %s order: %s qty=%s price=%s",
            order_type,
            side,
            symbol,
            qty,
            price,
        )
        raw = self._exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=qty,
            price=price,
        )
        return self._parse_order(raw)

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Exchange order id.
            symbol: Market symbol (required by most exchanges).

        Returns:
            ``True`` if the cancellation succeeded.
        """
        try:
            self._exchange.cancel_order(order_id, symbol)
            logger.info("Cancelled order %s on %s", order_id, symbol)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    # -- account ------------------------------------------------------------

    def get_balances(self) -> list[CCXTBalance]:
        """Return non-zero balances from the exchange account.

        Returns:
            List of :class:`CCXTBalance`.
        """
        raw = self._exchange.fetch_balance()
        balances: list[CCXTBalance] = []
        total_dict: dict[str, float] = raw.get("total", {})
        free_dict: dict[str, float] = raw.get("free", {})
        used_dict: dict[str, float] = raw.get("used", {})
        for currency, total in total_dict.items():
            total_val = float(total or 0)
            if total_val > 0:
                balances.append(
                    CCXTBalance(
                        currency=currency,
                        free=float(free_dict.get(currency) or 0),
                        used=float(used_dict.get(currency) or 0),
                        total=total_val,
                    )
                )
        return balances

    # -- market data --------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 100,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV candles from the exchange.

        Args:
            symbol: Market symbol, e.g. ``"BTC/USDT"``.
            timeframe: Candle interval (``"1m"``, ``"5m"``, ``"1h"``, …).
            since: Start time as a Unix timestamp in milliseconds.
            limit: Maximum number of candles to return.

        Returns:
            List of :class:`OHLCVBar`.
        """
        if not self._exchange.has.get("fetchOHLCV"):
            raise NotImplementedError(
                f"{self._exchange_id} does not support fetchOHLCV"
            )

        raw = self._exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since, limit=limit
        )
        bars: list[OHLCVBar] = []
        for row in raw:
            if len(row) < 6:
                continue
            bars.append(
                OHLCVBar(
                    timestamp=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        return bars

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _parse_order(raw: dict[str, Any]) -> CCXTOrderResult:
        """Convert a raw ccxt order dict into a :class:`CCXTOrderResult`."""
        return CCXTOrderResult(
            order_id=str(raw.get("id", "")),
            symbol=raw.get("symbol", ""),
            side=raw.get("side", ""),
            order_type=raw.get("type", ""),
            status=raw.get("status", ""),
            qty=float(raw.get("amount", 0)),
            filled_qty=float(raw.get("filled", 0)),
            price=float(raw["price"]) if raw.get("price") is not None else None,
            avg_fill_price=(
                float(raw["average"]) if raw.get("average") is not None else None
            ),
            raw=raw,
        )

    @property
    def exchange_id(self) -> str:
        """Return the exchange identifier."""
        return self._exchange_id
