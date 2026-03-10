"""Coinbase Advanced Trade REST execution client.

Handles order placement, cancellation, balance queries and fill retrieval
through the Coinbase Advanced Trade API.  All network calls go through a
thin rate-limited wrapper to stay within API quotas.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
_DEFAULT_REST_URL = "https://api.coinbase.com"
_RATE_LIMIT_INTERVAL = 0.1  # seconds between requests


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class OrderResult:
    """Represents the outcome of an order submission or query."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    status: str
    qty: float
    filled_qty: float = 0.0
    price: float | None = None
    avg_fill_price: float | None = None
    created_at: str = ""
    error: str = ""


@dataclass
class AccountBalance:
    """Balance for a single currency in the trading account."""

    currency: str
    available: float
    hold: float

    @property
    def total(self) -> float:
        return self.available + self.hold


@dataclass
class Fill:
    """A single trade fill."""

    trade_id: str
    order_id: str
    symbol: str
    side: str
    price: float
    size: float
    fee: float
    timestamp: str


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class CoinbaseExecutor:
    """Synchronous REST client for Coinbase Advanced Trade order execution.

    Args:
        api_key: Coinbase API key.
        api_secret: Coinbase API secret.
        base_url: REST endpoint override.
        rate_limit_interval: Minimum seconds between consecutive API calls.

    Example::

        executor = CoinbaseExecutor(api_key="…", api_secret="…")
        result = executor.place_order("BTC-USD", OrderSide.BUY, qty=0.001)
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = _DEFAULT_REST_URL,
        rate_limit_interval: float = _RATE_LIMIT_INTERVAL,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._rate_limit_interval = rate_limit_interval
        self._last_request_ts: float = 0.0

    # -- public API ---------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: OrderSide | str,
        qty: float,
        order_type: OrderType | str = OrderType.MARKET,
        price: float | None = None,
    ) -> OrderResult:
        """Submit a new order to Coinbase.

        Args:
            symbol: Product id, e.g. ``"BTC-USD"``.
            side: ``"BUY"`` or ``"SELL"``.
            qty: Base quantity.
            order_type: ``"MARKET"`` or ``"LIMIT"``.
            price: Required for limit orders.

        Returns:
            An :class:`OrderResult` describing the submitted order.
        """
        side = OrderSide(str(side).upper())
        order_type = OrderType(str(order_type).upper())

        if order_type is OrderType.LIMIT and price is None:
            raise ValueError("price is required for LIMIT orders")

        client_order_id = uuid.uuid4().hex
        order_config = self._build_order_config(order_type, qty, price)

        body = {
            "client_order_id": client_order_id,
            "product_id": symbol.upper(),
            "side": side.value,
            "order_configuration": order_config,
        }

        data = self._request("POST", "/api/v3/brokerage/orders", body=body)

        success_resp = data.get("success_response", {})
        error_resp = data.get("error_response", {})
        if error_resp:
            logger.error("Order rejected: %s", error_resp)
            return OrderResult(
                order_id=success_resp.get("order_id", ""),
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                status=OrderStatus.FAILED.value,
                qty=qty,
                error=error_resp.get("message", str(error_resp)),
            )

        return OrderResult(
            order_id=success_resp.get("order_id", client_order_id),
            symbol=symbol,
            side=side.value,
            order_type=order_type.value,
            status=OrderStatus.PENDING.value,
            qty=qty,
            price=price,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Returns:
            ``True`` if the cancellation request was accepted.
        """
        body = {"order_ids": [order_id]}
        data = self._request(
            "POST", "/api/v3/brokerage/orders/batch_cancel", body=body
        )
        results = data.get("results", [])
        if results and results[0].get("success"):
            logger.info("Order %s cancelled.", order_id)
            return True
        logger.warning("Cancel request for %s was not successful: %s", order_id, data)
        return False

    def get_balances(self) -> list[AccountBalance]:
        """Retrieve all non-zero account balances.

        Returns:
            List of :class:`AccountBalance` objects.
        """
        data = self._request("GET", "/api/v3/brokerage/accounts")
        accounts: list[AccountBalance] = []
        for acct in data.get("accounts", []):
            available = float(acct.get("available_balance", {}).get("value", 0))
            hold = float(acct.get("hold", {}).get("value", 0))
            if available > 0 or hold > 0:
                accounts.append(
                    AccountBalance(
                        currency=acct.get("currency", ""),
                        available=available,
                        hold=hold,
                    )
                )
        return accounts

    def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Return currently open orders, optionally filtered by *symbol*.

        Args:
            symbol: If provided, only orders for this product are returned.

        Returns:
            List of :class:`OrderResult` objects.
        """
        params = {"order_status": "OPEN"}
        if symbol:
            params["product_id"] = symbol.upper()
        data = self._request(
            "GET", "/api/v3/brokerage/orders/historical/batch", params=params
        )
        return [self._parse_order(o) for o in data.get("orders", [])]

    def get_fills(
        self,
        symbol: str | None = None,
        order_id: str | None = None,
    ) -> list[Fill]:
        """Retrieve recent fills.

        Args:
            symbol: Filter by product id.
            order_id: Filter by a specific order.

        Returns:
            List of :class:`Fill` objects.
        """
        params: dict[str, str] = {}
        if symbol:
            params["product_id"] = symbol.upper()
        if order_id:
            params["order_id"] = order_id
        data = self._request(
            "GET", "/api/v3/brokerage/orders/historical/fills", params=params
        )
        fills: list[Fill] = []
        for f in data.get("fills", []):
            fills.append(
                Fill(
                    trade_id=f.get("trade_id", ""),
                    order_id=f.get("order_id", ""),
                    symbol=f.get("product_id", ""),
                    side=f.get("side", ""),
                    price=float(f.get("price", 0)),
                    size=float(f.get("size", 0)),
                    fee=float(f.get("commission", 0)),
                    timestamp=f.get("trade_time", ""),
                )
            )
        return fills

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _build_order_config(
        order_type: OrderType,
        qty: float,
        price: float | None,
    ) -> dict[str, Any]:
        """Build the ``order_configuration`` payload expected by the API."""
        if order_type is OrderType.MARKET:
            return {"market_market_ioc": {"base_size": str(qty)}}
        return {
            "limit_limit_gtc": {
                "base_size": str(qty),
                "limit_price": str(price),
            }
        }

    def _rate_limit(self) -> None:
        """Block until the minimum interval since the last request has passed."""
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self._rate_limit_interval:
            time.sleep(self._rate_limit_interval - elapsed)
        self._last_request_ts = time.monotonic()

    def _sign(self, timestamp: str, method: str, path: str, body: str) -> str:
        """Generate the CB-ACCESS-SIGN HMAC-SHA256 signature."""
        message = f"{timestamp}{method.upper()}{path}{body}"
        return hmac.new(
            self._api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute an authenticated HTTP request against the Coinbase API.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            RuntimeError: On non-2xx responses.
        """
        import urllib.request
        import urllib.parse
        import urllib.error

        self._rate_limit()

        url = self._base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)

        body_str = json.dumps(body) if body else ""
        timestamp = str(int(time.time()))
        signature = self._sign(timestamp, method, path, body_str)

        headers = {
            "Content-Type": "application/json",
            "CB-ACCESS-KEY": self._api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
        }

        req = urllib.request.Request(
            url,
            data=body_str.encode() if body_str else None,
            headers=headers,
            method=method,
        )

        logger.debug("%s %s", method, url)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode() if exc.fp else ""
            logger.error(
                "HTTP %s from %s %s: %s", exc.code, method, url, error_body
            )
            raise RuntimeError(
                f"Coinbase API error {exc.code}: {error_body}"
            ) from exc

    @staticmethod
    def _parse_order(data: dict[str, Any]) -> OrderResult:
        """Parse a raw order dict into an :class:`OrderResult`."""
        return OrderResult(
            order_id=data.get("order_id", ""),
            symbol=data.get("product_id", ""),
            side=data.get("side", ""),
            order_type=data.get("order_type", ""),
            status=data.get("status", ""),
            qty=float(data.get("base_size", 0)),
            filled_qty=float(data.get("filled_size", 0)),
            price=float(data["limit_price"]) if "limit_price" in data else None,
            avg_fill_price=(
                float(data["average_filled_price"])
                if "average_filled_price" in data
                else None
            ),
            created_at=data.get("created_time", ""),
        )
