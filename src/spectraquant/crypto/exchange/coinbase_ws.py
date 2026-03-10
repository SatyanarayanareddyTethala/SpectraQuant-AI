"""Coinbase WebSocket client for real-time market data with candle aggregation.

Connects to the Coinbase WebSocket feed, streams trades/tickers/orderbook,
aggregates trades into OHLCV candles, and flushes completed candles to parquet.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coinbase WebSocket endpoints
# ---------------------------------------------------------------------------
_ENDPOINTS: dict[str, dict[str, str]] = {
    "ws-feed": {
        "production": "wss://ws-feed.exchange.coinbase.com",
        "sandbox": "wss://ws-feed-public.sandbox.exchange.coinbase.com",
    },
    "ws-direct": {
        "production": "wss://ws-direct.exchange.coinbase.com",
        "sandbox": "wss://ws-direct.sandbox.exchange.coinbase.com",
    },
}

_VALID_CHANNELS = frozenset({"ticker", "matches", "level2"})

# Retry defaults (tenacity)
_RETRY_MIN_WAIT = 1
_RETRY_MAX_WAIT = 60
_RETRY_MAX_ATTEMPTS = 0  # 0 = infinite


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    """A single trade event."""

    symbol: str
    price: float
    size: float
    side: str
    timestamp: datetime
    trade_id: str = ""


@dataclass
class Ticker:
    """Snapshot of the current best bid/ask and last trade."""

    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    timestamp: datetime


@dataclass
class OrderBookLevel:
    """A single price level in the orderbook."""

    price: float
    size: float


@dataclass
class OrderBook:
    """Level-2 orderbook snapshot for a single product."""

    symbol: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Candle:
    """OHLCV candle for a single interval."""

    symbol: str
    timestamp: datetime  # UTC, start of interval
    open: float = 0.0
    high: float = -np.inf
    low: float = np.inf
    close: float = 0.0
    volume: float = 0.0
    trade_count: int = 0


# ---------------------------------------------------------------------------
# Candle aggregator
# ---------------------------------------------------------------------------
class CandleAggregator:
    """Aggregates trades into fixed-interval OHLCV candles.

    Parameters
    ----------
    intervals : list of int
        Candle widths in seconds (e.g. ``[60, 300]`` for 1m and 5m).
    output_dir : str or Path
        Directory for writing parquet files.
    """

    def __init__(
        self,
        intervals: list[int] | None = None,
        output_dir: str | Path = "data/prices/crypto",
    ) -> None:
        self._intervals = intervals or [60, 300]
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # _buckets[interval_sec][symbol] = current Candle being built
        self._buckets: dict[int, dict[str, Candle]] = {
            iv: {} for iv in self._intervals
        }
        self._candle_callbacks: list[Callable[[Candle], Any]] = []

    def on_candle(self, callback: Callable[[Candle], Any]) -> None:
        """Register a callback invoked when a candle closes."""
        self._candle_callbacks.append(callback)

    def _bucket_start(self, ts: datetime, interval: int) -> datetime:
        """Return the start of the bucket that *ts* falls into."""
        epoch = int(ts.timestamp())
        aligned = epoch - (epoch % interval)
        return datetime.fromtimestamp(aligned, tz=timezone.utc)

    def ingest(self, trade: Trade) -> list[Candle]:
        """Incorporate a trade; return any completed candles."""
        closed: list[tuple[int, Candle]] = []
        for iv in self._intervals:
            bucket_ts = self._bucket_start(trade.timestamp, iv)
            bucket = self._buckets[iv]
            current = bucket.get(trade.symbol)

            # New bucket → close old candle
            if current is not None and current.timestamp != bucket_ts:
                closed.append((iv, current))
                current = None

            if current is None:
                current = Candle(
                    symbol=trade.symbol,
                    timestamp=bucket_ts,
                    open=trade.price,
                    high=trade.price,
                    low=trade.price,
                    close=trade.price,
                    volume=trade.size,
                    trade_count=1,
                )
                bucket[trade.symbol] = current
            else:
                current.high = max(current.high, trade.price)
                current.low = min(current.low, trade.price)
                current.close = trade.price
                current.volume += trade.size
                current.trade_count += 1

        for iv, candle in closed:
            self._flush_candle(candle, iv)
            for cb in self._candle_callbacks:
                _safe_call(cb, candle)
        return [c for _, c in closed]

    def _flush_candle(self, candle: Candle, interval: int) -> None:
        """Append a completed candle to the parquet file for its symbol."""
        safe_symbol = candle.symbol.replace("/", "-")
        interval_label = _interval_label(interval)
        path = self._output_dir / f"{safe_symbol}_{interval_label}.parquet"

        row = pd.DataFrame(
            [
                {
                    "timestamp": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "trade_count": candle.trade_count,
                }
            ]
        )
        row["timestamp"] = pd.to_datetime(row["timestamp"], utc=True)
        row = row.set_index("timestamp")

        if path.exists():
            existing = pd.read_parquet(path)
            if not isinstance(existing.index, pd.DatetimeIndex):
                existing.index = pd.to_datetime(existing.index, utc=True)
            combined = pd.concat([existing, row])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            combined.to_parquet(path, engine="pyarrow")
        else:
            row.to_parquet(path, engine="pyarrow")
        logger.debug("Flushed candle %s %s → %s", candle.symbol, candle.timestamp, path)

    def flush_all(self) -> list[Candle]:
        """Force-flush any open (incomplete) candles.  Returns the flushed list."""
        flushed: list[Candle] = []
        for iv in self._intervals:
            for sym, candle in list(self._buckets[iv].items()):
                if candle.trade_count > 0:
                    self._flush_candle(candle, iv)
                    flushed.append(candle)
            self._buckets[iv].clear()
        return flushed


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------
class CoinbaseWSClient:
    """Async WebSocket client for Coinbase real-time market data.

    Example::

        client = CoinbaseWSClient(
            endpoint="ws-feed",
            environment="production",
        )
        client.subscribe(["BTC-USD", "ETH-USD"], ["ticker", "matches"])
        await client.start()
    """

    def __init__(
        self,
        endpoint: str = "ws-feed",
        environment: str = "production",
        ws_url: str | None = None,
        api_key: str = "",
        api_secret: str = "",
        candle_intervals: list[int] | None = None,
        output_dir: str | Path = "data/prices/crypto",
    ) -> None:
        if ws_url:
            self._ws_url = ws_url
        else:
            ep = _ENDPOINTS.get(endpoint, _ENDPOINTS["ws-feed"])
            self._ws_url = ep.get(environment, ep["production"])

        self._api_key = api_key
        self._api_secret = api_secret

        self._symbols: list[str] = []
        self._channels: list[str] = []

        # In-memory caches
        self._latest_trades: dict[str, Trade] = {}
        self._latest_tickers: dict[str, Ticker] = {}
        self._orderbooks: dict[str, OrderBook] = {}

        # User callbacks
        self._trade_callbacks: list[Callable[[Trade], Any]] = []
        self._ticker_callbacks: list[Callable[[Ticker], Any]] = []
        self._orderbook_callbacks: list[Callable[[OrderBook], Any]] = []

        # Candle aggregation
        self.aggregator = CandleAggregator(
            intervals=candle_intervals,
            output_dir=output_dir,
        )

        self._running = False
        self._ws: Any = None

    # -- configuration ------------------------------------------------------

    def subscribe(
        self,
        symbols: list[str],
        channels: list[str] | None = None,
    ) -> None:
        """Register *symbols* and *channels* to subscribe to on connect."""
        channels = channels or ["ticker"]
        invalid = set(channels) - _VALID_CHANNELS
        if invalid:
            raise ValueError(
                f"Unsupported channel(s): {invalid}. "
                f"Valid: {sorted(_VALID_CHANNELS)}"
            )
        self._symbols = [s.upper() for s in symbols]
        self._channels = list(channels)
        logger.info(
            "Subscription: symbols=%s channels=%s", self._symbols, self._channels
        )

    def on_trade(self, callback: Callable[[Trade], Any]) -> None:
        """Register a callback invoked on every incoming trade."""
        self._trade_callbacks.append(callback)

    def on_ticker(self, callback: Callable[[Ticker], Any]) -> None:
        """Register a callback invoked on every ticker update."""
        self._ticker_callbacks.append(callback)

    def on_orderbook(self, callback: Callable[[OrderBook], Any]) -> None:
        """Register a callback invoked on every orderbook update."""
        self._orderbook_callbacks.append(callback)

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Open the WebSocket and consume messages with retry/backoff."""
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "Install 'websockets': pip install websockets"
            ) from exc

        if not self._symbols:
            raise RuntimeError("Call subscribe() before start().")

        self._running = True
        logger.info("Connecting to %s …", self._ws_url)

        attempt = 0
        while self._running:
            try:
                async with websockets.connect(self._ws_url) as ws:
                    self._ws = ws
                    attempt = 0  # reset on successful connect
                    await self._send_subscribe(ws)
                    await self._recv_loop(ws)
            except asyncio.CancelledError:
                logger.info("WebSocket task cancelled.")
                break
            except Exception:
                if not self._running:
                    break
                attempt += 1
                wait = min(_RETRY_MIN_WAIT * (2 ** attempt), _RETRY_MAX_WAIT)
                logger.exception(
                    "Connection lost (attempt %d) — retrying in %.1fs",
                    attempt,
                    wait,
                )
                await asyncio.sleep(wait)

        # Flush remaining candle data
        self.aggregator.flush_all()
        logger.info("WebSocket client stopped.")

    async def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        logger.info("Stop requested — connection closed.")

    # -- polling accessors --------------------------------------------------

    def latest_trade(self, symbol: str) -> Trade | None:
        return self._latest_trades.get(symbol.upper())

    def latest_ticker(self, symbol: str) -> Ticker | None:
        return self._latest_tickers.get(symbol.upper())

    def latest_orderbook(self, symbol: str) -> OrderBook | None:
        return self._orderbooks.get(symbol.upper())

    # -- internals ----------------------------------------------------------

    async def _send_subscribe(self, ws: Any) -> None:
        """Send subscribe messages for each channel (Coinbase Exchange format)."""
        msg: dict[str, Any] = {
            "type": "subscribe",
            "product_ids": self._symbols,
            "channels": self._channels,
        }
        await ws.send(json.dumps(msg))
        logger.debug("Sent subscribe: %s", msg)

    async def _recv_loop(self, ws: Any) -> None:
        """Read messages from ws and dispatch to handlers."""
        async for raw in ws:
            if not self._running:
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON message ignored: %.120s", raw)
                continue
            msg_type = data.get("type", "")
            self._dispatch(msg_type, data)

    def _dispatch(self, msg_type: str, data: dict[str, Any]) -> None:
        """Route a parsed message to the appropriate handler."""
        if msg_type in ("match", "last_match"):
            self._handle_match(data)
        elif msg_type == "ticker":
            self._handle_ticker(data)
        elif msg_type == "l2update":
            self._handle_l2update(data)
        elif msg_type == "snapshot":
            self._handle_snapshot(data)

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse ISO-8601 timestamp to UTC datetime."""
        if not ts_str:
            return datetime.now(timezone.utc)
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)

    def _handle_match(self, data: dict[str, Any]) -> None:
        """Handle a trade (match) message and feed to aggregator."""
        trade = Trade(
            symbol=data.get("product_id", ""),
            price=float(data.get("price", 0)),
            size=float(data.get("size", 0)),
            side=data.get("side", ""),
            timestamp=self._parse_timestamp(data.get("time", "")),
            trade_id=str(data.get("trade_id", "")),
        )
        self._latest_trades[trade.symbol] = trade

        # Feed into candle aggregation
        self.aggregator.ingest(trade)

        for cb in self._trade_callbacks:
            _safe_call(cb, trade)

    def _handle_ticker(self, data: dict[str, Any]) -> None:
        ticker = Ticker(
            symbol=data.get("product_id", ""),
            price=float(data.get("price", 0)),
            bid=float(data.get("best_bid", 0)),
            ask=float(data.get("best_ask", 0)),
            volume_24h=float(data.get("volume_24h", 0)),
            timestamp=self._parse_timestamp(data.get("time", "")),
        )
        self._latest_tickers[ticker.symbol] = ticker
        for cb in self._ticker_callbacks:
            _safe_call(cb, ticker)

    def _handle_l2update(self, data: dict[str, Any]) -> None:
        product_id = data.get("product_id", "")
        book = self._orderbooks.setdefault(
            product_id, OrderBook(symbol=product_id)
        )
        for change in data.get("changes", []):
            if len(change) < 3:
                continue
            side, price_str, size_str = change[0], change[1], change[2]
            level = OrderBookLevel(price=float(price_str), size=float(size_str))
            if side == "buy":
                book.bids = _upsert_level(book.bids, level)
            elif side == "sell":
                book.asks = _upsert_level(book.asks, level)
        for cb in self._orderbook_callbacks:
            _safe_call(cb, book)

    def _handle_snapshot(self, data: dict[str, Any]) -> None:
        product_id = data.get("product_id", "")
        book = OrderBook(symbol=product_id)
        for bid in data.get("bids", []):
            if len(bid) >= 2:
                book.bids.append(
                    OrderBookLevel(price=float(bid[0]), size=float(bid[1]))
                )
        for ask in data.get("asks", []):
            if len(ask) >= 2:
                book.asks.append(
                    OrderBookLevel(price=float(ask[0]), size=float(ask[1]))
                )
        book.bids.sort(key=lambda lv: lv.price, reverse=True)
        book.asks.sort(key=lambda lv: lv.price)
        self._orderbooks[product_id] = book
        for cb in self._orderbook_callbacks:
            _safe_call(cb, book)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_call(cb: Callable[..., Any], arg: Any) -> None:
    try:
        cb(arg)
    except Exception:
        logger.exception("Callback %s raised an exception", cb)


def _interval_label(interval: int) -> str:
    """Convert an interval in seconds to a human-readable label."""
    if interval < 60:
        return f"{interval}s"
    minutes = interval // 60
    if interval % 60 == 0 and minutes < 60:
        return f"{minutes}m"
    hours = interval // 3600
    if interval % 3600 == 0:
        return f"{hours}h"
    return f"{interval}s"


def _upsert_level(
    levels: list[OrderBookLevel], new: OrderBookLevel,
) -> list[OrderBookLevel]:
    out = [lv for lv in levels if lv.price != new.price]
    if new.size > 0:
        out.append(new)
    out.sort(key=lambda lv: lv.price, reverse=True)
    return out
