"""Tests for candle aggregation correctness."""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from spectraquant.crypto.exchange.coinbase_ws import CandleAggregator, Trade


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Temporary output directory for parquet files."""
    return tmp_path / "candles"


@pytest.fixture
def aggregator(tmp_output: Path) -> CandleAggregator:
    """CandleAggregator with 60-second (1m) interval."""
    return CandleAggregator(intervals=[60], output_dir=tmp_output)


def _make_trade(symbol: str, price: float, size: float, ts: datetime) -> Trade:
    return Trade(
        symbol=symbol,
        price=price,
        size=size,
        side="buy",
        timestamp=ts,
        trade_id="",
    )


class TestCandleAggregation:
    """Verify OHLCV candle building from synthetic trades."""

    def test_single_candle(self, aggregator: CandleAggregator) -> None:
        """Trades within the same minute produce one candle on flush."""
        base = datetime(2025, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        trades = [
            _make_trade("BTC-USD", 50000.0, 0.5, base),
            _make_trade("BTC-USD", 50100.0, 0.3, base.replace(second=20)),
            _make_trade("BTC-USD", 49900.0, 0.2, base.replace(second=40)),
        ]
        closed = []
        for t in trades:
            closed.extend(aggregator.ingest(t))
        # No candle closed yet (all same minute)
        assert len(closed) == 0

        # Force flush
        flushed = aggregator.flush_all()
        assert len(flushed) == 1
        candle = flushed[0]
        assert candle.symbol == "BTC-USD"
        assert candle.open == 50000.0
        assert candle.high == 50100.0
        assert candle.low == 49900.0
        assert candle.close == 49900.0
        assert abs(candle.volume - 1.0) < 1e-9
        assert candle.trade_count == 3

    def test_candle_close_on_new_minute(self, aggregator: CandleAggregator) -> None:
        """A trade in the next minute closes the previous candle."""
        t1 = datetime(2025, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
        t3 = datetime(2025, 1, 1, 12, 1, 5, tzinfo=timezone.utc)  # next minute

        aggregator.ingest(_make_trade("ETH-USD", 3000.0, 1.0, t1))
        aggregator.ingest(_make_trade("ETH-USD", 3050.0, 2.0, t2))
        closed = aggregator.ingest(_make_trade("ETH-USD", 3100.0, 0.5, t3))

        assert len(closed) == 1
        candle = closed[0]
        assert candle.open == 3000.0
        assert candle.high == 3050.0
        assert candle.close == 3050.0
        assert candle.volume == 3.0
        assert candle.trade_count == 2

    def test_parquet_output(self, aggregator: CandleAggregator, tmp_output: Path) -> None:
        """Closed candles are written to parquet with UTC DatetimeIndex."""
        t1 = datetime(2025, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 1, 12, 1, 5, tzinfo=timezone.utc)

        aggregator.ingest(_make_trade("BTC-USD", 50000.0, 1.0, t1))
        aggregator.ingest(_make_trade("BTC-USD", 50100.0, 0.5, t2))

        # Find written parquet
        parquets = list(tmp_output.glob("*.parquet"))
        assert len(parquets) >= 1

        df = pd.read_parquet(parquets[0])
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None  # UTC
        assert "open" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_multiple_symbols(self, aggregator: CandleAggregator) -> None:
        """Candles are tracked independently per symbol."""
        t1 = datetime(2025, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        aggregator.ingest(_make_trade("BTC-USD", 50000.0, 1.0, t1))
        aggregator.ingest(_make_trade("ETH-USD", 3000.0, 5.0, t1))

        flushed = aggregator.flush_all()
        symbols = {c.symbol for c in flushed}
        assert "BTC-USD" in symbols
        assert "ETH-USD" in symbols

    def test_utc_bucket_alignment(self, aggregator: CandleAggregator) -> None:
        """Candle timestamps align to minute boundaries in UTC."""
        t = datetime(2025, 6, 15, 8, 23, 45, tzinfo=timezone.utc)
        aggregator.ingest(_make_trade("SOL-USD", 150.0, 10.0, t))
        flushed = aggregator.flush_all()
        assert len(flushed) == 1
        # Should align to 08:23:00
        assert flushed[0].timestamp.second == 0
        assert flushed[0].timestamp.minute == 23
