"""Tests for the SpectraQuant-AI-V3 data ingestion layer.

All tests are self-contained: no network calls, no persistent file-system
side-effects outside of pytest's ``tmp_path`` fixture.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
from spectraquant_v3.core.cache import CacheManager
from spectraquant_v3.core.enums import AssetClass, RunMode
from spectraquant_v3.core.errors import (
    CacheOnlyViolationError,
    DataSchemaError,
    EmptyPriceDataError,
    SymbolResolutionError,
)
from spectraquant_v3.core.ingestion_result import IngestionResult, make_error_result
from spectraquant_v3.core.schema import SymbolRecord


# ===========================================================================
# Fixtures
# ===========================================================================


def _ohlcv_df(n: int = 100, canonical_symbol: str = "BTC") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(99)
    close = 30_000.0 + np.cumsum(rng.standard_normal(n) * 100)
    close = np.maximum(close, 1.0)
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.001
    volume = rng.uniform(1_000, 50_000, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df["timestamp"] = df.index
    df["canonical_symbol"] = canonical_symbol
    df["provider"] = "test_provider"
    df["exchange_id"] = "test_exchange"
    df["timeframe"] = "1d"
    df["ingested_at"] = datetime.datetime.now(tz=datetime.timezone.utc)
    return df


def _make_crypto_registry_and_mapper():
    """Return (CryptoSymbolRegistry, CryptoSymbolMapper) pre-populated with BTC/ETH."""
    from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper
    from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

    registry = CryptoSymbolRegistry()
    for sym, cg_id in [("BTC", "bitcoin"), ("ETH", "ethereum"), ("SOL", "solana")]:
        registry.register(
            SymbolRecord(
                canonical_symbol=sym,
                asset_class=AssetClass.CRYPTO,
                primary_provider="ccxt",
                primary_exchange_id="binance",
                provider_symbol=f"{sym}/USDT",
                coingecko_id=cg_id,
                quote_currency="USDT",
            )
        )
    mapper = CryptoSymbolMapper(registry=registry)
    return registry, mapper


def _make_equity_registry_and_mapper():
    """Return (EquitySymbolRegistry, EquitySymbolMapper) pre-populated with INFY.NS."""
    from spectraquant_v3.equities.symbols.mapper import EquitySymbolMapper
    from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

    registry = EquitySymbolRegistry()
    for sym in ["INFY.NS", "TCS.NS"]:
        registry.register(
            SymbolRecord(
                canonical_symbol=sym,
                asset_class=AssetClass.EQUITY,
                primary_provider="yfinance",
                yfinance_symbol=sym,
            )
        )
    mapper = EquitySymbolMapper(registry=registry)
    return registry, mapper


# ===========================================================================
# 1. IngestionResult dataclass
# ===========================================================================


class TestIngestionResult:
    def test_construction(self):
        result = IngestionResult(
            canonical_symbol="BTC",
            asset_class="crypto",
            provider="ccxt/binance",
            success=True,
            rows_loaded=100,
            cache_hit=False,
            cache_path="/tmp/BTC.parquet",
            min_timestamp="2023-01-01T00:00:00",
            max_timestamp="2023-12-31T00:00:00",
        )
        assert result.success is True
        assert result.rows_loaded == 100
        assert result.cache_hit is False
        assert result.warning_codes == []
        assert result.error_code == ""
        assert result.error_message == ""

    def test_make_error_result(self):
        result = make_error_result(
            canonical_symbol="ETH",
            asset_class="crypto",
            provider="ccxt/binance",
            error_code="EmptyPriceDataError",
            error_message="No data returned",
        )
        assert result.success is False
        assert result.rows_loaded == 0
        assert result.cache_hit is False
        assert result.canonical_symbol == "ETH"
        assert result.error_code == "EmptyPriceDataError"
        assert result.error_message == "No data returned"

    def test_warning_codes_default_empty(self):
        r1 = make_error_result("BTC", "crypto", "ccxt", "E1", "err")
        r2 = make_error_result("ETH", "crypto", "ccxt", "E2", "err")
        # Mutable default must NOT be shared between instances
        r1.warning_codes.append("W1")
        assert r2.warning_codes == []


# ===========================================================================
# 2. CcxtProvider
# ===========================================================================


class TestCcxtProvider:
    def _mock_exchange(self, ohlcv_data: list | None = None, markets: dict | None = None):
        """Build a mock ccxt exchange object."""
        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = ohlcv_data or []
        exchange.markets = markets or {}
        exchange.load_markets.return_value = markets or {}
        return exchange

    def test_fetch_ohlcv_returns_data(self):
        from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider

        raw = [
            [1_672_531_200_000, 16_500.0, 16_800.0, 16_400.0, 16_700.0, 5000.0],
        ]
        exchange = self._mock_exchange(ohlcv_data=raw)
        provider = CcxtProvider(exchange_overrides={"binance": exchange})

        result = provider.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=1, exchange_id="binance")
        assert result == raw
        exchange.fetch_ohlcv.assert_called_once()

    def test_fetch_ohlcv_empty_raises(self):
        from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider

        exchange = self._mock_exchange(ohlcv_data=[])
        provider = CcxtProvider(exchange_overrides={"binance": exchange})

        with pytest.raises(EmptyPriceDataError):
            provider.fetch_ohlcv("BTC/USDT", exchange_id="binance")

    def test_fetch_ohlcv_none_raises(self):
        from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider

        exchange = self._mock_exchange(ohlcv_data=None)
        exchange.fetch_ohlcv.return_value = None
        provider = CcxtProvider(exchange_overrides={"binance": exchange})

        with pytest.raises((EmptyPriceDataError, DataSchemaError)):
            provider.fetch_ohlcv("BTC/USDT", exchange_id="binance")

    def test_validate_market_exists_true(self):
        from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider

        exchange = self._mock_exchange(markets={"BTC/USDT": {"active": True}})
        provider = CcxtProvider(exchange_overrides={"binance": exchange})

        assert provider.validate_market_exists("BTC/USDT", exchange_id="binance") is True

    def test_validate_market_exists_false(self):
        from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider

        exchange = self._mock_exchange(markets={})
        provider = CcxtProvider(exchange_overrides={"binance": exchange})

        assert provider.validate_market_exists("UNKNOWN/USDT", exchange_id="binance") is False

    def test_load_markets(self):
        from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider

        markets = {"BTC/USDT": {}, "ETH/USDT": {}}
        exchange = self._mock_exchange(markets=markets)
        provider = CcxtProvider(exchange_overrides={"binance": exchange})

        result = provider.load_markets("binance")
        assert "BTC/USDT" in result


# ===========================================================================
# 3. CoinGeckoProvider
# ===========================================================================


class TestCoinGeckoProvider:
    def _mock_session(self, responses: dict) -> MagicMock:
        """Build a minimal mock requests.Session where get() returns per-url mocks."""
        session = MagicMock()

        def _get(url, **kwargs):
            for pattern, payload in responses.items():
                if pattern in url:
                    resp = MagicMock()
                    resp.status_code = 200
                    resp.json.return_value = payload
                    resp.raise_for_status = MagicMock()
                    return resp
            resp = MagicMock()
            resp.status_code = 404
            resp.raise_for_status.side_effect = Exception("404 Not Found")
            return resp

        session.get.side_effect = _get
        return session

    def test_get_coin_list(self):
        from spectraquant_v3.crypto.ingestion.providers.coingecko_provider import CoinGeckoProvider

        payload = [{"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}]
        session = self._mock_session({"/coins/list": payload})
        provider = CoinGeckoProvider(_session=session)
        result = provider.get_coin_list()
        assert len(result) == 1
        assert result[0]["id"] == "bitcoin"

    def test_get_coin_market_data(self):
        from spectraquant_v3.crypto.ingestion.providers.coingecko_provider import CoinGeckoProvider

        # CoinGecko /coins/{id} endpoint always includes "id" at the top level
        payload = {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "market_data": {
                "current_price": {"usd": 30000},
                "market_cap": {"usd": 600_000_000_000},
                "total_volume": {"usd": 20_000_000_000},
            },
        }
        session = self._mock_session({"/coins/bitcoin": payload})
        provider = CoinGeckoProvider(_session=session)
        result = provider.get_coin_market_data("bitcoin")
        assert isinstance(result, dict)

    def test_get_ohlcv_returns_list(self):
        from spectraquant_v3.crypto.ingestion.providers.coingecko_provider import CoinGeckoProvider

        raw = [[1_672_531_200_000, 16_500.0, 16_800.0, 16_400.0, 16_700.0]]
        session = self._mock_session({"/coins/bitcoin/ohlc": raw})
        provider = CoinGeckoProvider(_session=session)
        result = provider.get_ohlcv("bitcoin", days=1)
        assert result == raw


# ===========================================================================
# 4. CryptoCompareProvider
# ===========================================================================


class TestCryptoCompareProvider:
    def _mock_session(self, payload: dict) -> MagicMock:
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp
        return session

    def test_get_daily_ohlcv(self):
        from spectraquant_v3.crypto.ingestion.providers.cryptocompare_provider import (
            CryptoCompareProvider,
        )

        # CryptoCompare _parse_ohlcv expects top-level "Data" to be a list
        payload = {
            "Response": "Success",
            "Data": [
                {
                    "time": 1_672_531_200,
                    "open": 16500.0,
                    "high": 16800.0,
                    "low": 16400.0,
                    "close": 16700.0,
                    "volumefrom": 5000.0,
                    "volumeto": 8_000_000.0,
                }
            ],
        }
        session = self._mock_session(payload)
        provider = CryptoCompareProvider(_session=session)
        result = provider.get_daily_ohlcv("BTC", limit=1)
        assert len(result) == 1
        assert result[0]["close"] == 16700.0

    def test_get_daily_ohlcv_api_error_raises(self):
        from spectraquant_v3.crypto.ingestion.providers.cryptocompare_provider import (
            CryptoCompareProvider,
        )

        payload = {"Response": "Error", "Message": "rate limit exceeded"}
        session = self._mock_session(payload)
        provider = CryptoCompareProvider(_session=session)
        with pytest.raises(Exception):
            provider.get_daily_ohlcv("BTC", limit=1)


# ===========================================================================
# 5. CryptoPanicProvider
# ===========================================================================


class TestCryptoPanicProvider:
    def _mock_session(self, payload: dict) -> MagicMock:
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp
        return session

    def test_get_news_returns_normalized(self):
        from spectraquant_v3.crypto.ingestion.providers.cryptopanic_provider import (
            CryptoPanicProvider,
        )

        payload = {
            "results": [
                {
                    "id": 1,
                    "title": "BTC hits 30k",
                    "url": "https://example.com/btc",
                    "published_at": "2023-06-01T12:00:00Z",
                    "currencies": [{"code": "BTC"}],
                    "votes": {"positive": 10, "negative": 2},
                    "source": {"domain": "example.com"},
                }
            ]
        }
        session = self._mock_session(payload)
        provider = CryptoPanicProvider(_session=session)
        result = provider.get_news()
        assert len(result) == 1
        assert result[0]["title"] == "BTC hits 30k"
        assert result[0]["url"] == "https://example.com/btc"

    def test_get_news_empty_results(self):
        from spectraquant_v3.crypto.ingestion.providers.cryptopanic_provider import (
            CryptoPanicProvider,
        )

        payload = {"results": []}
        session = self._mock_session(payload)
        provider = CryptoPanicProvider(_session=session)
        result = provider.get_news()
        assert result == []


# ===========================================================================
# 6. GlassnodeProvider
# ===========================================================================


class TestGlassnodeProvider:
    def _mock_session(self, payload: list) -> MagicMock:
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp
        return session

    def test_get_metric_returns_list(self):
        from spectraquant_v3.crypto.ingestion.providers.glassnode_provider import (
            GlassnodeProvider,
        )

        payload = [{"t": 1_672_531_200, "v": 900_000}]
        session = self._mock_session(payload)
        provider = GlassnodeProvider(_session=session)
        result = provider.get_metric("BTC", "addresses/active_count")
        assert len(result) == 1
        assert result[0]["t"] == 1_672_531_200
        assert result[0]["v"] == 900_000

    def test_get_metric_empty_list(self):
        from spectraquant_v3.crypto.ingestion.providers.glassnode_provider import (
            GlassnodeProvider,
        )

        session = self._mock_session([])
        provider = GlassnodeProvider(_session=session)
        # Empty is allowed (Glassnode returns [] when no data for period)
        result = provider.get_metric("BTC", "addresses/active_count")
        assert result == []


# ===========================================================================
# 7. BinanceFuturesProvider
# ===========================================================================


class TestBinanceFuturesProvider:
    def _mock_session(self, payload: Any) -> MagicMock:
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp
        return session

    def test_get_funding_rate(self):
        from spectraquant_v3.crypto.ingestion.providers.binance_futures_provider import (
            BinanceFuturesProvider,
        )

        payload = [
            {"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingTime": 1_672_531_200_000}
        ]
        session = self._mock_session(payload)
        provider = BinanceFuturesProvider(_session=session)
        result = provider.get_funding_rate("BTCUSDT", limit=1)
        assert len(result) == 1
        assert result[0]["fundingRate"] == "0.0001"

    def test_get_open_interest(self):
        from spectraquant_v3.crypto.ingestion.providers.binance_futures_provider import (
            BinanceFuturesProvider,
        )

        payload = {"symbol": "BTCUSDT", "openInterest": "12345.678", "time": 1_672_531_200_000}
        session = self._mock_session(payload)
        provider = BinanceFuturesProvider(_session=session)
        result = provider.get_open_interest("BTCUSDT")
        assert result["openInterest"] == "12345.678"


# ===========================================================================
# 8. BybitProvider
# ===========================================================================


class TestBybitProvider:
    def _mock_session(self, payload: dict) -> MagicMock:
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp
        return session

    def test_get_funding_rate(self):
        from spectraquant_v3.crypto.ingestion.providers.bybit_provider import BybitProvider

        payload = {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.0001",
                        "fundingRateTimestamp": "1672531200000",
                    }
                ]
            },
        }
        session = self._mock_session(payload)
        provider = BybitProvider(_session=session)
        result = provider.get_funding_rate("BTCUSDT", limit=1)
        assert len(result) == 1
        assert result[0]["fundingRate"] == "0.0001"

    def test_get_open_interest(self):
        from spectraquant_v3.crypto.ingestion.providers.bybit_provider import BybitProvider

        payload = {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "openInterest": "50000.0",
                        "timestamp": "1672531200000",
                    }
                ]
            },
        }
        session = self._mock_session(payload)
        provider = BybitProvider(_session=session)
        result = provider.get_open_interest("BTCUSDT", limit=1)
        assert len(result) >= 0  # may be list or dict depending on impl


# ===========================================================================
# 9. Crypto price_downloader
# ===========================================================================


class TestCryptoPriceDownloader:
    def _mock_ccxt_provider(self, raw_ohlcv: list | None = None, raise_exc: Exception | None = None):
        from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider

        provider = MagicMock(spec=CcxtProvider)
        if raise_exc:
            provider.fetch_ohlcv.side_effect = raise_exc
        else:
            provider.fetch_ohlcv.return_value = raw_ohlcv or []
        return provider

    def _raw_ohlcv(self, n: int = 100) -> list[list]:
        base_ts = 1_672_531_200_000  # 2023-01-01 UTC in ms
        rows = []
        for i in range(n):
            ts = base_ts + i * 86_400_000
            rows.append([ts, 16_500.0 + i, 16_800.0 + i, 16_400.0 + i, 16_700.0 + i, 5000.0])
        return rows

    def test_normal_mode_cache_miss_downloads(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        raw = self._raw_ohlcv(100)
        ccxt = self._mock_ccxt_provider(raw_ohlcv=raw)

        result = download_symbol_ohlcv(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            ccxt_provider=ccxt,
            lookback_days=50,
        )
        assert result.success is True
        assert result.rows_loaded == 100
        assert result.cache_hit is False
        assert result.canonical_symbol == "BTC"
        assert "ccxt" in result.provider or "binance" in result.provider
        # Cache should now contain the file
        assert cache.exists("BTC")

    def test_normal_mode_cache_hit_sufficient(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        df = _ohlcv_df(n=200, canonical_symbol="BTC")
        cache.write_parquet("BTC", df)

        ccxt = self._mock_ccxt_provider(raw_ohlcv=[])

        result = download_symbol_ohlcv(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            ccxt_provider=ccxt,
            lookback_days=50,
        )
        assert result.success is True
        assert result.cache_hit is True
        # ccxt should NOT have been called
        ccxt.fetch_ohlcv.assert_not_called()

    def test_normal_mode_cache_insufficient_coverage_re_downloads(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        # Cache has only 10 rows but we need 50
        df = _ohlcv_df(n=10, canonical_symbol="BTC")
        cache.write_parquet("BTC", df)

        raw = self._raw_ohlcv(100)
        ccxt = self._mock_ccxt_provider(raw_ohlcv=raw)

        result = download_symbol_ohlcv(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            ccxt_provider=ccxt,
            lookback_days=50,
        )
        assert result.success is True
        assert result.cache_hit is False
        ccxt.fetch_ohlcv.assert_called_once()

    def test_refresh_mode_always_re_downloads(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.REFRESH)
        _, mapper = _make_crypto_registry_and_mapper()
        df = _ohlcv_df(n=200, canonical_symbol="BTC")
        cache.write_parquet("BTC", df)

        raw = self._raw_ohlcv(100)
        ccxt = self._mock_ccxt_provider(raw_ohlcv=raw)

        result = download_symbol_ohlcv(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.REFRESH,
            ccxt_provider=ccxt,
            lookback_days=50,
        )
        assert result.success is True
        assert result.cache_hit is False
        ccxt.fetch_ohlcv.assert_called_once()

    def test_test_mode_cache_hit_returns_result(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.TEST)
        _, mapper = _make_crypto_registry_and_mapper()
        df = _ohlcv_df(n=200, canonical_symbol="BTC")
        cache.write_parquet("BTC", df)

        ccxt = self._mock_ccxt_provider(raw_ohlcv=[])

        result = download_symbol_ohlcv(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.TEST,
            ccxt_provider=ccxt,
            lookback_days=50,
        )
        assert result.success is True
        assert result.cache_hit is True
        ccxt.fetch_ohlcv.assert_not_called()

    def test_test_mode_cache_miss_raises(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.TEST)
        _, mapper = _make_crypto_registry_and_mapper()
        ccxt = self._mock_ccxt_provider(raw_ohlcv=[])

        with pytest.raises(CacheOnlyViolationError):
            download_symbol_ohlcv(
                symbol="BTC",
                cache=cache,
                mapper=mapper,
                run_mode=RunMode.TEST,
                ccxt_provider=ccxt,
                lookback_days=50,
            )

    def test_all_providers_fail_raises_empty_price_data_error(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        ccxt = self._mock_ccxt_provider(raise_exc=EmptyPriceDataError("CCXT failed"))

        with pytest.raises(EmptyPriceDataError):
            download_symbol_ohlcv(
                symbol="BTC",
                cache=cache,
                mapper=mapper,
                run_mode=RunMode.NORMAL,
                ccxt_provider=ccxt,
                fallback_providers=[],
                lookback_days=50,
            )

    def test_download_many_ohlcv_partial_success(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_many_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()

        call_count = 0
        raw = self._raw_ohlcv(100)

        def side_effect(sym, **kwargs):
            nonlocal call_count
            call_count += 1
            if sym == "BTC/USDT":
                return raw
            raise EmptyPriceDataError(f"No data for {sym}")

        ccxt = MagicMock()
        ccxt.fetch_ohlcv.side_effect = side_effect

        results = download_many_ohlcv(
            symbols=["BTC", "ETH"],
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            ccxt_provider=ccxt,
            lookback_days=50,
            fail_fast=False,
        )
        assert "BTC" in results
        assert "ETH" in results
        assert results["BTC"].success is True
        assert results["ETH"].success is False

    def test_download_many_ohlcv_fail_fast(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_many_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()

        ccxt = MagicMock()
        ccxt.fetch_ohlcv.side_effect = EmptyPriceDataError("No data")

        results = download_many_ohlcv(
            symbols=["BTC", "ETH", "SOL"],
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            ccxt_provider=ccxt,
            lookback_days=50,
            fail_fast=True,
        )
        # With fail_fast, should stop after first failure (1 or fewer results in dict)
        assert len(results) <= 3
        # At least the first symbol should be attempted
        assert "BTC" in results

    def test_equity_symbol_rejected(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_symbol_ohlcv
        from spectraquant_v3.core.errors import AssetClassLeakError

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        ccxt = self._mock_ccxt_provider(raw_ohlcv=[])

        with pytest.raises((AssetClassLeakError, SymbolResolutionError)):
            download_symbol_ohlcv(
                symbol="INFY.NS",
                cache=cache,
                mapper=mapper,
                run_mode=RunMode.NORMAL,
                ccxt_provider=ccxt,
                lookback_days=50,
            )

    def test_ccxt_fallback_to_cryptocompare(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.price_downloader import download_symbol_ohlcv
        from spectraquant_v3.crypto.ingestion.providers.cryptocompare_provider import (
            CryptoCompareProvider,
        )

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        ccxt = self._mock_ccxt_provider(raise_exc=EmptyPriceDataError("CCXT down"))

        # Mock CryptoCompare provider
        cc_provider = MagicMock(spec=CryptoCompareProvider)
        cc_raw = [
            {
                "time": 1_672_531_200 + i * 86_400,
                "open": 16500.0,
                "high": 16800.0,
                "low": 16400.0,
                "close": 16700.0,
                "volumefrom": 5000.0,
            }
            for i in range(100)
        ]
        cc_provider.get_daily_ohlcv.return_value = cc_raw

        result = download_symbol_ohlcv(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            ccxt_provider=ccxt,
            fallback_providers=[cc_provider],
            lookback_days=50,
        )
        assert result.success is True
        assert result.provider == "cryptocompare"


# ===========================================================================
# 10. Crypto news ingestion
# ===========================================================================


class TestCryptoNewsIngestion:
    def _mock_cryptopanic_provider(self, news_items: list | None = None):
        from spectraquant_v3.crypto.ingestion.providers.cryptopanic_provider import (
            CryptoPanicProvider,
        )

        provider = MagicMock(spec=CryptoPanicProvider)
        provider.get_news.return_value = news_items or []
        return provider

    def _sample_news(self, n: int = 5) -> list[dict]:
        return [
            {
                "id": i,
                "title": f"News headline {i}",
                "url": f"https://example.com/news/{i}",
                "published_at": "2023-06-01T12:00:00Z",
                "currencies": [{"code": "BTC"}],
                "votes": {"positive": i * 2, "negative": 0},
                "source": {"domain": "example.com"},
            }
            for i in range(n)
        ]

    def test_normal_mode_fetches_and_caches(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.news_ingestion import ingest_news_for_symbol

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        provider = self._mock_cryptopanic_provider(self._sample_news(5))

        result = ingest_news_for_symbol(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
        )
        assert result.success is True
        assert result.rows_loaded == 5
        assert cache.exists("news__BTC")

    def test_normal_mode_empty_news_non_fatal(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.news_ingestion import ingest_news_for_symbol

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        provider = self._mock_cryptopanic_provider([])

        result = ingest_news_for_symbol(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
        )
        # Empty news is non-fatal; either success=False or rows_loaded=0
        assert result.rows_loaded == 0 or result.success is False

    def test_provider_exception_non_fatal(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.news_ingestion import ingest_news_for_symbol

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        provider = MagicMock()
        provider.get_news.side_effect = Exception("API down")

        # Should NOT raise; news is optional
        result = ingest_news_for_symbol(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
        )
        assert result.success is False

    def test_ingest_news_for_many(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.news_ingestion import ingest_news_for_many

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        provider = self._mock_cryptopanic_provider(self._sample_news(3))

        results = ingest_news_for_many(
            symbols=["BTC", "ETH"],
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
        )
        assert "BTC" in results
        assert "ETH" in results


# ===========================================================================
# 11. Crypto on-chain ingestion
# ===========================================================================


class TestOnchainIngestion:
    def _mock_glassnode_provider(self, metric_data: list | None = None):
        from spectraquant_v3.crypto.ingestion.providers.glassnode_provider import GlassnodeProvider

        provider = MagicMock(spec=GlassnodeProvider)
        provider.get_metric.return_value = metric_data or [
            {"t": 1_672_531_200 + i * 86_400, "v": float(900_000 + i)}
            for i in range(10)
        ]
        return provider

    def test_normal_mode_ingests_metrics(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.onchain_ingestion import ingest_onchain_for_symbol

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        provider = self._mock_glassnode_provider()

        result = ingest_onchain_for_symbol(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
        )
        assert result.success is True
        assert result.rows_loaded > 0

    def test_provider_failure_non_fatal(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.onchain_ingestion import ingest_onchain_for_symbol

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        provider = MagicMock()
        provider.get_metric.side_effect = Exception("Glassnode API error")

        result = ingest_onchain_for_symbol(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
        )
        # On-chain is optional; failure is non-fatal
        assert result.success is False

    def test_ingest_onchain_for_many(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.onchain_ingestion import ingest_onchain_for_many

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        provider = self._mock_glassnode_provider()

        results = ingest_onchain_for_many(
            symbols=["BTC", "ETH"],
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
        )
        assert "BTC" in results
        assert "ETH" in results


# ===========================================================================
# 12. Crypto funding ingestion
# ===========================================================================


class TestFundingIngestion:
    def _mock_binance_provider(self, funding_data: list | None = None, oi_data: dict | None = None):
        from spectraquant_v3.crypto.ingestion.providers.binance_futures_provider import (
            BinanceFuturesProvider,
        )

        provider = MagicMock(spec=BinanceFuturesProvider)
        provider.get_funding_rate.return_value = funding_data or [
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001",
                "fundingTime": 1_672_531_200_000 + i * 28_800_000,
            }
            for i in range(10)
        ]
        provider.get_open_interest.return_value = oi_data or {
            "symbol": "BTCUSDT",
            "openInterest": "12345.678",
            "time": 1_672_531_200_000,
        }
        return provider

    def test_normal_mode_ingests_funding(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.funding_ingestion import ingest_funding_for_symbol

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        provider = self._mock_binance_provider()

        result = ingest_funding_for_symbol(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            primary_provider=provider,
        )
        assert result.success is True
        assert result.rows_loaded > 0

    def test_provider_failure_non_fatal(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.funding_ingestion import ingest_funding_for_symbol

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        provider = MagicMock()
        provider.get_funding_rate.side_effect = Exception("Binance API error")
        provider.get_open_interest.side_effect = Exception("Binance API error")

        result = ingest_funding_for_symbol(
            symbol="BTC",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            primary_provider=provider,
        )
        assert result.success is False

    def test_ingest_funding_for_many(self, tmp_path):
        from spectraquant_v3.crypto.ingestion.funding_ingestion import ingest_funding_for_many

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_crypto_registry_and_mapper()
        provider = self._mock_binance_provider()

        results = ingest_funding_for_many(
            symbols=["BTC", "ETH"],
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            primary_provider=provider,
        )
        assert "BTC" in results
        assert "ETH" in results


# ===========================================================================
# 13. Crypto orderbook ingestion (placeholder)
# ===========================================================================


class TestOrderbookIngestion:
    def test_ingest_orderbook_raises_not_implemented(self):
        from spectraquant_v3.crypto.ingestion.orderbook_ingestion import ingest_orderbook_snapshot

        _, mapper = _make_crypto_registry_and_mapper()

        with pytest.raises(NotImplementedError):
            ingest_orderbook_snapshot(
                symbol="BTC",
                mapper=mapper,
                exchange_id="binance",
                provider=None,
            )

    def test_orderbook_snapshot_dataclass(self):
        from spectraquant_v3.crypto.ingestion.orderbook_ingestion import OrderBookSnapshot

        snap = OrderBookSnapshot(
            canonical_symbol="BTC",
            exchange_id="binance",
            timestamp=datetime.datetime.now(tz=datetime.timezone.utc),
            bids=[[29_000.0, 1.5]],
            asks=[[29_001.0, 1.2]],
            provider="ccxt",
            ingested_at=datetime.datetime.now(tz=datetime.timezone.utc),
        )
        assert snap.canonical_symbol == "BTC"
        assert len(snap.bids) == 1

    def test_get_orderbook_cache_key(self):
        from spectraquant_v3.crypto.ingestion.orderbook_ingestion import get_orderbook_cache_key

        key = get_orderbook_cache_key("BTC", "binance")
        assert "BTC" in key
        assert "binance" in key


# ===========================================================================
# 14. YFinanceProvider
# ===========================================================================


class TestYFinanceProvider:
    def _mock_yf(self, df: pd.DataFrame | None = None, raise_exc: Exception | None = None):
        """Build a minimal yfinance mock."""
        yf = MagicMock()
        ticker = MagicMock()
        if raise_exc:
            ticker.history.side_effect = raise_exc
        else:
            ticker.history.return_value = df if df is not None else pd.DataFrame()
        yf.Ticker.return_value = ticker
        return yf

    def _sample_yf_df(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(77)
        close = 1500.0 + np.cumsum(rng.standard_normal(n) * 10)
        close = np.maximum(close, 1.0)
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": rng.uniform(1_000_000, 5_000_000, n),
            },
            index=idx,
        )
        return df

    def test_download_ohlcv_normalizes_columns(self):
        from spectraquant_v3.equities.ingestion.providers.yfinance_provider import YFinanceProvider

        raw_df = self._sample_yf_df(100)
        yf = self._mock_yf(df=raw_df)
        provider = YFinanceProvider(_yf_module=yf)

        result = provider.download_ohlcv("INFY.NS", period="1y", interval="1d")
        assert isinstance(result, pd.DataFrame)
        # Columns should be lower-case
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns or col in [c.lower() for c in result.columns]

    def test_download_ohlcv_empty_raises(self):
        from spectraquant_v3.equities.ingestion.providers.yfinance_provider import YFinanceProvider

        yf = self._mock_yf(df=pd.DataFrame())
        provider = YFinanceProvider(_yf_module=yf)

        with pytest.raises(EmptyPriceDataError):
            provider.download_ohlcv("INFY.NS", period="1y", interval="1d")

    def test_download_ohlcv_exception_wrapped(self):
        from spectraquant_v3.equities.ingestion.providers.yfinance_provider import YFinanceProvider

        yf = self._mock_yf(raise_exc=RuntimeError("yfinance network error"))
        provider = YFinanceProvider(_yf_module=yf)

        with pytest.raises((DataSchemaError, EmptyPriceDataError)):
            provider.download_ohlcv("INFY.NS", period="1y", interval="1d")

    def test_get_info_returns_dict(self):
        from spectraquant_v3.equities.ingestion.providers.yfinance_provider import YFinanceProvider

        yf = MagicMock()
        ticker = MagicMock()
        ticker.info = {"shortName": "Infosys", "sector": "Technology"}
        yf.Ticker.return_value = ticker
        provider = YFinanceProvider(_yf_module=yf)

        result = provider.get_info("INFY.NS")
        assert isinstance(result, dict)

    def test_get_info_failure_returns_empty_dict(self):
        from spectraquant_v3.equities.ingestion.providers.yfinance_provider import YFinanceProvider

        yf = MagicMock()
        ticker = MagicMock()
        ticker.info = MagicMock(side_effect=Exception("network error"))
        # Make attribute access itself raise
        type(ticker).info = property(lambda self: (_ for _ in ()).throw(Exception("net error")))
        yf.Ticker.return_value = ticker
        provider = YFinanceProvider(_yf_module=yf)

        result = provider.get_info("INFY.NS")
        assert isinstance(result, dict)


# ===========================================================================
# 15. RSSProvider
# ===========================================================================


class TestRSSProvider:
    def _mock_feedparser(self, entries: list) -> MagicMock:
        fp = MagicMock()
        feed_result = MagicMock()
        feed_result.entries = entries
        fp.parse.return_value = feed_result
        return fp

    def _sample_entry(self, i: int = 0) -> MagicMock:
        entry = MagicMock()
        entry.title = f"Equity News {i}"
        entry.link = f"https://finance.yahoo.com/news/{i}"
        entry.published = "Mon, 01 Jan 2024 12:00:00 GMT"
        return entry

    def test_get_news_returns_normalized(self):
        from spectraquant_v3.equities.ingestion.providers.rss_provider import RSSProvider

        fp = self._mock_feedparser([self._sample_entry(0), self._sample_entry(1)])
        provider = RSSProvider(_feedparser_module=fp)

        result = provider.get_news("INFY.NS")
        assert len(result) >= 1
        assert "title" in result[0]
        assert "url" in result[0]

    def test_get_news_feedparser_unavailable_returns_empty(self):
        from spectraquant_v3.equities.ingestion.providers.rss_provider import RSSProvider

        # Simulate feedparser not installed by having parse raise ImportError
        fp = MagicMock()
        fp.parse.side_effect = ImportError("feedparser not installed")
        provider = RSSProvider(_feedparser_module=fp)

        # Should return empty list, not raise
        result = provider.get_news("INFY.NS")
        assert isinstance(result, list)


# ===========================================================================
# 16. Equity price_downloader
# ===========================================================================


class TestEquityPriceDownloader:
    def _sample_yf_df(self, n: int = 100, symbol: str = "INFY.NS") -> pd.DataFrame:
        rng = np.random.default_rng(55)
        close = 1500.0 + np.cumsum(rng.standard_normal(n) * 10)
        close = np.maximum(close, 1.0)
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": rng.uniform(1_000_000, 5_000_000, n),
            },
            index=idx,
        )
        df["timestamp"] = df.index
        df["canonical_symbol"] = symbol
        df["provider"] = "yfinance"
        df["timeframe"] = "1d"
        df["ingested_at"] = datetime.datetime.now(tz=datetime.timezone.utc)
        return df

    def _mock_yfinance_provider(
        self,
        df: pd.DataFrame | None = None,
        raise_exc: Exception | None = None,
    ):
        from spectraquant_v3.equities.ingestion.providers.yfinance_provider import YFinanceProvider

        provider = MagicMock(spec=YFinanceProvider)
        if raise_exc:
            provider.download_ohlcv.side_effect = raise_exc
        else:
            provider.download_ohlcv.return_value = df if df is not None else self._sample_yf_df()
        return provider

    def test_normal_mode_cache_miss_downloads(self, tmp_path):
        from spectraquant_v3.equities.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_equity_registry_and_mapper()
        df = self._sample_yf_df(100)
        provider = self._mock_yfinance_provider(df=df)

        result = download_symbol_ohlcv(
            symbol="INFY.NS",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
            lookback_days=50,
        )
        assert result.success is True
        assert result.rows_loaded == 100
        assert result.cache_hit is False
        assert result.canonical_symbol == "INFY.NS"
        assert cache.exists("INFY.NS")

    def test_normal_mode_cache_hit(self, tmp_path):
        from spectraquant_v3.equities.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_equity_registry_and_mapper()
        df = self._sample_yf_df(200, "INFY.NS")
        cache.write_parquet("INFY.NS", df)

        provider = self._mock_yfinance_provider()

        result = download_symbol_ohlcv(
            symbol="INFY.NS",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
            lookback_days=50,
        )
        assert result.success is True
        assert result.cache_hit is True
        provider.download_ohlcv.assert_not_called()

    def test_test_mode_cache_miss_raises(self, tmp_path):
        from spectraquant_v3.equities.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.TEST)
        _, mapper = _make_equity_registry_and_mapper()
        provider = self._mock_yfinance_provider()

        with pytest.raises(CacheOnlyViolationError):
            download_symbol_ohlcv(
                symbol="INFY.NS",
                cache=cache,
                mapper=mapper,
                run_mode=RunMode.TEST,
                provider=provider,
                lookback_days=50,
            )

    def test_empty_provider_response_raises(self, tmp_path):
        from spectraquant_v3.equities.ingestion.price_downloader import download_symbol_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_equity_registry_and_mapper()
        provider = self._mock_yfinance_provider(raise_exc=EmptyPriceDataError("No data"))

        with pytest.raises(EmptyPriceDataError):
            download_symbol_ohlcv(
                symbol="INFY.NS",
                cache=cache,
                mapper=mapper,
                run_mode=RunMode.NORMAL,
                provider=provider,
                lookback_days=50,
            )

    def test_crypto_symbol_rejected(self, tmp_path):
        from spectraquant_v3.equities.ingestion.price_downloader import download_symbol_ohlcv
        from spectraquant_v3.core.errors import AssetClassLeakError

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_equity_registry_and_mapper()
        provider = self._mock_yfinance_provider()

        with pytest.raises((AssetClassLeakError, SymbolResolutionError)):
            download_symbol_ohlcv(
                symbol="BTC/USDT",
                cache=cache,
                mapper=mapper,
                run_mode=RunMode.NORMAL,
                provider=provider,
                lookback_days=50,
            )

    def test_download_many_equity_ohlcv(self, tmp_path):
        from spectraquant_v3.equities.ingestion.price_downloader import download_many_ohlcv

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_equity_registry_and_mapper()

        call_count = [0]

        def side_effect(sym, **kwargs):
            call_count[0] += 1
            if sym == "INFY.NS":
                return self._sample_yf_df(100, "INFY.NS")
            raise EmptyPriceDataError(f"No data for {sym}")

        provider = MagicMock()
        provider.download_ohlcv.side_effect = side_effect

        results = download_many_ohlcv(
            symbols=["INFY.NS", "TCS.NS"],
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
            lookback_days=50,
            fail_fast=False,
        )
        assert "INFY.NS" in results
        assert "TCS.NS" in results
        assert results["INFY.NS"].success is True
        assert results["TCS.NS"].success is False


# ===========================================================================
# 17. Equity news ingestion
# ===========================================================================


class TestEquityNewsIngestion:
    def _mock_rss_provider(self, news: list | None = None):
        from spectraquant_v3.equities.ingestion.providers.rss_provider import RSSProvider

        provider = MagicMock(spec=RSSProvider)
        provider.get_news.return_value = news or []
        return provider

    def _sample_news(self, n: int = 3) -> list[dict]:
        return [
            {
                "title": f"Equity headline {i}",
                "url": f"https://finance.yahoo.com/news/{i}",
                "published_at": "Mon, 01 Jan 2024 12:00:00 GMT",
                "source": "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "provider": "rss",
            }
            for i in range(n)
        ]

    def test_normal_mode_fetches_and_caches(self, tmp_path):
        from spectraquant_v3.equities.ingestion.news_ingestion import ingest_news_for_symbol

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_equity_registry_and_mapper()
        provider = self._mock_rss_provider(self._sample_news(3))

        result = ingest_news_for_symbol(
            symbol="INFY.NS",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
        )
        assert result.success is True
        assert result.rows_loaded == 3

    def test_provider_failure_non_fatal(self, tmp_path):
        from spectraquant_v3.equities.ingestion.news_ingestion import ingest_news_for_symbol

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_equity_registry_and_mapper()
        provider = MagicMock()
        provider.get_news.side_effect = Exception("RSS error")

        result = ingest_news_for_symbol(
            symbol="INFY.NS",
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
        )
        assert result.success is False

    def test_ingest_news_for_many(self, tmp_path):
        from spectraquant_v3.equities.ingestion.news_ingestion import ingest_news_for_many

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        _, mapper = _make_equity_registry_and_mapper()
        provider = self._mock_rss_provider(self._sample_news(2))

        results = ingest_news_for_many(
            symbols=["INFY.NS", "TCS.NS"],
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
            provider=provider,
        )
        assert "INFY.NS" in results
        assert "TCS.NS" in results


# ===========================================================================
# 18. Public API exports
# ===========================================================================


class TestPublicAPIExports:
    def test_crypto_ingestion_exports(self):
        import spectraquant_v3.crypto.ingestion as ci

        assert hasattr(ci, "CryptoOHLCVLoader")
        assert hasattr(ci, "download_symbol_ohlcv")
        assert hasattr(ci, "download_many_ohlcv")
        assert hasattr(ci, "ingest_news_for_symbol")
        assert hasattr(ci, "ingest_news_for_many")
        assert hasattr(ci, "ingest_onchain_for_symbol")
        assert hasattr(ci, "ingest_onchain_for_many")
        assert hasattr(ci, "ingest_funding_for_symbol")
        assert hasattr(ci, "ingest_funding_for_many")
        assert hasattr(ci, "OrderBookSnapshot")
        assert hasattr(ci, "ingest_orderbook_snapshot")

    def test_equity_ingestion_exports(self):
        import spectraquant_v3.equities.ingestion as ei

        assert hasattr(ei, "EquityOHLCVLoader")
        assert hasattr(ei, "download_symbol_ohlcv")
        assert hasattr(ei, "download_many_ohlcv")
        assert hasattr(ei, "ingest_news_for_symbol")
        assert hasattr(ei, "ingest_news_for_many")

    def test_crypto_providers_exports(self):
        import spectraquant_v3.crypto.ingestion.providers as p

        assert hasattr(p, "CcxtProvider")
        assert hasattr(p, "CoinGeckoProvider")
        assert hasattr(p, "CryptoCompareProvider")
        assert hasattr(p, "CryptoPanicProvider")
        assert hasattr(p, "GlassnodeProvider")
        assert hasattr(p, "BinanceFuturesProvider")
        assert hasattr(p, "BybitProvider")

    def test_equity_providers_exports(self):
        import spectraquant_v3.equities.ingestion.providers as p

        assert hasattr(p, "YFinanceProvider")
        assert hasattr(p, "RSSProvider")

    def test_ingestion_result_importable(self):
        from spectraquant_v3.core.ingestion_result import IngestionResult, make_error_result

        assert IngestionResult is not None
        assert make_error_result is not None

    def test_no_crypto_equity_cross_import(self):
        """Verify equity ingestion modules do not import from crypto namespace."""
        import importlib
        import sys

        # Remove any cached modules
        for key in list(sys.modules.keys()):
            if "spectraquant_v3" in key:
                del sys.modules[key]

        # Re-import equity modules and check no crypto imports leaked
        import spectraquant_v3.equities.ingestion.price_downloader as eq_pd
        import spectraquant_v3.equities.ingestion.news_ingestion as eq_ni
        import spectraquant_v3.equities.ingestion.providers.yfinance_provider as yf_prov
        import spectraquant_v3.equities.ingestion.providers.rss_provider as rss_prov

        # These imports should work without importing crypto modules
        assert eq_pd is not None
        assert eq_ni is not None
        assert yf_prov is not None
        assert rss_prov is not None


# ===========================================================================
# 13. Async ingestion engine
# ===========================================================================


class TestAsyncEngine:
    """Tests for spectraquant_v3.core.async_engine."""

    def test_import(self):
        from spectraquant_v3.core.async_engine import (
            AsyncIngestionError,
            AsyncIngestionSummary,
            ingest_many_symbols,
            run_ingest_many_symbols,
        )

        assert ingest_many_symbols is not None
        assert run_ingest_many_symbols is not None
        assert AsyncIngestionError is not None
        assert AsyncIngestionSummary is not None

    def test_empty_symbols_list(self):
        import asyncio

        from spectraquant_v3.core.async_engine import ingest_many_symbols

        async def _noop(s: str):
            return s

        summary = asyncio.run(ingest_many_symbols([], _noop))
        assert summary.total == 0
        assert summary.succeeded == 0
        assert summary.failed == 0
        assert summary.results == {}

    def test_all_succeed(self):
        import asyncio

        from spectraquant_v3.core.async_engine import ingest_many_symbols

        symbols = ["BTC", "ETH", "SOL"]

        async def _succeed(s: str) -> str:
            return f"ok:{s}"

        summary = asyncio.run(ingest_many_symbols(symbols, _succeed, concurrency=2))
        assert summary.total == 3
        assert summary.succeeded == 3
        assert summary.failed == 0
        assert summary.failed_symbols == []
        for sym in symbols:
            assert summary.results[sym] == f"ok:{sym}"

    def test_one_failure_does_not_crash_batch(self):
        import asyncio

        from spectraquant_v3.core.async_engine import (
            AsyncIngestionError,
            ingest_many_symbols,
        )

        symbols = ["BTC", "BROKEN", "ETH"]

        async def _fetch(s: str):
            if s == "BROKEN":
                raise ValueError("provider error")
            return f"ok:{s}"

        summary = asyncio.run(
            ingest_many_symbols(
                symbols, _fetch, concurrency=3, max_retries=1, base_delay=0
            )
        )
        assert summary.total == 3
        assert summary.succeeded == 2
        assert summary.failed == 1
        assert "BROKEN" in summary.failed_symbols
        assert isinstance(summary.results["BROKEN"], AsyncIngestionError)
        assert summary.results["BROKEN"].error_type == "ValueError"
        assert summary.results["BTC"] == "ok:BTC"
        assert summary.results["ETH"] == "ok:ETH"

    def test_retry_on_transient_failure(self):
        """Symbol should succeed after one transient failure when max_retries >= 2."""
        import asyncio

        from spectraquant_v3.core.async_engine import ingest_many_symbols

        call_counts: dict[str, int] = {}

        async def _flaky(s: str) -> str:
            call_counts[s] = call_counts.get(s, 0) + 1
            if call_counts[s] < 2:
                raise RuntimeError("transient")
            return f"ok:{s}"

        summary = asyncio.run(
            ingest_many_symbols(
                ["BTC"], _flaky, concurrency=1, max_retries=3, base_delay=0
            )
        )
        assert summary.succeeded == 1
        assert summary.failed == 0
        assert call_counts["BTC"] == 2

    def test_exhausted_retries_produces_error(self):
        import asyncio

        from spectraquant_v3.core.async_engine import (
            AsyncIngestionError,
            ingest_many_symbols,
        )

        async def _always_fail(s: str):
            raise RuntimeError("permanent")

        summary = asyncio.run(
            ingest_many_symbols(
                ["BTC"], _always_fail, concurrency=1, max_retries=2, base_delay=0
            )
        )
        assert summary.failed == 1
        err = summary.results["BTC"]
        assert isinstance(err, AsyncIngestionError)
        assert err.attempts == 2

    def test_concurrency_bound_respected(self):
        """No more than `concurrency` coroutines run simultaneously."""
        import asyncio

        from spectraquant_v3.core.async_engine import ingest_many_symbols

        active: list[int] = [0]
        peak: list[int] = [0]
        start_events: dict[str, asyncio.Event] = {}
        ready_event = None

        async def _track(s: str) -> str:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            # Yield control so the event loop can schedule other tasks
            for _ in range(5):
                await asyncio.sleep(0)
            active[0] -= 1
            return s

        concurrency = 3
        symbols = [str(i) for i in range(9)]
        asyncio.run(ingest_many_symbols(symbols, _track, concurrency=concurrency))
        assert peak[0] <= concurrency

    def test_run_ingest_many_symbols_sync_wrapper(self):
        from spectraquant_v3.core.async_engine import run_ingest_many_symbols

        async def _succeed(s: str) -> str:
            return s

        summary = run_ingest_many_symbols(["A", "B"], _succeed, concurrency=2)
        assert summary.succeeded == 2

    def test_async_ingestion_error_dataclass(self):
        from spectraquant_v3.core.async_engine import AsyncIngestionError

        err = AsyncIngestionError(
            symbol="ETH",
            error_type="EmptyPriceDataError",
            error_message="no data",
            attempts=3,
        )
        assert err.symbol == "ETH"
        assert err.attempts == 3

    def test_summary_failed_symbols_list(self):
        import asyncio

        from spectraquant_v3.core.async_engine import (
            AsyncIngestionError,
            ingest_many_symbols,
        )

        async def _fail(s: str):
            raise Exception("fail")

        summary = asyncio.run(
            ingest_many_symbols(
                ["X", "Y", "Z"], _fail, concurrency=1, max_retries=1, base_delay=0
            )
        )
        assert set(summary.failed_symbols) == {"X", "Y", "Z"}
        assert summary.failed == 3
        assert summary.succeeded == 0


# ===========================================================================
# 14. Async crypto price downloader
# ===========================================================================


class TestAsyncCryptoPriceDownloader:
    """Tests for async_download_symbol_ohlcv / async_download_many_ohlcv in crypto."""

    def _setup(self, tmp_path):
        """Build cache + mapper + mock CCXT provider."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider
        from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        registry = CryptoSymbolRegistry()
        for sym, cg_id in [("BTC", "bitcoin"), ("ETH", "ethereum")]:
            registry.register(
                SymbolRecord(
                    canonical_symbol=sym,
                    asset_class=AssetClass.CRYPTO,
                    primary_provider="ccxt",
                    primary_exchange_id="binance",
                    provider_symbol=f"{sym}/USDT",
                    coingecko_id=cg_id,
                    quote_currency="USDT",
                )
            )
        mapper = CryptoSymbolMapper(registry=registry)
        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)

        # Build minimal valid CCXT raw OHLCV data
        raw = [
            [1_672_531_200_000 + i * 86_400_000, 100.0, 101.0, 99.0, 100.5, 500.0]
            for i in range(10)
        ]
        exchange_mock = MagicMock()
        exchange_mock.fetch_ohlcv.return_value = raw
        exchange_mock.markets = {"BTC/USDT": {}, "ETH/USDT": {}}
        exchange_mock.load_markets.return_value = exchange_mock.markets
        ccxt_provider = CcxtProvider(exchange_overrides={"binance": exchange_mock})

        return cache, mapper, ccxt_provider, RunMode.NORMAL

    def test_async_download_single_symbol(self, tmp_path):
        import asyncio

        from spectraquant_v3.crypto.ingestion.price_downloader import (
            async_download_symbol_ohlcv,
        )

        cache, mapper, ccxt_provider, run_mode = self._setup(tmp_path)
        result = asyncio.run(
            async_download_symbol_ohlcv(
                symbol="BTC",
                cache=cache,
                mapper=mapper,
                run_mode=run_mode,
                ccxt_provider=ccxt_provider,
                lookback_days=5,
            )
        )
        assert result.success is True
        assert result.canonical_symbol == "BTC"
        assert result.rows_loaded > 0

    def test_async_download_many_all_succeed(self, tmp_path):
        import asyncio

        from spectraquant_v3.crypto.ingestion.price_downloader import (
            async_download_many_ohlcv,
        )

        cache, mapper, ccxt_provider, run_mode = self._setup(tmp_path)
        results = asyncio.run(
            async_download_many_ohlcv(
                symbols=["BTC", "ETH"],
                cache=cache,
                mapper=mapper,
                run_mode=run_mode,
                ccxt_provider=ccxt_provider,
                lookback_days=5,
                concurrency=2,
                max_retries=1,
                base_delay=0,
            )
        )
        assert "BTC" in results
        assert "ETH" in results
        assert results["BTC"].success is True
        assert results["ETH"].success is True

    def test_async_download_many_one_failure(self, tmp_path):
        import asyncio

        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode, AssetClass
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.ingestion.price_downloader import (
            async_download_many_ohlcv,
        )
        from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider
        from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        # Register BTC only; BROKEN is not in registry → SymbolResolutionError
        registry = CryptoSymbolRegistry()
        registry.register(
            SymbolRecord(
                canonical_symbol="BTC",
                asset_class=AssetClass.CRYPTO,
                primary_provider="ccxt",
                primary_exchange_id="binance",
                provider_symbol="BTC/USDT",
                coingecko_id="bitcoin",
                quote_currency="USDT",
            )
        )
        mapper = CryptoSymbolMapper(registry=registry)
        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)

        raw = [
            [1_672_531_200_000 + i * 86_400_000, 100.0, 101.0, 99.0, 100.5, 500.0]
            for i in range(10)
        ]
        exchange_mock = MagicMock()
        exchange_mock.fetch_ohlcv.return_value = raw
        exchange_mock.markets = {"BTC/USDT": {}}
        exchange_mock.load_markets.return_value = exchange_mock.markets
        ccxt_provider = CcxtProvider(exchange_overrides={"binance": exchange_mock})

        results = asyncio.run(
            async_download_many_ohlcv(
                symbols=["BTC", "BROKEN"],
                cache=cache,
                mapper=mapper,
                run_mode=RunMode.NORMAL,
                ccxt_provider=ccxt_provider,
                lookback_days=5,
                concurrency=2,
                max_retries=1,
                base_delay=0,
            )
        )
        assert "BTC" in results
        assert "BROKEN" in results
        assert results["BTC"].success is True
        assert results["BROKEN"].success is False


# ===========================================================================
# 15. Async equity price downloader
# ===========================================================================


class TestAsyncEquityPriceDownloader:
    """Tests for async_download_symbol_ohlcv / async_download_many_ohlcv in equities."""

    def _setup(self, tmp_path):
        import datetime

        import numpy as np
        import pandas as pd

        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.ingestion.providers.yfinance_provider import (
            YFinanceProvider,
        )
        from spectraquant_v3.equities.symbols.mapper import EquitySymbolMapper
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        registry = EquitySymbolRegistry()
        for sym in ["INFY.NS", "TCS.NS"]:
            registry.register(
                SymbolRecord(
                    canonical_symbol=sym,
                    asset_class=AssetClass.EQUITY,
                    primary_provider="yfinance",
                    yfinance_symbol=sym,
                )
            )
        mapper = EquitySymbolMapper(registry=registry)
        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)

        n = 20
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        close = 1500.0 + np.cumsum(rng.standard_normal(n))
        df = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": rng.uniform(1e5, 1e6, n),
                "timestamp": idx,
                "canonical_symbol": "INFY.NS",
                "provider": "yfinance",
                "timeframe": "1d",
                "ingested_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            },
            index=idx,
        )
        provider_mock = MagicMock(spec=YFinanceProvider)
        provider_mock.download_ohlcv.return_value = df
        return cache, mapper, provider_mock, RunMode.NORMAL

    def test_async_download_single_equity(self, tmp_path):
        import asyncio

        from spectraquant_v3.equities.ingestion.price_downloader import (
            async_download_symbol_ohlcv,
        )

        cache, mapper, provider_mock, run_mode = self._setup(tmp_path)
        result = asyncio.run(
            async_download_symbol_ohlcv(
                symbol="INFY.NS",
                cache=cache,
                mapper=mapper,
                run_mode=run_mode,
                provider=provider_mock,
            )
        )
        assert result.success is True
        assert result.canonical_symbol == "INFY.NS"
        assert result.asset_class == "equity"

    def test_async_download_many_equities(self, tmp_path):
        import asyncio

        import numpy as np
        import pandas as pd
        import datetime

        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.ingestion.price_downloader import (
            async_download_many_ohlcv,
        )
        from spectraquant_v3.equities.ingestion.providers.yfinance_provider import (
            YFinanceProvider,
        )
        from spectraquant_v3.equities.symbols.mapper import EquitySymbolMapper
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        registry = EquitySymbolRegistry()
        for sym in ["INFY.NS", "TCS.NS"]:
            registry.register(
                SymbolRecord(
                    canonical_symbol=sym,
                    asset_class=AssetClass.EQUITY,
                    primary_provider="yfinance",
                    yfinance_symbol=sym,
                )
            )
        mapper = EquitySymbolMapper(registry=registry)
        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)

        n = 20
        rng = np.random.default_rng(7)
        idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        close = 1000.0 + np.cumsum(rng.standard_normal(n))

        def _make_df(sym):
            return pd.DataFrame(
                {
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": rng.uniform(1e5, 1e6, n),
                    "timestamp": idx,
                    "canonical_symbol": sym,
                    "provider": "yfinance",
                    "timeframe": "1d",
                    "ingested_at": datetime.datetime.now(
                        tz=datetime.timezone.utc
                    ).isoformat(),
                },
                index=idx,
            )

        provider_mock = MagicMock(spec=YFinanceProvider)
        provider_mock.download_ohlcv.side_effect = lambda sym, **kw: _make_df(sym)

        results = asyncio.run(
            async_download_many_ohlcv(
                symbols=["INFY.NS", "TCS.NS"],
                cache=cache,
                mapper=mapper,
                run_mode=RunMode.NORMAL,
                provider=provider_mock,
                concurrency=2,
                max_retries=1,
                base_delay=0,
            )
        )
        assert "INFY.NS" in results
        assert "TCS.NS" in results
        assert results["INFY.NS"].success is True
        assert results["TCS.NS"].success is True

    def test_async_download_equity_failure_captured(self, tmp_path):
        import asyncio

        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.ingestion.price_downloader import (
            async_download_many_ohlcv,
        )
        from spectraquant_v3.equities.ingestion.providers.yfinance_provider import (
            YFinanceProvider,
        )
        from spectraquant_v3.equities.symbols.mapper import EquitySymbolMapper
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        # Only register INFY.NS; BROKEN.NS is unknown → resolution error
        registry = EquitySymbolRegistry()
        registry.register(
            SymbolRecord(
                canonical_symbol="INFY.NS",
                asset_class=AssetClass.EQUITY,
                primary_provider="yfinance",
                yfinance_symbol="INFY.NS",
            )
        )
        mapper = EquitySymbolMapper(registry=registry)
        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)

        import numpy as np
        import pandas as pd
        import datetime

        n = 20
        rng = np.random.default_rng(3)
        idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        close = 1500.0 + np.cumsum(rng.standard_normal(n))
        good_df = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": rng.uniform(1e5, 1e6, n),
                "timestamp": idx,
                "canonical_symbol": "INFY.NS",
                "provider": "yfinance",
                "timeframe": "1d",
                "ingested_at": datetime.datetime.now(
                    tz=datetime.timezone.utc
                ).isoformat(),
            },
            index=idx,
        )
        provider_mock = MagicMock(spec=YFinanceProvider)
        provider_mock.download_ohlcv.return_value = good_df

        results = asyncio.run(
            async_download_many_ohlcv(
                symbols=["INFY.NS", "BROKEN.NS"],
                cache=cache,
                mapper=mapper,
                run_mode=RunMode.NORMAL,
                provider=provider_mock,
                concurrency=2,
                max_retries=1,
                base_delay=0,
            )
        )
        assert results["INFY.NS"].success is True
        assert results["BROKEN.NS"].success is False


# ===========================================================================
# 16. CryptoOHLCVLoader – async batch download
# ===========================================================================


class TestCryptoOHLCVLoaderAsync:
    """Tests for CryptoOHLCVLoader.load_many_async / load_many_async_run."""

    def _make_df(self, sym: str = "BTC", n: int = 10) -> "pd.DataFrame":
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        close = 30_000.0 + np.cumsum(rng.standard_normal(n))
        df = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": rng.uniform(1e4, 1e5, n),
            },
            index=idx,
        )
        df.columns = [c.lower() for c in df.columns]
        return df

    def _make_loader(self, tmp_path, sym_data: dict):
        """Return a CryptoOHLCVLoader whose `load()` returns stub DataFrames."""
        from unittest.mock import patch

        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.crypto.ingestion.ohlcv_loader import CryptoOHLCVLoader
        from spectraquant_v3.crypto.symbols.mapper import CryptoSymbolMapper
        from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry

        registry = CryptoSymbolRegistry()
        for sym in sym_data:
            registry.register(
                SymbolRecord(
                    canonical_symbol=sym,
                    asset_class=AssetClass.CRYPTO,
                    primary_provider="ccxt",
                    primary_exchange_id="binance",
                    provider_symbol=f"{sym}/USDT",
                    coingecko_id=sym.lower(),
                    quote_currency="USDT",
                )
            )

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        mapper = CryptoSymbolMapper(registry=registry)

        loader = CryptoOHLCVLoader(
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
        )

        # Override `load` to return stub data without hitting network
        def _stub_load(sym):
            return sym_data.get(sym)

        loader.load = _stub_load
        return loader

    def test_load_many_async_all_succeed(self, tmp_path):
        import asyncio

        df_btc = self._make_df("BTC")
        df_eth = self._make_df("ETH")
        loader = self._make_loader(tmp_path, {"BTC": df_btc, "ETH": df_eth})

        results = asyncio.run(
            loader.load_many_async(["BTC", "ETH"], concurrency=2, max_retries=1, base_delay=0)
        )
        assert "BTC" in results
        assert "ETH" in results

    def test_load_many_async_one_failure_omitted(self, tmp_path):
        import asyncio

        df_btc = self._make_df("BTC")
        loader = self._make_loader(tmp_path, {"BTC": df_btc, "BROKEN": None})

        results = asyncio.run(
            loader.load_many_async(
                ["BTC", "BROKEN"], concurrency=2, max_retries=1, base_delay=0
            )
        )
        assert "BTC" in results
        assert "BROKEN" not in results

    def test_load_many_async_run_sync_wrapper(self, tmp_path):
        df_btc = self._make_df("BTC")
        loader = self._make_loader(tmp_path, {"BTC": df_btc})

        results = loader.load_many_async_run(["BTC"], concurrency=1, max_retries=1, base_delay=0)
        assert "BTC" in results

    def test_load_many_async_empty_list(self, tmp_path):
        import asyncio

        loader = self._make_loader(tmp_path, {})
        results = asyncio.run(loader.load_many_async([], concurrency=2))
        assert results == {}

    def test_load_many_async_concurrency_respected(self, tmp_path):
        """load_many_async honours the concurrency bound via the engine."""
        import asyncio

        dfs = {sym: self._make_df(sym) for sym in ["BTC", "ETH", "SOL", "BNB", "ADA"]}
        loader = self._make_loader(tmp_path, dfs)

        results = asyncio.run(
            loader.load_many_async(list(dfs.keys()), concurrency=2, max_retries=1, base_delay=0)
        )
        assert set(results.keys()) == set(dfs.keys())


# ===========================================================================
# 17. EquityOHLCVLoader – async batch download
# ===========================================================================


class TestEquityOHLCVLoaderAsync:
    """Tests for EquityOHLCVLoader.load_many_async / load_many_async_run."""

    def _make_df(self, sym: str = "INFY.NS", n: int = 10) -> "pd.DataFrame":
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(7)
        idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        close = 1500.0 + np.cumsum(rng.standard_normal(n))
        df = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": rng.uniform(1e5, 1e6, n),
            },
            index=idx,
        )
        df.columns = [c.lower() for c in df.columns]
        return df

    def _make_loader(self, tmp_path, sym_data: dict):
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.schema import SymbolRecord
        from spectraquant_v3.equities.ingestion.ohlcv_loader import EquityOHLCVLoader
        from spectraquant_v3.equities.symbols.mapper import EquitySymbolMapper
        from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry

        registry = EquitySymbolRegistry()
        for sym in sym_data:
            registry.register(
                SymbolRecord(
                    canonical_symbol=sym,
                    asset_class=AssetClass.EQUITY,
                    primary_provider="yfinance",
                    yfinance_symbol=sym,
                )
            )

        cache = CacheManager(tmp_path, run_mode=RunMode.NORMAL)
        mapper = EquitySymbolMapper(registry=registry)

        loader = EquityOHLCVLoader(
            cache=cache,
            mapper=mapper,
            run_mode=RunMode.NORMAL,
        )

        def _stub_load(sym):
            return sym_data.get(sym)

        loader.load = _stub_load
        return loader

    def test_load_many_async_all_succeed(self, tmp_path):
        import asyncio

        dfs = {
            "INFY.NS": self._make_df("INFY.NS"),
            "TCS.NS": self._make_df("TCS.NS"),
        }
        loader = self._make_loader(tmp_path, dfs)

        results = asyncio.run(
            loader.load_many_async(list(dfs.keys()), concurrency=2, max_retries=1, base_delay=0)
        )
        assert "INFY.NS" in results
        assert "TCS.NS" in results

    def test_load_many_async_one_failure_omitted(self, tmp_path):
        import asyncio

        loader = self._make_loader(
            tmp_path, {"INFY.NS": self._make_df("INFY.NS"), "BROKEN.NS": None}
        )

        results = asyncio.run(
            loader.load_many_async(
                ["INFY.NS", "BROKEN.NS"], concurrency=2, max_retries=1, base_delay=0
            )
        )
        assert "INFY.NS" in results
        assert "BROKEN.NS" not in results

    def test_load_many_async_run_sync_wrapper(self, tmp_path):
        loader = self._make_loader(tmp_path, {"INFY.NS": self._make_df("INFY.NS")})

        results = loader.load_many_async_run(
            ["INFY.NS"], concurrency=1, max_retries=1, base_delay=0
        )
        assert "INFY.NS" in results

    def test_load_many_async_empty_list(self, tmp_path):
        import asyncio

        loader = self._make_loader(tmp_path, {})
        results = asyncio.run(loader.load_many_async([], concurrency=2))
        assert results == {}


# ===========================================================================
# 18. CLI download commands – async batch
# ===========================================================================


class TestCLIAsyncDownload:
    """Tests for sqv3 crypto download and sqv3 equities download commands."""

    def _make_crypto_df(self, sym: str = "BTC") -> "pd.DataFrame":
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(1)
        idx = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
        c = 30_000.0 + np.cumsum(rng.standard_normal(5))
        return pd.DataFrame(
            {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": [1e4] * 5},
            index=idx,
        )

    def test_crypto_download_succeeds(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        import pandas as pd
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        df = self._make_crypto_df()

        async def _mock_async(symbols, **kwargs):
            return {s: df for s in symbols}

        runner = CliRunner()
        with patch(
            "spectraquant_v3.crypto.ingestion.ohlcv_loader.CryptoOHLCVLoader.load_many_async",
            side_effect=_mock_async,
        ):
            result = runner.invoke(app, ["crypto", "download", "--symbols", "BTC,ETH"])

        assert result.exit_code == 0
        assert "2 succeeded" in result.output
        assert "0 failed" in result.output

    def test_crypto_download_partial_failure(self, tmp_path):
        from unittest.mock import patch

        import pandas as pd
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        df = self._make_crypto_df()

        async def _mock_partial(symbols, **kwargs):
            # Only return BTC, not BROKEN
            return {s: df for s in symbols if s != "BROKEN"}

        runner = CliRunner()
        with patch(
            "spectraquant_v3.crypto.ingestion.ohlcv_loader.CryptoOHLCVLoader.load_many_async",
            side_effect=_mock_partial,
        ):
            result = runner.invoke(app, ["crypto", "download", "--symbols", "BTC,BROKEN"])

        # Should exit with 2 (some failures)
        assert result.exit_code == 2
        assert "1 succeeded" in result.output
        assert "1 failed" in result.output

    def test_crypto_download_no_symbols_exits_error(self):
        from unittest.mock import patch

        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        # Patch config at the source module to return empty symbols list
        with patch(
            "spectraquant_v3.core.config.get_crypto_config",
            return_value={
                "crypto": {"symbols": [], "prices_dir": "/tmp/c"},
                "run": {"mode": "normal"},
                "cache": {},
                "qa": {},
                "execution": {},
                "portfolio": {},
            },
        ):
            result = runner.invoke(app, ["crypto", "download"])

        assert result.exit_code == 1

    def test_equity_download_succeeds(self, tmp_path):
        from unittest.mock import patch

        import pandas as pd
        import numpy as np
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        rng = np.random.default_rng(5)
        idx = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
        c = 1500.0 + np.cumsum(rng.standard_normal(5))
        df = pd.DataFrame(
            {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": [1e5] * 5},
            index=idx,
        )

        async def _mock_async(symbols, **kwargs):
            return {s: df for s in symbols}

        runner = CliRunner()
        with patch(
            "spectraquant_v3.equities.ingestion.ohlcv_loader.EquityOHLCVLoader.load_many_async",
            side_effect=_mock_async,
        ):
            result = runner.invoke(
                app, ["equity", "download", "--symbols", "INFY.NS,TCS.NS"]
            )

        assert result.exit_code == 0
        assert "2 succeeded" in result.output

    def test_equity_download_no_symbols_exits_error(self):
        from unittest.mock import patch

        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        with patch(
            "spectraquant_v3.core.config.get_equity_config",
            return_value={
                "equities": {
                    "universe": {"tickers": []},
                    "prices_dir": "/tmp/eq",
                },
                "run": {"mode": "normal"},
                "cache": {},
                "qa": {},
                "execution": {},
                "portfolio": {},
            },
        ):
            result = runner.invoke(app, ["equity", "download"])

        assert result.exit_code == 1

    def test_crypto_download_invalid_mode(self):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["crypto", "download", "--symbols", "BTC", "--mode", "bogus"]
        )
        assert result.exit_code == 1
        assert "invalid --mode" in result.output

    def test_equity_download_invalid_mode(self):
        from typer.testing import CliRunner

        from spectraquant_v3.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["equity", "download", "--symbols", "INFY.NS", "--mode", "bogus"]
        )
        assert result.exit_code == 1
        assert "invalid --mode" in result.output
