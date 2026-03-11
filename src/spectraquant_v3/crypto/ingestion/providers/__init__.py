"""Crypto ingestion providers sub-package for SpectraQuant-AI-V3.

Each provider encapsulates a single external data source and exposes a clean,
mockable interface.  All providers raise :exc:`~spectraquant_v3.core.errors.SpectraQuantError`
subclasses on failure — silent errors are forbidden.

Available providers
-------------------
``CcxtProvider``
    OHLCV and market data via the ccxt multi-exchange library.
    Supports: binance, coinbase, kraken (and any other ccxt exchange).

``CoinGeckoProvider``
    Coin metadata, market data, and OHLCV from the CoinGecko public API.

``CryptoCompareProvider``
    Daily and hourly OHLCV from the CryptoCompare min-API.

``CryptoPanicProvider``
    Curated crypto news from the CryptoPanic API.

``GlassnodeProvider``
    On-chain metrics (addresses, transactions, fees, …) from Glassnode.

``BinanceFuturesProvider``
    Funding rates and open interest from Binance USD-M Futures.

``BybitProvider``
    Funding rates and open interest from Bybit linear perpetuals.
"""

from __future__ import annotations

from spectraquant_v3.crypto.ingestion.providers.binance_futures_provider import (
    BinanceFuturesProvider,
)
from spectraquant_v3.crypto.ingestion.providers.bybit_provider import BybitProvider
from spectraquant_v3.crypto.ingestion.providers.ccxt_provider import CcxtProvider
from spectraquant_v3.crypto.ingestion.providers.coingecko_provider import CoinGeckoProvider
from spectraquant_v3.crypto.ingestion.providers.cryptocompare_provider import (
    CryptoCompareProvider,
)
from spectraquant_v3.crypto.ingestion.providers.cryptopanic_provider import CryptoPanicProvider
from spectraquant_v3.crypto.ingestion.providers.glassnode_provider import GlassnodeProvider
from spectraquant_v3.crypto.ingestion.providers.perplexity_provider import PerplexityNewsProvider

__all__ = [
    "CcxtProvider",
    "CoinGeckoProvider",
    "CryptoCompareProvider",
    "CryptoPanicProvider",
    "GlassnodeProvider",
    "BinanceFuturesProvider",
    "BybitProvider",
    "PerplexityNewsProvider",
]
