"""Crypto ingestion sub-package.

Public API
----------
``CryptoOHLCVLoader``
    Cache-managed OHLCV loader for canonical crypto tickers.
    Loads from parquet cache (cache-first) or downloads via yfinance.

``download_symbol_ohlcv`` / ``download_many_ohlcv``
    Multi-provider OHLCV downloader (CCXT primary, CryptoCompare / CoinGecko
    fallbacks) with cache-first semantics and strict run-mode enforcement.

``ingest_news_for_symbol`` / ``ingest_news_for_many``
    Non-fatal CryptoPanic news ingestion.

``ingest_onchain_for_symbol`` / ``ingest_onchain_for_many``
    Non-fatal Glassnode on-chain metric ingestion.

``ingest_funding_for_symbol`` / ``ingest_funding_for_many``
    Non-fatal funding rate + open interest ingestion (Binance / Bybit).

``OrderBookSnapshot`` / ``ingest_orderbook_snapshot``
    Orderbook snapshot types and interface placeholder (websocket TBD).
"""

from spectraquant_v3.crypto.ingestion.funding_ingestion import (
    ingest_funding_for_many,
    ingest_funding_for_symbol,
)
from spectraquant_v3.crypto.ingestion.news_ingestion import (
    ingest_news_for_many,
    ingest_news_for_symbol,
)
from spectraquant_v3.crypto.ingestion.ohlcv_loader import CryptoOHLCVLoader
from spectraquant_v3.crypto.ingestion.onchain_ingestion import (
    ingest_onchain_for_many,
    ingest_onchain_for_symbol,
)
from spectraquant_v3.crypto.ingestion.orderbook_ingestion import (
    OrderBookSnapshot,
    ingest_orderbook_snapshot,
)
from spectraquant_v3.crypto.ingestion.price_downloader import (
    download_many_ohlcv,
    download_symbol_ohlcv,
)

__all__ = [
    # Legacy loader (preserved)
    "CryptoOHLCVLoader",
    # OHLCV downloader
    "download_symbol_ohlcv",
    "download_many_ohlcv",
    # News
    "ingest_news_for_symbol",
    "ingest_news_for_many",
    # On-chain
    "ingest_onchain_for_symbol",
    "ingest_onchain_for_many",
    # Funding
    "ingest_funding_for_symbol",
    "ingest_funding_for_many",
    # Orderbook
    "OrderBookSnapshot",
    "ingest_orderbook_snapshot",
]
