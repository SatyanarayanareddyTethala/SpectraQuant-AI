"""Equity ingestion sub-package.

Public API
----------
``EquityOHLCVLoader``
    Cache-managed OHLCV loader for canonical equity tickers.
    Loads from parquet cache (cache-first) or downloads via yfinance.
``download_symbol_ohlcv``
    Download OHLCV for a single canonical equity ticker; returns IngestionResult.
``download_many_ohlcv``
    Batch OHLCV download; returns dict[symbol, IngestionResult].
``ingest_news_for_symbol``
    Ingest RSS news for a single canonical equity ticker; returns IngestionResult.
``ingest_news_for_many``
    Batch news ingestion; returns dict[symbol, IngestionResult].
"""

from spectraquant_v3.equities.ingestion.news_ingestion import (
    ingest_news_for_many,
    ingest_news_for_symbol,
)
from spectraquant_v3.equities.ingestion.ohlcv_loader import EquityOHLCVLoader
from spectraquant_v3.equities.ingestion.price_downloader import (
    download_many_ohlcv,
    download_symbol_ohlcv,
)

__all__ = [
    "EquityOHLCVLoader",
    "download_symbol_ohlcv",
    "download_many_ohlcv",
    "ingest_news_for_symbol",
    "ingest_news_for_many",
]
