"""Provider abstraction layer for market data and news ingestion."""
from __future__ import annotations

from spectraquant.providers.interfaces import NewsProvider, PriceProvider
from spectraquant.providers.newsapi_adapter import NewsAPIProvider
from spectraquant.providers.router import MultiProviderRouter
from spectraquant.providers.yfinance_adapter import YFinancePriceProvider

__all__ = [
    "PriceProvider",
    "NewsProvider",
    "YFinancePriceProvider",
    "NewsAPIProvider",
    "MultiProviderRouter",
]
