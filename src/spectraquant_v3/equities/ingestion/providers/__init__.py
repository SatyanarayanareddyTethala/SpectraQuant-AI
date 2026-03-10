"""Equity ingestion provider implementations for SpectraQuant-AI-V3.

Each provider is responsible for a single data source.  Providers are thin
wrappers — they handle transport and normalization only, and must never
write to the cache or make run-mode decisions.
"""

from spectraquant_v3.equities.ingestion.providers.yfinance_provider import YFinanceProvider
from spectraquant_v3.equities.ingestion.providers.rss_provider import RSSProvider

__all__ = ["YFinanceProvider", "RSSProvider"]
