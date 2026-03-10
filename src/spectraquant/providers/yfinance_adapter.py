"""YFinance price provider adapter."""
from __future__ import annotations

import logging

import pandas as pd

from spectraquant.core.providers.yfinance import YfinanceProvider
from spectraquant.providers.interfaces import PriceProvider

logger = logging.getLogger(__name__)


class YFinancePriceProvider(PriceProvider):
    """Adapter for YFinance data provider."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize YFinance provider adapter.
        
        Args:
            config: Configuration dictionary for the provider
        """
        self._provider = YfinanceProvider(config)
        self._name = "yfinance"

    def fetch_daily(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch daily price data for a ticker."""
        try:
            return self._provider.fetch_daily(ticker, period, interval)
        except Exception as exc:
            logger.error("YFinance daily fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()

    def fetch_intraday(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch intraday price data for a ticker."""
        try:
            return self._provider.fetch_intraday(ticker, period, interval)
        except Exception as exc:
            logger.error("YFinance intraday fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()

    def is_healthy(self) -> bool:
        """Check if YFinance provider is healthy."""
        try:
            from spectraquant.core.providers.yfinance import provider_health_summary
            
            health = provider_health_summary()
            if health.get("calls", 0) == 0:
                return True
            
            success_rate = health.get("success", 0) / max(health.get("calls", 1), 1)
            return success_rate > 0.5
        except Exception as exc:
            logger.warning("Health check failed: %s", exc)
            return False

    def get_name(self) -> str:
        """Get the name of this provider."""
        return self._name
