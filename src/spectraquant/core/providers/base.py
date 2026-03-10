"""Provider interface for market data."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Base interface for data providers."""

    @abstractmethod
    def fetch_daily(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_intraday(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        raise NotImplementedError


def get_provider(name: str) -> type[DataProvider]:
    if name == "yfinance":
        from spectraquant.core.providers.yfinance import YfinanceProvider

        return YfinanceProvider
    if name == "mock":
        from spectraquant.core.providers.mock import MockProvider

        return MockProvider
    raise ValueError(f"Unknown provider: {name}")
