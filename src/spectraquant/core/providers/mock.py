"""Mock provider for offline testing."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from spectraquant.core.providers.base import DataProvider
from spectraquant.core.time import ensure_datetime_column


class MockProvider(DataProvider):
    def __init__(self, data: Dict[str, pd.DataFrame] | None = None) -> None:
        self._data = data or {}

    def fetch_daily(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        df = self._data.get(ticker, pd.DataFrame()).copy()
        if df.empty:
            return df
        return ensure_datetime_column(df, "date")

    def fetch_intraday(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        df = self._data.get(ticker, pd.DataFrame()).copy()
        if df.empty:
            return df
        return ensure_datetime_column(df, "date")
