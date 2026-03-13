"""yfinance OHLCV provider for equity research mode.

This provider wraps yfinance for research-grade historical data.
It is designed so that the provider can later be swapped for a more
official data vendor without changing any downstream logic.

Design contract:
- Input: list of yfinance-style symbols (e.g. ["INFY.NS", "TCS.NS"])
- Output: dict[symbol, pd.DataFrame] with columns [open, high, low, close, volume]
- Index: UTC DatetimeIndex
- test_mode=True: raises CacheOnlyViolationError instead of making network calls
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from spectraquant.core.enums import RunMode
from spectraquant.core.errors import CacheOnlyViolationError

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and ensure UTC index."""
    df.columns = [str(c).lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


class YFinanceEquityProvider:
    """Fetch equity OHLCV data via yfinance.

    Args:
        cache_dir: Base directory for parquet cache files.
        run_mode: Controls cache/download behaviour.
            - NORMAL: cache-first, download on miss.
            - TEST:   cache-only, raise on miss.
            - REFRESH: bypass cache, force redownload.
        start_date: Earliest date for historical data.
        end_date: Latest date for historical data (default: today).
        period_days: Alternative to start_date; lookback in days.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/equities/prices",
        run_mode: RunMode = RunMode.NORMAL,
        start_date: date | None = None,
        end_date: date | None = None,
        period_days: int = 365 * 5,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._run_mode = run_mode
        self._period_days = period_days
        today = datetime.now(timezone.utc).date()
        self._end_date = end_date or today
        self._start_date = start_date or (today - timedelta(days=period_days))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for each symbol in *symbols*.

        Returns:
            dict mapping symbol → DataFrame.  Symbols with no data are
            omitted (with a warning).
        """
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self._fetch_one(sym)
            if df is not None and not df.empty:
                result[sym] = df
            else:
                logger.warning("No OHLCV data returned for equity symbol %r", sym)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("\\", "_")
        return self._cache_dir / f"{safe}.parquet"

    def _fetch_one(self, symbol: str) -> pd.DataFrame | None:
        cache_path = self._cache_path(symbol)

        # REFRESH mode: skip cache
        if self._run_mode != RunMode.REFRESH and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                return _normalise_columns(df)
            except Exception as exc:  # pragma: no cover
                logger.warning("Cache read failed for %r: %s", symbol, exc)

        # TEST mode: fail loudly on cache miss
        if self._run_mode == RunMode.TEST:
            raise CacheOnlyViolationError(symbol, data_type="equity OHLCV")

        # Download via yfinance
        return self._download(symbol)

    def _download(self, symbol: str) -> pd.DataFrame | None:
        try:
            import yfinance as yf  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for equity data download. "
                "Install with: pip install yfinance"
            ) from exc

        logger.info("Downloading equity OHLCV for %r via yfinance", symbol)
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self._start_date.isoformat(),
                end=self._end_date.isoformat(),
                auto_adjust=True,
            )
            if df is None or df.empty:
                logger.warning("yfinance returned empty data for %r", symbol)
                return None
            df = _normalise_columns(df)
            # Persist to cache
            self._write_cache(symbol, df)
            return df
        except Exception as exc:
            logger.error("yfinance download failed for %r: %s", symbol, exc)
            return None

    def _write_cache(self, symbol: str, df: pd.DataFrame) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self._cache_path(symbol), engine="pyarrow")
        except Exception as exc:  # pragma: no cover
            logger.warning("Cache write failed for %r: %s", symbol, exc)
