"""yfinance OHLCV provider for SpectraQuant-AI-V3 equity ingestion.

This module wraps yfinance so that the rest of the pipeline never touches the
library directly.  All raw yfinance exceptions are caught and re-raised as
:class:`~spectraquant_v3.core.errors.DataSchemaError` or
:class:`~spectraquant_v3.core.errors.EmptyPriceDataError`.

This module must NEVER import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from spectraquant_v3.core.errors import DataSchemaError, EmptyPriceDataError
from spectraquant_v3.core.time import normalize_datetime_frame

logger = logging.getLogger(__name__)

# Minimum columns expected in a raw yfinance OHLCV response (lower-cased).
_REQUIRED_OHLCV_COLS: frozenset[str] = frozenset({"open", "high", "low", "close", "volume"})


class YFinanceProvider:
    """Downloads OHLCV data and metadata from Yahoo Finance via yfinance.

    Args:
        _yf_module: Optional yfinance module injected for testing.  When
            ``None`` (the default) the real ``yfinance`` package is imported
            lazily on first use.
    """

    def __init__(self, _yf_module: Any = None) -> None:
        self._yf = _yf_module

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_yf(self) -> Any:
        """Return the yfinance module, importing lazily if needed."""
        if self._yf is not None:
            return self._yf
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError as exc:
            raise DataSchemaError(
                "yfinance is not installed. "
                "Install it with: pip install yfinance"
            ) from exc
        return yf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_ohlcv(
        self,
        yf_symbol: str,
        period: str = "5y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download OHLCV history for *yf_symbol* from Yahoo Finance.

        Column names are normalised to lower-case and the DatetimeIndex is
        reset so that the date is available as a plain ``"timestamp"`` column.

        Args:
            yf_symbol: yfinance-compatible ticker, e.g. ``"INFY.NS"``.
            period:    yfinance history period string, e.g. ``"5y"`` or ``"1y"``.
            interval:  yfinance interval string, e.g. ``"1d"``.

        Returns:
            DataFrame with lower-case OHLCV columns and a ``"timestamp"``
            column containing the bar open time.

        Raises:
            EmptyPriceDataError: When yfinance returns an empty DataFrame.
            DataSchemaError:     When required OHLCV columns are absent, or
                                 when yfinance raises an unexpected error.
        """
        yf = self._get_yf()

        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
        except Exception as exc:
            raise DataSchemaError(
                f"yfinance download failed for '{yf_symbol}': {exc}"
            ) from exc

        if df is None or df.empty:
            raise EmptyPriceDataError(
                f"yfinance returned an empty DataFrame for '{yf_symbol}'. "
                "The symbol may be delisted, misspelled, or outside the "
                f"requested period='{period}'."
            )

        # Normalize column names to lowercase.
        df.columns = [c.lower() for c in df.columns]

        # Verify required columns are present before resetting the index.
        actual_cols = set(df.columns)
        missing = _REQUIRED_OHLCV_COLS - actual_cols
        if missing:
            raise DataSchemaError(
                f"yfinance response for '{yf_symbol}' is missing required "
                f"OHLCV columns: {sorted(missing)}. "
                f"Present columns: {sorted(actual_cols)}."
            )

        # Reset index to promote the DatetimeIndex to a plain column, then
        # rename it to the canonical 'timestamp' name.
        df = df.reset_index()
        # yfinance may name the index 'Date' or 'Datetime' depending on interval.
        for candidate in ("Date", "Datetime", "date", "datetime", "index"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "timestamp"})
                break

        df = normalize_datetime_frame(
            df,
            label=f"yfinance OHLCV '{yf_symbol}'",
            timestamp_column="timestamp",
        )

        logger.debug(
            "YFinanceProvider: downloaded %d rows for '%s' (period=%s, interval=%s)",
            len(df),
            yf_symbol,
            period,
            interval,
        )
        return df

    def get_info(self, yf_symbol: str) -> dict[str, Any]:
        """Return the yfinance ``Ticker.info`` dict for *yf_symbol*.

        Returns an empty dict on any failure so that callers can treat
        metadata as optional without special-casing.

        Args:
            yf_symbol: yfinance-compatible ticker, e.g. ``"INFY.NS"``.

        Returns:
            Ticker info dict, or ``{}`` on failure.
        """
        try:
            yf = self._get_yf()
            return yf.Ticker(yf_symbol).info or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "YFinanceProvider.get_info: failed to fetch info for '%s': %s",
                yf_symbol,
                exc,
            )
            return {}
