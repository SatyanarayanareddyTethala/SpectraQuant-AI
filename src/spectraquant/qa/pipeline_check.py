"""Dataset and price data validation helpers."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from spectraquant.core.time import ensure_datetime_column
from spectraquant.data.normalize import normalize_price_columns, normalize_price_frame

logger = logging.getLogger(__name__)


def check_dataset_integrity(X, y, meta_df) -> None:
    """Validate dataset arrays and associated metadata."""

    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert isinstance(y, np.ndarray), "y must be a numpy array"
    assert X.ndim == 2, "X must be 2D"
    assert y.ndim == 1, "y must be 1D"
    assert len(X) == len(y), "X and y must have the same length"
    assert np.isfinite(X).all(), "X contains NaNs or infinities"
    assert np.isfinite(y).all(), "y contains NaNs or infinities"

    assert isinstance(meta_df, pd.DataFrame), "meta_df must be a DataFrame"
    assert not meta_df.empty, "meta_df is empty"
    required_cols = {"ticker", "date", "Close"}
    assert required_cols.issubset(meta_df.columns), "meta_df missing required columns"
    meta_df = ensure_datetime_column(meta_df, "date")
    assert pd.api.types.is_datetime64tz_dtype(meta_df["date"]), "meta_df date column must be timezone-aware"

    n_tickers = meta_df["ticker"].nunique(dropna=True)
    logger.info(
        "Dataset integrity OK: X shape=%s, y length=%s, tickers=%s, features=%s",
        X.shape,
        len(y),
        n_tickers,
        X.shape[1],
    )


def check_price_data(universe_data: Dict[str, pd.DataFrame], min_rows: int = 30) -> None:
    """Validate cached price data for each ticker."""

    assert universe_data, "price data is empty"

    for ticker, df in universe_data.items():
        assert isinstance(df, pd.DataFrame), f"Price data for {ticker} must be DataFrame"
        assert not df.empty, f"Price data for {ticker} is empty"
        normalized = normalize_price_columns(df, ticker)
        normalized = normalize_price_frame(normalized)
        assert not normalized.empty, f"Price data for {ticker} is empty after normalization"
        assert len(normalized) >= min_rows, f"Price data for {ticker} has fewer than {min_rows} rows"

        if "close" not in normalized.columns:
            raise AssertionError(f"No Close column for {ticker} after normalization")

        close_series = pd.to_numeric(normalized["close"], errors="coerce")
        assert close_series.notna().all(), f"Close column for {ticker} contains NaNs"
        assert np.isfinite(close_series.to_numpy()).all(), f"Close column for {ticker} has non-finite values"

    logger.info("Price data integrity OK for %s tickers (min_rows=%s)", len(universe_data), min_rows)
