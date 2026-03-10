"""Crypto derivatives feature engineering – funding rate, open interest & basis."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype(float)


def _has_columns(frame: pd.DataFrame, cols: set[str]) -> bool:
    return cols.issubset(frame.columns)


# ------------------------------------------------------------------
# Funding-rate features
# ------------------------------------------------------------------


def compute_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute funding rate, annualized funding, z-score, and momentum.

    Requires a *funding_rate* column.  Skips gracefully if missing.
    """
    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]

    if not _has_columns(frame, {"funding_rate"}):
        logger.info("funding_rate column missing – skipping funding features")
        return pd.DataFrame(index=frame.index)

    fr = _to_numeric(frame["funding_rate"])

    # Annualized funding (3 funding periods per day × 365 days)
    annualized_funding = fr * 3 * 365

    # Z-score over a 30-period rolling window
    fr_mean = fr.rolling(30).mean()
    fr_std = fr.rolling(30).std()
    fr_zscore = (fr - fr_mean) / fr_std.replace(0, np.nan)

    # Momentum: difference between short and long rolling means
    fr_momentum = fr.rolling(7).mean() - fr.rolling(30).mean()

    return pd.DataFrame(
        {
            "funding_rate": fr,
            "annualized_funding": annualized_funding,
            "funding_rate_zscore": fr_zscore,
            "funding_rate_momentum": fr_momentum,
        },
        index=frame.index,
    )


# ------------------------------------------------------------------
# Open-interest features
# ------------------------------------------------------------------


def compute_oi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute open-interest change, OI/volume ratio, OI z-score, and long/short ratio features.

    Requires *open_interest*.  *volume* and *long_short_ratio* are optional
    and used when present.
    """
    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]

    if not _has_columns(frame, {"open_interest"}):
        logger.info("open_interest column missing – skipping OI features")
        return pd.DataFrame(index=frame.index)

    oi = _to_numeric(frame["open_interest"])

    features: dict[str, pd.Series] = {}

    # OI percentage change
    features["oi_change"] = oi.pct_change()

    # OI / volume ratio
    if _has_columns(frame, {"volume"}):
        vol = _to_numeric(frame["volume"])
        features["oi_volume_ratio"] = oi / vol.replace(0, np.nan)

    # OI z-score (30-period)
    oi_mean = oi.rolling(30).mean()
    oi_std = oi.rolling(30).std()
    features["oi_zscore"] = (oi - oi_mean) / oi_std.replace(0, np.nan)

    # Long/short ratio features (if column exists)
    if _has_columns(frame, {"long_short_ratio"}):
        lsr = _to_numeric(frame["long_short_ratio"])
        features["long_short_ratio"] = lsr
        features["long_short_ratio_change"] = lsr.pct_change()
        lsr_mean = lsr.rolling(20).mean()
        lsr_std = lsr.rolling(20).std()
        features["long_short_ratio_zscore"] = (lsr - lsr_mean) / lsr_std.replace(0, np.nan)

    return pd.DataFrame(features, index=frame.index)


# ------------------------------------------------------------------
# Basis features
# ------------------------------------------------------------------


def compute_basis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute spot-futures basis, annualized basis, and basis momentum.

    Requires *close* (spot price) and *futures_price*.
    """
    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]

    if not _has_columns(frame, {"close", "futures_price"}):
        logger.info("close/futures_price columns missing – skipping basis features")
        return pd.DataFrame(index=frame.index)

    spot = _to_numeric(frame["close"])
    futures = _to_numeric(frame["futures_price"])

    spot_safe = spot.replace(0, np.nan)

    basis = (futures - spot) / spot_safe
    annualized_basis = basis * 365  # simple annualization
    basis_momentum = basis.rolling(7).mean() - basis.rolling(30).mean()

    return pd.DataFrame(
        {
            "basis": basis,
            "annualized_basis": annualized_basis,
            "basis_momentum": basis_momentum,
        },
        index=frame.index,
    )


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------


def compute_derivatives_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derivatives features and return a single DataFrame."""
    logger.info("Computing derivatives features (%d rows)", len(df))

    parts = [
        compute_funding_features(df),
        compute_oi_features(df),
        compute_basis_features(df),
    ]
    result = pd.concat(parts, axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


__all__ = [
    "compute_funding_features",
    "compute_oi_features",
    "compute_basis_features",
    "compute_derivatives_features",
]
