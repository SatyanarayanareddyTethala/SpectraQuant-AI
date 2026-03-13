"""Anomaly detection from on-chain z-score features.

Provides two complementary detectors:

* **compute_anomaly_scores** — aggregate z-score magnitude into a single
  anomaly score per row and flag rows that exceed a threshold.
* **detect_spikes** — rolling σ-based spike detector for an individual
  time series.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_anomaly_scores(
    features_df: pd.DataFrame,
    *,
    threshold: float = 2.5,
) -> pd.DataFrame:
    """Compute an anomaly score as the mean of absolute z-scores that exceed *threshold*.

    Parameters
    ----------
    features_df:
        Feature frame produced by :func:`spectraquant.onchain.features.compute_onchain_features`.
        Must contain columns ending in ``_zscore``.
    threshold:
        Minimum absolute z-score for a metric to contribute to the
        anomaly score.  Defaults to 2.5.

    Returns
    -------
    pd.DataFrame
        Copy of *features_df* with two additional columns:

        * ``anomaly_score`` — mean absolute z-score of metrics exceeding
          the threshold (0.0 when none exceed it).
        * ``is_anomaly`` — boolean flag (``True`` when ``anomaly_score > 0``).
    """
    zscore_cols = [c for c in features_df.columns if c.endswith("_zscore")]

    if not zscore_cols:
        logger.warning("No z-score columns found — returning input unchanged.")
        out = features_df.copy()
        out["anomaly_score"] = 0.0
        out["is_anomaly"] = False
        return out

    abs_z = features_df[zscore_cols].abs()
    exceeds = abs_z.where(abs_z >= threshold)

    out = features_df.copy()
    out["anomaly_score"] = exceeds.mean(axis=1).fillna(0.0)
    out["is_anomaly"] = out["anomaly_score"] > 0.0

    n_anomalies = int(out["is_anomaly"].sum())
    logger.info(
        "Anomaly detection: %d / %d rows flagged (threshold=%.2f).",
        n_anomalies,
        len(out),
        threshold,
    )
    return out


def detect_spikes(
    series: pd.Series,
    *,
    window: int = 20,
    n_sigma: float = 3.0,
) -> pd.Series:
    """Rolling spike detection returning a boolean mask.

    A value is considered a *spike* when it deviates from the rolling
    mean by more than ``n_sigma`` rolling standard deviations.

    Parameters
    ----------
    series:
        Numeric time series (typically a single on-chain metric).
    window:
        Look-back window for rolling statistics.
    n_sigma:
        Number of standard deviations to qualify as a spike.

    Returns
    -------
    pd.Series
        Boolean series aligned with *series* where ``True`` marks a spike.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().fillna(0)
    deviation = (series - rolling_mean).abs()
    is_spike = deviation > (n_sigma * rolling_std)

    n_spikes = int(is_spike.sum())
    logger.debug(
        "Spike detection: %d / %d points (window=%d, n_sigma=%.1f).",
        n_spikes,
        len(series),
        window,
        n_sigma,
    )
    return is_spike
