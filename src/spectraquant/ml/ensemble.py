"""Ensemble probability layer and signal conversion for SpectraQuant ML.

Combines class-1 probabilities from Random Forest, XGBoost, and an optional
SARIMAX directional signal into a single ensemble probability, then converts
that probability into a trading signal compatible with the existing
SpectraQuant signal architecture (``signal_score`` in [-1, 1]).

Weights
-------
Default weights (``w_rf=0.4, w_xgb=0.4, w_ts=0.2``) are configured to give
equal emphasis to the two ML models and reduce but not ignore the time-series
forecast.  These can be overridden via the ``ml`` config section.

Signal conversion
-----------------
``ensemble_to_signal`` maps the ensemble probability to a signal score::

    probability > threshold  → signal_score = +1 (buy)
    probability < (1 - threshold) → signal_score = -1 (sell)
    otherwise                → signal_score =  0 (hold)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ensemble_probability(
    rf_prob: np.ndarray,
    xgb_prob: np.ndarray,
    ts_signal: "np.ndarray | None" = None,
    w_rf: float = 0.4,
    w_xgb: float = 0.4,
    w_ts: float = 0.2,
) -> np.ndarray:
    """Blend model probabilities into a single ensemble probability.

    Parameters
    ----------
    rf_prob:
        Random Forest P(class=1) array, values in [0, 1].
    xgb_prob:
        XGBoost P(class=1) array, values in [0, 1].  Must be same length as
        *rf_prob*.
    ts_signal:
        Optional SARIMAX directional signal in [0, 1].  If ``None`` the RF and
        XGBoost weights are renormalised to sum to 1.
    w_rf, w_xgb, w_ts:
        Blend weights.  Must sum to 1.0 when all three components are used.

    Returns
    -------
    np.ndarray
        Ensemble probabilities, same shape as *rf_prob*, in [0, 1].

    Raises
    ------
    ValueError
        If array lengths are inconsistent or weights are out of [0, 1].
    """
    rf_arr = np.asarray(rf_prob, dtype=float)
    xgb_arr = np.asarray(xgb_prob, dtype=float)

    if rf_arr.shape != xgb_arr.shape:
        raise ValueError(
            f"ensemble_probability: rf_prob and xgb_prob must have the same "
            f"shape, got {rf_arr.shape} vs {xgb_arr.shape}."
        )
    for name, val in [("w_rf", w_rf), ("w_xgb", w_xgb), ("w_ts", w_ts)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(
                f"ensemble_probability: weight {name}={val} is outside [0, 1]."
            )

    if ts_signal is None:
        # Renormalise RF and XGB weights to sum to 1
        total = w_rf + w_xgb
        if total <= 0:
            raise ValueError("ensemble_probability: w_rf + w_xgb must be > 0.")
        effective_rf = w_rf / total
        effective_xgb = w_xgb / total
        result = effective_rf * rf_arr + effective_xgb * xgb_arr
    else:
        ts_arr = np.asarray(ts_signal, dtype=float)
        if ts_arr.shape != rf_arr.shape:
            # Broadcast scalar or raise
            if ts_arr.ndim == 0 or (ts_arr.ndim == 1 and len(ts_arr) == 1):
                ts_arr = np.full_like(rf_arr, float(ts_arr.flat[0]))
            else:
                raise ValueError(
                    f"ensemble_probability: ts_signal shape {ts_arr.shape} "
                    f"is incompatible with rf_prob shape {rf_arr.shape}."
                )
        result = w_rf * rf_arr + w_xgb * xgb_arr + w_ts * ts_arr

    return np.clip(result, 0.0, 1.0)


def ensemble_to_signal(
    ensemble_prob: np.ndarray,
    threshold: float = 0.55,
    index: "pd.Index | None" = None,
) -> pd.Series:
    """Convert ensemble probabilities to SpectraQuant-compatible signal scores.

    Maps the ensemble probability to one of three discrete signal values::

        prob > threshold           → +1 (buy)
        prob < (1 - threshold)     → -1 (sell)
        otherwise                  → 0  (hold)

    Parameters
    ----------
    ensemble_prob:
        Output of :func:`ensemble_probability`.
    threshold:
        Decision boundary (e.g. ``0.55`` means require 55 % confidence
        before generating a directional signal).  Must be in (0.5, 1.0).
    index:
        Optional pandas Index to attach to the returned Series (e.g. the
        datetime index of the test fold).

    Returns
    -------
    pd.Series
        Integer values in {-1, 0, 1}.

    Raises
    ------
    ValueError
        If *threshold* is not in (0.5, 1.0).
    """
    if not (0.5 < threshold < 1.0):
        raise ValueError(
            f"ensemble_to_signal: threshold must be in (0.5, 1.0), got {threshold}."
        )

    prob = np.asarray(ensemble_prob, dtype=float)
    signal = np.where(prob > threshold, 1, np.where(prob < 1.0 - threshold, -1, 0))
    return pd.Series(signal, index=index, dtype=int)


__all__ = ["ensemble_probability", "ensemble_to_signal"]
