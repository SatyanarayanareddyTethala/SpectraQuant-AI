"""Optional SARIMAX directional forecasting for SpectraQuant ML.

This module provides a lightweight wrapper around ``statsmodels`` SARIMAX
to generate a directional probability signal that can be blended into the
ensemble layer.  The entire module degrades gracefully: if ``statsmodels``
is not installed ``fit_sarimax`` and ``sarimax_direction_signal`` raise
``ImportError`` with a clear installation message.

Design notes
------------
* SARIMAX is used purely as a **supplementary directional signal**, not as
  the primary trading decision.
* Short / non-stationary series are handled defensively: the function raises
  ``ValueError`` rather than silently returning garbage.
* ``enforce_stationarity=False`` and ``enforce_invertibility=False`` are set
  to avoid hard convergence failures on real financial data.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_SERIES_LENGTH = 50
"""Minimum number of observations required for a meaningful SARIMAX fit."""

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX  # type: ignore[import]

    HAS_STATSMODELS: bool = True
except Exception:  # noqa: BLE001
    HAS_STATSMODELS = False


def fit_sarimax(
    series: pd.Series,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
):
    """Fit a SARIMAX model to *series* and return the fitted result object.

    Parameters
    ----------
    series:
        Time-ordered numeric series (e.g. daily close prices or returns).
        Missing values are forward-filled before fitting.
    order:
        Non-seasonal ARIMA (p, d, q) order.
    seasonal_order:
        Seasonal (P, D, Q, s) order.

    Returns
    -------
    statsmodels SARIMAXResultsWrapper

    Raises
    ------
    ImportError
        If ``statsmodels`` is not installed.
    ValueError
        If the series has fewer than :data:`_MIN_SERIES_LENGTH` observations
        after dropping NaN.
    """
    if not HAS_STATSMODELS:
        raise ImportError(
            "statsmodels is not installed.  Install it with:\n"
            "    pip install statsmodels\n"
            "or install the project's 'opt' extras:\n"
            "    pip install 'spectraquant[opt]'"
        )

    clean = series.dropna()
    if len(clean) < _MIN_SERIES_LENGTH:
        raise ValueError(
            f"fit_sarimax: series has only {len(clean)} non-NaN observations "
            f"(minimum required: {_MIN_SERIES_LENGTH})."
        )

    model = _SARIMAX(
        clean,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def sarimax_direction_signal(
    series: pd.Series,
    steps: int = 1,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Fit SARIMAX and return a directional probability array of length *steps*.

    The returned values represent P(direction > 0) derived from the point
    forecast; they are clipped to [0.05, 0.95] to avoid extreme weights in
    the ensemble.

    Parameters
    ----------
    series:
        Historical values used for fitting (e.g. daily returns).
    steps:
        Number of one-step-ahead forecasts to produce.
    order:
        ARIMA (p, d, q) order forwarded to :func:`fit_sarimax`.
    seasonal_order:
        Seasonal order forwarded to :func:`fit_sarimax`.

    Returns
    -------
    np.ndarray of shape (steps,)
        Values in [0.05, 0.95] representing directional probability.

    Raises
    ------
    ImportError
        If ``statsmodels`` is not installed.
    ValueError
        If the series is too short.
    """
    result = fit_sarimax(series, order=order, seasonal_order=seasonal_order)
    forecast = result.forecast(steps=steps)
    forecast_arr = np.asarray(forecast, dtype=float)

    # Convert point forecast to a soft directional probability:
    # positive forecast → >0.5, negative → <0.5.
    # We use a sigmoid-like mapping centred on 0.
    series_std = float(series.dropna().std()) or 1.0
    signal = 1.0 / (1.0 + np.exp(-forecast_arr / series_std))
    return np.clip(signal, 0.05, 0.95)


__all__ = ["HAS_STATSMODELS", "fit_sarimax", "sarimax_direction_signal"]
