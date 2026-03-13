"""Downside risk engine for SpectraQuant-AI.

Computes downside risk metrics from historical analog distributions,
ATR bands, and Value-at-Risk / Conditional Value-at-Risk (CVaR).

Usage
-----
>>> from spectraquant.pricing.downside_engine import DownsideEngine
>>> engine = DownsideEngine()
>>> risk = engine.estimate(last_price=2880.0, analog_returns=[-0.03, 0.01, -0.05, ...], atr=45.0)
>>> risk.expected_downside_pct   # e.g. -3.2
>>> risk.var_95_pct              # e.g. -5.8
>>> risk.cvar_95_pct             # e.g. -7.1
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = ["DownsideEngine", "DownsideRisk"]


@dataclass
class DownsideRisk:
    """Downside risk metrics for a ticker.

    Attributes
    ----------
    expected_downside_pct : float
        Mean of negative analog returns expressed as percentage (e.g. -3.2).
    var_95_pct : float
        5th-percentile return (Value-at-Risk at 95% confidence) as percentage.
    cvar_95_pct : float
        Mean of returns below var_95 (Conditional VaR / Expected Shortfall)
        as percentage.
    crash_probability : float
        Fraction of analogs with return < ``crash_threshold`` (e.g. -10%).
    lower_bound : float
        Price-level lower bound: last_price * (1 + var_95_pct / 100).
    atr_stop : float
        Suggested ATR-based stop price.
    risk_score : float
        Composite risk score [0, 1]; higher = more dangerous.
    """

    expected_downside_pct: float
    var_95_pct: float
    cvar_95_pct: float
    crash_probability: float
    lower_bound: float
    atr_stop: float
    risk_score: float


class DownsideEngine:
    """Compute downside risk metrics.

    Parameters
    ----------
    atr_mult : float
        ATR multiplier for the stop-price calculation (default 2.0).
    crash_threshold : float
        Return threshold below which an outcome is considered a "crash"
        (default -0.10, i.e. −10%).
    vol_fallback_pct : float
        Synthetic ATR as fraction of price when ATR is unavailable.
    """

    def __init__(
        self,
        atr_mult: float = 2.0,
        crash_threshold: float = -0.10,
        vol_fallback_pct: float = 0.02,
    ) -> None:
        self._atr_mult = atr_mult
        self._crash_threshold = crash_threshold
        self._vol_fallback_pct = vol_fallback_pct

    def estimate(
        self,
        last_price: float,
        analog_returns: Optional[List[float]] = None,
        atr: Optional[float] = None,
        implied_vol: Optional[float] = None,
    ) -> DownsideRisk:
        """Estimate downside risk metrics.

        Parameters
        ----------
        last_price : float
            Current / last closing price.
        analog_returns : list[float], optional
            Historical returns from analogous events.  If empty or None,
            estimates are derived from ATR / implied vol only.
        atr : float, optional
            Average True Range in price units.
        implied_vol : float, optional
            Implied volatility (fractional, e.g. 0.25 for 25% annualised).
            Used as a fallback when analog_returns is not available.

        Returns
        -------
        DownsideRisk
        """
        if last_price <= 0:
            raise ValueError(f"last_price must be positive, got {last_price}")

        effective_atr = (
            atr if (atr is not None and atr > 0) else last_price * self._vol_fallback_pct
        )
        atr_stop = last_price - effective_atr * self._atr_mult

        # ---- Analog-based statistics ----------------------------------------
        if analog_returns and len(analog_returns) >= 3:
            returns = np.array(analog_returns, dtype=float)
            negatives = returns[returns < 0]

            expected_downside = float(negatives.mean()) if len(negatives) > 0 else 0.0

            # VaR and CVaR at 95%
            var_95 = float(np.percentile(returns, 5))
            tail = returns[returns <= var_95]
            cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

            crash_prob = float((returns < self._crash_threshold).mean())

        elif implied_vol is not None and implied_vol > 0:
            # Approximation using a normal distribution scaled to 1-day returns
            daily_vol = implied_vol / math.sqrt(252)
            expected_downside = -daily_vol
            var_95 = -1.645 * daily_vol
            cvar_95 = -2.063 * daily_vol  # E[Z | Z < -1.645] ≈ -2.063
            crash_prob = max(0.0, min(0.5, daily_vol * 2))

        else:
            # Minimal ATR-based estimate
            daily_move_pct = effective_atr / last_price
            expected_downside = -daily_move_pct
            var_95 = -daily_move_pct * 1.645
            cvar_95 = -daily_move_pct * 2.063
            crash_prob = 0.0

        lower_bound = last_price * (1.0 + var_95)

        # ---- Composite risk score [0, 1] ------------------------------------
        # High crash_prob, large CVaR, and large ATR all push score up
        risk_score = min(
            1.0,
            crash_prob * 3.0
            + abs(cvar_95) * 5.0
            + (effective_atr / last_price) * 10.0,
        )

        return DownsideRisk(
            expected_downside_pct=round(expected_downside * 100, 4),
            var_95_pct=round(var_95 * 100, 4),
            cvar_95_pct=round(cvar_95 * 100, 4),
            crash_probability=round(crash_prob, 4),
            lower_bound=round(lower_bound, 4),
            atr_stop=round(atr_stop, 4),
            risk_score=round(risk_score, 4),
        )
