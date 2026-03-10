"""Target price engine for SpectraQuant-AI.

Generates bull / base / bear target price scenarios from an expected move
estimate combined with ATR bands and historical analog return distributions.

Usage
-----
>>> from spectraquant.pricing.target_engine import TargetEngine
>>> engine = TargetEngine()
>>> scenarios = engine.build(last_price=2880.0, expected_move=0.042, atr=45.0)
>>> scenarios.base_target   # e.g. 3000.96
>>> scenarios.stop_price    # e.g. 2835.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["TargetEngine", "PriceScenarios"]

# Default ATR multipliers for stop and bear target
_DEFAULT_STOP_ATR_MULT = 1.5
_DEFAULT_BEAR_ATR_MULT = 2.5
_DEFAULT_BULL_MULT = 1.5  # bull target = base + (base - entry) * 1.5


@dataclass
class PriceScenarios:
    """Bull / base / bear price scenario outputs.

    Attributes
    ----------
    base_target : float
        Central expected price = last_price * (1 + expected_move).
    bull_target : float
        Optimistic scenario (larger upside).
    bear_target : float
        Pessimistic scenario (limited upside / moderate downside).
    stop_price : float
        Suggested stop-loss price.
    expected_move_pct : float
        Expected move used for base target, as a percentage (e.g. 4.2 for 4.2%).
    risk_reward : float
        (base_target - last_price) / (last_price - stop_price); 0 if undefined.
    """

    base_target: float
    bull_target: float
    bear_target: float
    stop_price: float
    expected_move_pct: float
    risk_reward: float


class TargetEngine:
    """Compute bull / base / bear price scenarios.

    Parameters
    ----------
    stop_atr_mult : float
        Stop price = last_price - atr * stop_atr_mult.
    bear_atr_mult : float
        Bear target = last_price - atr * bear_atr_mult (for short or downside).
    bull_multiplier : float
        Bull target upside multiplier relative to base move.
    vol_fallback_pct : float
        If ATR is unavailable, use last_price * vol_fallback_pct as a
        synthetic ATR proxy.
    """

    def __init__(
        self,
        stop_atr_mult: float = _DEFAULT_STOP_ATR_MULT,
        bear_atr_mult: float = _DEFAULT_BEAR_ATR_MULT,
        bull_multiplier: float = _DEFAULT_BULL_MULT,
        vol_fallback_pct: float = 0.02,
    ) -> None:
        self._stop_atr_mult = stop_atr_mult
        self._bear_atr_mult = bear_atr_mult
        self._bull_multiplier = bull_multiplier
        self._vol_fallback_pct = vol_fallback_pct

    def build(
        self,
        last_price: float,
        expected_move: float,
        atr: Optional[float] = None,
        analog_distribution: Optional[Dict[str, Any]] = None,
        iv_band: Optional[float] = None,
    ) -> PriceScenarios:
        """Build price scenarios.

        Parameters
        ----------
        last_price : float
            Current / last closing price.
        expected_move : float
            Expected fractional return (e.g. 0.042 for +4.2%).
        atr : float, optional
            Average True Range in price units.  If None, a synthetic ATR
            of ``last_price * vol_fallback_pct`` is used.
        analog_distribution : dict, optional
            Analog outcome distribution with optional keys:
            ``p25_move``, ``p75_move`` used for bull / bear widening.
        iv_band : float, optional
            Implied volatility band as a fractional price move (e.g. 0.05).
            Used to widen bull / bear targets when available.

        Returns
        -------
        PriceScenarios
        """
        if last_price <= 0:
            raise ValueError(f"last_price must be positive, got {last_price}")

        effective_atr = atr if (atr is not None and atr > 0) else (
            last_price * self._vol_fallback_pct
        )

        # Base scenario
        base_target = last_price * (1.0 + expected_move)

        # Stop price
        stop_price = last_price - effective_atr * self._stop_atr_mult

        # Bull / bear widening
        if analog_distribution:
            p75 = float(analog_distribution.get("p75_move", expected_move * 1.5) or 0)
            p25 = float(analog_distribution.get("p25_move", expected_move * 0.5) or 0)
            bull_move = max(expected_move * self._bull_multiplier, p75)
            bear_move = min(expected_move, p25)
        else:
            bull_move = expected_move * self._bull_multiplier
            bear_move = expected_move * 0.4  # conservative bear

        # Apply IV band widening if available
        if iv_band and iv_band > 0:
            bull_move = bull_move + iv_band * 0.5
            bear_move = bear_move - iv_band * 0.3

        bull_target = last_price * (1.0 + bull_move)
        bear_target = last_price * (1.0 + bear_move)

        # Risk / reward ratio
        upside = base_target - last_price
        downside = last_price - stop_price
        risk_reward = (upside / downside) if downside > 1e-9 else 0.0

        return PriceScenarios(
            base_target=round(base_target, 4),
            bull_target=round(bull_target, 4),
            bear_target=round(bear_target, 4),
            stop_price=round(stop_price, 4),
            expected_move_pct=round(expected_move * 100, 4),
            risk_reward=round(risk_reward, 4),
        )

    def build_from_analogs(
        self,
        last_price: float,
        analog_returns: List[float],
        atr: Optional[float] = None,
    ) -> PriceScenarios:
        """Build scenarios directly from a list of analog historical returns.

        Parameters
        ----------
        last_price : float
            Current / last closing price.
        analog_returns : list[float]
            Observed returns from historical analogous events.
        atr : float, optional
            Average True Range in price units.

        Returns
        -------
        PriceScenarios
        """
        if not analog_returns:
            return self.build(last_price, 0.0, atr)

        import statistics
        median_move = statistics.median(analog_returns)
        sorted_returns = sorted(analog_returns)
        n = len(sorted_returns)
        p25_idx = max(0, int(n * 0.25) - 1)
        p75_idx = min(n - 1, int(n * 0.75))

        analog_dist = {
            "p25_move": sorted_returns[p25_idx],
            "p75_move": sorted_returns[p75_idx],
        }
        return self.build(last_price, median_move, atr, analog_distribution=analog_dist)
