"""Risk management: position sizing, cost modelling, and limit checks."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class PositionSizer:
    """Multi-constraint position sizer.

    Computes the number of shares as the **minimum** of three constraints:
    1. Risk-budget constraint  (equity × risk_fraction / risk_per_share)
    2. Maximum-position cap     (b_max)
    3. ADV participation cap    (adv × adv_cap)
    """

    def compute(
        self,
        equity: float,
        risk_fraction: float,
        risk_per_share: float,
        b_max: float,
        adv: float,
        adv_cap: float = 0.02,
    ) -> float:
        """Return the position size in shares (float, caller should floor).

        Parameters
        ----------
        equity : float
            Current account equity.
        risk_fraction : float
            Fraction of equity risked on this trade (e.g. 0.01).
        risk_per_share : float
            Dollar risk per share (entry − stop).
        b_max : float
            Hard cap on shares.
        adv : float
            Average daily volume (shares).
        adv_cap : float
            Max fraction of ADV to participate (default 2 %).

        Returns
        -------
        float
            Computed position size (≥ 0).
        """
        if risk_per_share <= 0:
            logger.warning("risk_per_share <= 0; returning 0 shares")
            return 0.0

        shares_by_risk = (equity * risk_fraction) / risk_per_share
        shares_by_cap = b_max
        shares_by_adv = adv * adv_cap if adv > 0 else float("inf")

        shares = max(0.0, min(shares_by_risk, shares_by_cap, shares_by_adv))
        logger.debug(
            "PositionSizer: risk=%.0f cap=%.0f adv=%.0f → %.0f",
            shares_by_risk,
            shares_by_cap,
            shares_by_adv,
            shares,
        )
        return shares


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class CostModel:
    """Transaction cost estimator (commission + slippage)."""

    def estimate_cost(
        self,
        price: float,
        shares: float,
        spread_bps: float,
        volatility: float,
        adv: float,
        config: Optional[Dict] = None,
    ) -> float:
        """Estimate one-way dollar cost of a trade.

        Parameters
        ----------
        price : float
            Execution price.
        shares : float
            Number of shares.
        spread_bps : float
            Bid-ask spread in basis points.
        volatility : float
            Annualised volatility (e.g. 0.25 for 25 %).
        adv : float
            Average daily volume (shares).
        config : dict, optional
            Override default cost parameters.

        Returns
        -------
        float
            Estimated total dollar cost (commission + slippage).
        """
        cfg = config or {}
        commission_per_share = cfg.get("commission_per_share", 0.005)
        base_bps = cfg.get("base_slippage_bps", 2.0)
        spread_w = cfg.get("spread_weight", 0.5)
        vol_w = cfg.get("volatility_weight", 0.3)
        part_w = cfg.get("participation_weight", 0.2)

        commission = shares * commission_per_share

        vol_bps = volatility * 10_000 / math.sqrt(252) if volatility > 0 else 0.0
        participation = (shares / adv) if adv > 0 else 0.0

        slippage_bps = (
            base_bps
            + spread_w * spread_bps
            + vol_w * vol_bps
            + part_w * math.sqrt(participation) * 10_000
        )
        slippage_dollars = price * shares * slippage_bps / 10_000

        total = commission + slippage_dollars
        logger.debug(
            "CostModel: comm=%.2f slip=%.2f total=%.2f",
            commission,
            slippage_dollars,
            total,
        )
        return total


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------

class RiskLimits:
    """Portfolio-level constraint checks."""

    @staticmethod
    def check_daily_loss(current_pnl: float, limit: float) -> bool:
        """Return *True* if the daily loss limit has been **breached**.

        Parameters
        ----------
        current_pnl : float
            Realised + unrealised PnL for the day (negative = loss).
        limit : float
            Maximum allowed loss as a positive number.
        """
        breached = current_pnl <= -abs(limit)
        if breached:
            logger.warning("Daily loss breached: pnl=%.2f limit=%.2f", current_pnl, limit)
        return breached

    @staticmethod
    def check_gross_exposure(current: float, limit: float) -> bool:
        """Return *True* if gross exposure **exceeds** the limit.

        Parameters
        ----------
        current : float
            Sum of absolute position market values / equity.
        limit : float
            Maximum gross exposure (e.g. 1.0 = 100 %).
        """
        breached = current > limit
        if breached:
            logger.warning("Gross exposure breached: %.4f > %.4f", current, limit)
        return breached

    @staticmethod
    def check_name_exposure(current: float, limit: float) -> bool:
        """Return *True* if single-name exposure **exceeds** the limit."""
        breached = current > limit
        if breached:
            logger.warning("Name exposure breached: %.4f > %.4f", current, limit)
        return breached

    @staticmethod
    def check_sector_exposure(current: float, limit: float) -> bool:
        """Return *True* if sector exposure **exceeds** the limit."""
        breached = current > limit
        if breached:
            logger.warning("Sector exposure breached: %.4f > %.4f", current, limit)
        return breached
