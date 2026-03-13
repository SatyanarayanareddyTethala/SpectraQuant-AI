"""Portfolio constraint enforcement.

Applies position-level and portfolio-level limits after the allocator
produces raw weights.  Designed so the existing simulator is unaffected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    """Constraint specification for a portfolio."""

    max_weight: float = 0.25
    min_weight: float = -0.25
    max_gross_leverage: float = 1.0
    max_turnover: float | None = None
    max_positions: int | None = None
    sector_limits: dict[str, float] = field(default_factory=dict)
    sector_map: dict[str, str] = field(default_factory=dict)


def apply_constraints(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    prev_weights: pd.Series | None = None,
) -> pd.Series:
    """Enforce constraints on a raw weight vector.

    Processing order:
    1. Clip individual weights to [min_weight, max_weight].
    2. Enforce sector limits if provided.
    3. Enforce max positions (keep top-N by absolute weight).
    4. Enforce turnover cap relative to previous weights.
    5. Scale to max gross leverage.
    6. Re-normalize if needed.

    Parameters
    ----------
    weights : pd.Series
        Raw portfolio weights.
    constraints : PortfolioConstraints
        Constraint specification.
    prev_weights : pd.Series, optional
        Previous period weights (for turnover calculation).

    Returns
    -------
    pd.Series
        Constrained weights.
    """
    w = weights.copy()

    # 1. Clip individual weights
    w = w.clip(lower=constraints.min_weight, upper=constraints.max_weight)

    # 2. Sector limits
    if constraints.sector_limits and constraints.sector_map:
        w = _enforce_sector_limits(w, constraints.sector_map, constraints.sector_limits)

    # 3. Max positions
    if constraints.max_positions is not None and len(w) > constraints.max_positions:
        top_n = w.abs().nlargest(constraints.max_positions).index
        w = w.reindex(top_n).fillna(0.0)

    # 4. Turnover cap
    if constraints.max_turnover is not None and prev_weights is not None:
        w = _cap_turnover(w, prev_weights, constraints.max_turnover)

    # 5. Gross leverage
    gross = w.abs().sum()
    if gross > constraints.max_gross_leverage and gross > 1e-12:
        w = w * (constraints.max_gross_leverage / gross)

    # Guard: if all weights are zero but there were valid inputs, log warning
    if w.abs().sum() < 1e-12 and weights.abs().sum() > 1e-12:
        logger.warning(
            "Constraints zeroed all weights — reverting to clipped input"
        )
        w = weights.clip(lower=constraints.min_weight, upper=constraints.max_weight)
        gross = w.abs().sum()
        if gross > constraints.max_gross_leverage and gross > 1e-12:
            w = w * (constraints.max_gross_leverage / gross)

    return w


def _enforce_sector_limits(
    weights: pd.Series,
    sector_map: dict[str, str],
    sector_limits: dict[str, float],
) -> pd.Series:
    """Scale down over-allocated sectors proportionally."""
    w = weights.copy()
    for sector, limit in sector_limits.items():
        members = [sym for sym, sec in sector_map.items() if sec == sector and sym in w.index]
        if not members:
            continue
        sector_gross = w[members].abs().sum()
        if sector_gross > limit and sector_gross > 1e-12:
            w[members] = w[members] * (limit / sector_gross)
    return w


def _cap_turnover(
    new_weights: pd.Series,
    old_weights: pd.Series,
    max_turnover: float,
) -> pd.Series:
    """Reduce trades to stay within turnover budget."""
    all_syms = sorted(set(new_weights.index) | set(old_weights.index))
    nw = new_weights.reindex(all_syms, fill_value=0.0)
    ow = old_weights.reindex(all_syms, fill_value=0.0)

    diff = nw - ow
    turnover = diff.abs().sum()
    if turnover <= max_turnover or turnover < 1e-12:
        return new_weights

    scale = max_turnover / turnover
    blended = ow + diff * scale
    return blended.reindex(new_weights.index).fillna(0.0)
