"""Risk scoring for simulated portfolios."""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def _bounded(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def compute_risk_score(portfolio_metrics: Dict) -> Dict:
    """Compute a normalized risk score from portfolio metrics."""

    volatility = float(portfolio_metrics.get("volatility", 0) or 0)
    max_drawdown = float(portfolio_metrics.get("max_drawdown", 0) or 0)
    return_stability = float(portfolio_metrics.get("return_stability", 0) or 0)

    # Normalize metrics to 0-1 where higher is better for stability and lower risk
    vol_score = 1 - _bounded(volatility)
    dd_score = 1 - _bounded(abs(max_drawdown))
    stability_score = _bounded(return_stability)

    score = (vol_score + dd_score + stability_score) / 3
    risk_score = round(score * 100, 2)

    if risk_score >= 70:
        label = "LOW"
    elif risk_score >= 40:
        label = "MEDIUM"
    else:
        label = "HIGH"

    logger.info(
        "Computed risk score: %s (vol_score=%.2f, dd_score=%.2f, stability=%.2f)",
        risk_score,
        vol_score,
        dd_score,
        stability_score,
    )

    return {"risk_score": risk_score, "risk_label": label}

