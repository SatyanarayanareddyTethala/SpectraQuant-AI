"""Deployment Guardian — safety layer for strategy deployment.

Before activating a new strategy the guardian checks:
  - Capital-at-risk budget
  - Correlation with existing live strategies
  - Volatility exposure
  - Performance decay risk (based on evaluation confidence)

Deployment modes (in escalating trust):
  shadow_mode       — monitor signals only, no capital allocated
  paper_trade       — simulated execution at full size
  limited_capital   — live execution at a fraction of target size
  full_deployment   — live execution at full target size

The guardian NEVER moves directly to ``full_deployment`` from an
unvalidated or low-confidence strategy.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from spectraquant.intelligence.research_lab.evaluator import EvaluationReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deployment modes (ordered by trust level)
# ---------------------------------------------------------------------------

DEPLOYMENT_MODES = ("shadow_mode", "paper_trade", "limited_capital", "full_deployment")

# ---------------------------------------------------------------------------
# Safety thresholds
# ---------------------------------------------------------------------------

_MIN_CONFIDENCE_FOR_PAPER = 0.40
_MIN_CONFIDENCE_FOR_LIMITED = 0.60
_MIN_CONFIDENCE_FOR_FULL = 0.80

_MAX_CAPITAL_RISK_FRACTION = 0.05      # never risk more than 5% of capital on one strategy
_MAX_CORRELATION_WITH_LIVE = 0.80      # correlation > 80% = redundant strategy
_MAX_VOLATILITY_EXPOSURE = 0.30        # max vol exposure from new strategy
_MAX_DRAWDOWN_FOR_DEPLOYMENT = 0.15    # reject if drawdown > 15%


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DeploymentDecision:
    """Outcome of the deployment guardian check."""

    strategy_name: str
    experiment_id: str
    approved: bool
    mode: str                          # one of DEPLOYMENT_MODES
    rejection_reasons: List[str]
    capital_fraction: float            # fraction of target capital to allocate
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Guardian class
# ---------------------------------------------------------------------------

class DeploymentGuardian:
    """Safety layer between evaluation and live deployment.

    Parameters
    ----------
    max_capital_risk : float
        Maximum fraction of total capital the new strategy may risk.
    max_correlation : float
        Maximum acceptable correlation with existing live strategies.
    max_vol_exposure : float
        Maximum volatility exposure contributed by the new strategy.
    """

    def __init__(
        self,
        max_capital_risk: float = _MAX_CAPITAL_RISK_FRACTION,
        max_correlation: float = _MAX_CORRELATION_WITH_LIVE,
        max_vol_exposure: float = _MAX_VOLATILITY_EXPOSURE,
    ) -> None:
        self._max_capital_risk = max_capital_risk
        self._max_correlation = max_correlation
        self._max_vol_exposure = max_vol_exposure

    def review(
        self,
        report: EvaluationReport,
        live_strategy_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> DeploymentDecision:
        """Assess a strategy for deployment.

        Parameters
        ----------
        report : EvaluationReport
            Evaluation output from :class:`Evaluator`.
        live_strategy_metadata : list[dict], optional
            Metadata about currently live strategies.  Each dict may contain
            ``correlation`` and ``volatility_exposure`` keys.

        Returns
        -------
        DeploymentDecision
        """
        if live_strategy_metadata is None:
            live_strategy_metadata = []

        rejection_reasons: List[str] = []
        notes: List[str] = []

        # ---- Guard 1: strategy must be accepted by evaluator ----
        if not report.accepted:
            rejection_reasons.append(
                "Strategy was rejected by evaluator: "
                + "; ".join(report.rejection_reasons)
            )

        # ---- Guard 2: drawdown ceiling ----
        if report.max_drawdown > _MAX_DRAWDOWN_FOR_DEPLOYMENT:
            rejection_reasons.append(
                f"Max drawdown {report.max_drawdown:.1%} exceeds deployment ceiling "
                f"{_MAX_DRAWDOWN_FOR_DEPLOYMENT:.1%}"
            )

        # ---- Guard 3: correlation with existing live strategies ----
        for live in live_strategy_metadata:
            corr = float(live.get("correlation", 0.0))
            if corr > self._max_correlation:
                rejection_reasons.append(
                    f"Correlation {corr:.2f} with live strategy '{live.get('name', 'unknown')}' "
                    f"exceeds maximum {self._max_correlation:.2f} (redundant)"
                )

        # ---- Guard 4: volatility exposure ----
        total_vol = sum(
            float(s.get("volatility_exposure", 0.0)) for s in live_strategy_metadata
        )
        strategy_vol = report.max_drawdown * 0.5  # proxy from drawdown
        if total_vol + strategy_vol > self._max_vol_exposure:
            notes.append(
                f"Portfolio vol exposure would reach {total_vol + strategy_vol:.1%}; "
                "capital allocation reduced"
            )

        # ---- Guard 5: performance decay risk (low confidence) ----
        if report.confidence < _MIN_CONFIDENCE_FOR_PAPER:
            rejection_reasons.append(
                f"Confidence {report.confidence:.2f} too low for any live deployment"
            )

        # ---- Determine mode based on confidence ----
        if rejection_reasons:
            mode = "shadow_mode"
            capital_fraction = 0.0
            approved = False
        elif report.confidence >= _MIN_CONFIDENCE_FOR_FULL:
            mode = "limited_capital"   # still capped at limited; human must promote to full
            capital_fraction = self._max_capital_risk
            approved = True
            notes.append(
                "High confidence: recommended for limited_capital; "
                "manual review required before full_deployment"
            )
        elif report.confidence >= _MIN_CONFIDENCE_FOR_LIMITED:
            mode = "limited_capital"
            capital_fraction = self._max_capital_risk * 0.5
            approved = True
        else:
            mode = "paper_trade"
            capital_fraction = 0.0
            approved = True
            notes.append("Low confidence: paper_trade only; monitor for 10+ trading days")

        decision = DeploymentDecision(
            strategy_name=report.strategy_name,
            experiment_id=report.experiment_id,
            approved=approved,
            mode=mode,
            rejection_reasons=rejection_reasons,
            capital_fraction=round(capital_fraction, 4),
            notes=notes,
        )

        logger.info(
            "Deployment decision for '%s': approved=%s mode=%s",
            report.strategy_name,
            approved,
            mode,
        )
        return decision
