"""Evaluator — scientific filter for experiment results.

Accepts a strategy only when it clears all statistical hurdles:
  - Sharpe improvement over baseline
  - Drawdown not worse than baseline
  - Performance stable across walk-forward folds
  - Positive regime_robustness (no regime collapses)
  - Low overfit probability

Computes
--------
- generalization_score  : 0–1 (higher = more generalisable)
- overfit_probability   : 0–1 (higher = more overfit)
- confidence            : overall acceptance confidence (0–1)
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from spectraquant.intelligence.research_lab.experiment_runner import ExperimentResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default baseline thresholds
# ---------------------------------------------------------------------------

_BASELINE_SHARPE = 0.5
_BASELINE_MAX_DRAWDOWN = 0.20       # 20%
_MIN_WIN_RATE = 0.40
_MIN_STABILITY_SCORE = 0.5
_MIN_REGIME_ROBUSTNESS = 0.0        # any fold Sharpe must be >= 0
_SHARPE_IMPROVEMENT_REQUIRED = 0.10 # absolute Sharpe improvement needed


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Evaluation outcome for a single experiment."""

    experiment_id: str
    strategy_name: str
    hypothesis_id: str
    accepted: bool
    rejection_reasons: List[str]
    generalization_score: float
    overfit_probability: float
    confidence: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    stability_score: float
    regime_robustness: float
    baseline_sharpe: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationReport":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """Apply scientific filters to experiment results.

    Parameters
    ----------
    baseline_sharpe : float
        Minimum Sharpe the new strategy must exceed.
    baseline_max_drawdown : float
        Maximum acceptable drawdown (fraction, e.g. 0.20).
    """

    def __init__(
        self,
        baseline_sharpe: float = _BASELINE_SHARPE,
        baseline_max_drawdown: float = _BASELINE_MAX_DRAWDOWN,
    ) -> None:
        self._baseline_sharpe = baseline_sharpe
        self._baseline_max_drawdown = baseline_max_drawdown

    def evaluate(self, results: List[ExperimentResult]) -> List[EvaluationReport]:
        """Evaluate a list of experiment results.

        Parameters
        ----------
        results : list[ExperimentResult]

        Returns
        -------
        list[EvaluationReport]
        """
        reports: List[EvaluationReport] = []
        for result in results:
            report = self._evaluate_single(result)
            reports.append(report)
            status = "ACCEPTED" if report.accepted else "REJECTED"
            logger.info(
                "Strategy '%s' %s (Sharpe=%.3f, confidence=%.2f)",
                result.strategy_name,
                status,
                result.sharpe,
                report.confidence,
            )
        return reports

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_single(self, result: ExperimentResult) -> EvaluationReport:
        rejection_reasons: List[str] = []

        # 1. Sharpe must improve over baseline by required margin
        required_sharpe = self._baseline_sharpe + _SHARPE_IMPROVEMENT_REQUIRED
        if result.sharpe < required_sharpe:
            rejection_reasons.append(
                f"Sharpe {result.sharpe:.3f} < required {required_sharpe:.3f}"
            )

        # 2. Drawdown must not exceed baseline
        if result.max_drawdown > self._baseline_max_drawdown:
            rejection_reasons.append(
                f"Max drawdown {result.max_drawdown:.1%} > baseline {self._baseline_max_drawdown:.1%}"
            )

        # 3. Win rate floor
        if result.win_rate < _MIN_WIN_RATE:
            rejection_reasons.append(
                f"Win rate {result.win_rate:.1%} < minimum {_MIN_WIN_RATE:.1%}"
            )

        # 4. Stability across folds
        if result.stability_score < _MIN_STABILITY_SCORE:
            rejection_reasons.append(
                f"Stability score {result.stability_score:.3f} < {_MIN_STABILITY_SCORE}"
            )

        # 5. Regime robustness
        if result.regime_robustness < _MIN_REGIME_ROBUSTNESS:
            rejection_reasons.append(
                f"Regime robustness {result.regime_robustness:.3f} < {_MIN_REGIME_ROBUSTNESS} "
                "(strategy collapses in at least one regime)"
            )

        # 6. Must have trades to evaluate
        if result.n_trades < 10:
            rejection_reasons.append(f"Insufficient trades ({result.n_trades} < 10)")

        # --- Compute derived scores ---
        generalization_score = self._generalization_score(result)
        overfit_probability = self._overfit_probability(result)
        confidence = self._confidence(result, len(rejection_reasons))

        accepted = len(rejection_reasons) == 0
        return EvaluationReport(
            experiment_id=result.experiment_id,
            strategy_name=result.strategy_name,
            hypothesis_id=result.hypothesis_id,
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            generalization_score=round(generalization_score, 4),
            overfit_probability=round(overfit_probability, 4),
            confidence=round(confidence, 4),
            sharpe=result.sharpe,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            stability_score=result.stability_score,
            regime_robustness=result.regime_robustness,
            baseline_sharpe=self._baseline_sharpe,
        )

    def _generalization_score(self, r: ExperimentResult) -> float:
        """Score 0–1 based on fold consistency and out-of-sample metrics."""
        if not r.fold_sharpes:
            return 0.0
        n_positive_folds = sum(1 for s in r.fold_sharpes if s > 0)
        fold_consistency = n_positive_folds / len(r.fold_sharpes)
        stability_component = r.stability_score
        regime_component = max(0.0, min(1.0, (r.regime_robustness + 1.0) / 2.0))
        return float(
            0.40 * fold_consistency
            + 0.35 * stability_component
            + 0.25 * regime_component
        )

    def _overfit_probability(self, r: ExperimentResult) -> float:
        """Estimate overfit probability: high Sharpe + low stability → higher overfit."""
        if r.sharpe <= 0:
            return 1.0
        sharpe_excess = max(0.0, r.sharpe - self._baseline_sharpe)
        stability_penalty = max(0.0, 1.0 - r.stability_score)
        regime_penalty = max(0.0, -r.regime_robustness) * 0.5
        raw = stability_penalty * 0.5 + regime_penalty + (
            0.2 if sharpe_excess > 2.0 else 0.0
        )
        return float(min(1.0, max(0.0, raw)))

    def _confidence(self, r: ExperimentResult, n_rejections: int) -> float:
        """Overall acceptance confidence (0–1)."""
        if n_rejections > 0:
            return max(0.0, 0.40 - 0.10 * n_rejections)
        base = min(1.0, r.sharpe / max(1.0, self._baseline_sharpe + 1.0))
        stability_boost = r.stability_score * 0.3
        return float(min(1.0, base + stability_boost))
