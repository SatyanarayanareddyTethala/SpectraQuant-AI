"""Research Loop — autonomous nightly research cycle.

Orchestrates the full pipeline:

    collect_results()
    ↓ detect_failures()
    ↓ generate_hypotheses()
    ↓ create_strategies()
    ↓ run_experiments()
    ↓ evaluate()
    ↓ store_memory()
    ↓ deploy_candidate_if_safe()

Designed to run once per day after trading session ends.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine, Hypothesis
from spectraquant.intelligence.research_lab.strategy_generator import StrategyGenerator, StrategyConfig
from spectraquant.intelligence.research_lab.experiment_runner import ExperimentRunner, ExperimentResult
from spectraquant.intelligence.research_lab.evaluator import Evaluator, EvaluationReport
from spectraquant.intelligence.research_lab.research_memory import ResearchMemory
from spectraquant.intelligence.research_lab.deployment_guardian import (
    DeploymentGuardian,
    DeploymentDecision,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cycle result summary
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    """Summary of one research cycle execution."""

    cycle_id: str
    timestamp: str
    n_hypotheses: int
    n_strategies: int
    n_experiments: int
    n_accepted: int
    n_deployed: int
    deployment_decisions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# ResearchLoop class
# ---------------------------------------------------------------------------

class ResearchLoop:
    """Autonomous research loop controller.

    Parameters
    ----------
    memory_path : str
        Path to research memory JSON.
    experiment_dir : str
        Directory for experiment reports.
    baseline_sharpe : float
        Sharpe baseline the evaluator uses.
    """

    def __init__(
        self,
        memory_path: str = "data/intelligence/research_memory.json",
        experiment_dir: str = "reports/research/experiments",
        baseline_sharpe: float = 0.5,
    ) -> None:
        self._memory = ResearchMemory(path=memory_path)
        self._hypothesis_engine = HypothesisEngine(memory_path=memory_path)
        self._strategy_generator = StrategyGenerator()
        self._experiment_runner = ExperimentRunner(output_dir=experiment_dir)
        self._evaluator = Evaluator(baseline_sharpe=baseline_sharpe)
        self._guardian = DeploymentGuardian()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        metrics: Dict[str, Any],
        price_data: Optional[Any] = None,
        live_strategy_metadata: Optional[List[Dict[str, Any]]] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> CycleResult:
        """Execute one full research cycle.

        Parameters
        ----------
        metrics : dict
            Performance/failure metrics (see :class:`HypothesisEngine`).
        price_data : pd.DataFrame, optional
            Historical prices for backtesting.
        live_strategy_metadata : list[dict], optional
            Currently live strategy metadata for deployment correlation checks.
        market_context : dict, optional
            Snapshot of current market state to store in memory.

        Returns
        -------
        CycleResult
        """
        cycle_id = uuid.uuid4().hex[:8]
        logger.info("Research cycle %s started", cycle_id)
        errors: List[str] = []

        # --- Step 1: Store market context ---
        if market_context:
            try:
                self._memory.store_market_context(market_context)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"store_market_context failed: {exc}")

        # --- Step 2: Generate hypotheses ---
        hypotheses: List[Hypothesis] = []
        try:
            hypotheses = self._hypothesis_engine.generate(metrics)
            for h in hypotheses:
                self._memory.store_hypothesis(h)
            logger.info("Cycle %s: %d new hypotheses", cycle_id, len(hypotheses))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"generate_hypotheses failed: {exc}")
            logger.error("Hypothesis generation error: %s", exc)

        # --- Step 3: Generate strategies ---
        strategies: List[StrategyConfig] = []
        try:
            strategies = self._strategy_generator.generate(hypotheses)
            logger.info("Cycle %s: %d strategies created", cycle_id, len(strategies))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"create_strategies failed: {exc}")
            logger.error("Strategy generation error: %s", exc)

        # --- Step 4: Run experiments ---
        experiment_results: List[ExperimentResult] = []
        try:
            experiment_results = self._experiment_runner.run(strategies, price_data)
            for r in experiment_results:
                self._memory.store_experiment(r)
            logger.info("Cycle %s: %d experiments completed", cycle_id, len(experiment_results))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"run_experiments failed: {exc}")
            logger.error("Experiment runner error: %s", exc)

        # --- Step 5: Evaluate ---
        evaluations: List[EvaluationReport] = []
        try:
            evaluations = self._evaluator.evaluate(experiment_results)
            for ev in evaluations:
                self._memory.store_evaluation(ev)
            n_accepted = sum(1 for ev in evaluations if ev.accepted)
            logger.info(
                "Cycle %s: %d/%d strategies accepted",
                cycle_id, n_accepted, len(evaluations),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"evaluate failed: {exc}")
            logger.error("Evaluator error: %s", exc)
            n_accepted = 0

        # --- Step 6: Deploy candidates ---
        deployment_decisions: List[DeploymentDecision] = []
        try:
            for ev in evaluations:
                decision = self._guardian.review(ev, live_strategy_metadata)
                deployment_decisions.append(decision)
                if decision.approved:
                    logger.info(
                        "Cycle %s: strategy '%s' approved for %s (capital=%.1f%%)",
                        cycle_id,
                        decision.strategy_name,
                        decision.mode,
                        decision.capital_fraction * 100,
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"deploy_candidate failed: {exc}")
            logger.error("Deployment guardian error: %s", exc)

        # --- Step 7: Persist memory ---
        try:
            self._memory.save()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"save_memory failed: {exc}")
            logger.error("Memory save error: %s", exc)

        n_deployed = sum(1 for d in deployment_decisions if d.approved)
        logger.info(
            "Research cycle %s complete: hypotheses=%d strategies=%d "
            "experiments=%d accepted=%d deployed=%d errors=%d",
            cycle_id,
            len(hypotheses),
            len(strategies),
            len(experiment_results),
            n_accepted,
            n_deployed,
            len(errors),
        )

        return CycleResult(
            cycle_id=cycle_id,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            n_hypotheses=len(hypotheses),
            n_strategies=len(strategies),
            n_experiments=len(experiment_results),
            n_accepted=n_accepted,
            n_deployed=n_deployed,
            deployment_decisions=[d.to_dict() for d in deployment_decisions],
            errors=errors,
        )

    def status(self) -> Dict[str, Any]:
        """Return current memory summary."""
        return self._memory.summary()

    def history(self) -> Dict[str, Any]:
        """Return full memory contents."""
        return {
            "hypotheses": self._memory.get_hypotheses(),
            "experiments": self._memory.get_experiments(),
            "evaluations": self._memory.get_evaluations(),
            "successes": self._memory.get_successes(),
            "failures": self._memory.get_failures(),
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_research_cycle(
    metrics: Dict[str, Any],
    memory_path: str = "data/intelligence/research_memory.json",
    experiment_dir: str = "reports/research/experiments",
    price_data: Optional[Any] = None,
    live_strategy_metadata: Optional[List[Dict[str, Any]]] = None,
    market_context: Optional[Dict[str, Any]] = None,
    baseline_sharpe: float = 0.5,
) -> CycleResult:
    """Convenience wrapper — instantiate :class:`ResearchLoop` and run one cycle.

    Parameters
    ----------
    metrics : dict
        Performance/failure statistics driving hypothesis generation.
    memory_path : str
        Path to research memory JSON.
    experiment_dir : str
        Directory for experiment reports.
    price_data : pd.DataFrame, optional
        Historical price data for backtesting.
    live_strategy_metadata : list[dict], optional
        Currently live strategy info for deployment checks.
    market_context : dict, optional
        Market snapshot to persist.
    baseline_sharpe : float
        Minimum Sharpe the new strategy must beat.

    Returns
    -------
    CycleResult
    """
    loop = ResearchLoop(
        memory_path=memory_path,
        experiment_dir=experiment_dir,
        baseline_sharpe=baseline_sharpe,
    )
    return loop.run(
        metrics=metrics,
        price_data=price_data,
        live_strategy_metadata=live_strategy_metadata,
        market_context=market_context,
    )
