"""Research Lab — Autonomous Research Intelligence System.

This sub-package implements a self-researching AI layer that continuously
observes market behaviour, detects model weaknesses, generates hypotheses,
tests strategies, validates statistically, and deploys improvements safely.

Pipeline (runs nightly after trading):

    collect_results → detect_failures → generate_hypotheses →
    create_strategies → run_experiments → evaluate →
    store_memory → deploy_candidate_if_safe
"""
from __future__ import annotations

from spectraquant.intelligence.research_lab.hypothesis_engine import (
    HypothesisEngine,
    Hypothesis,
)
from spectraquant.intelligence.research_lab.strategy_generator import (
    StrategyGenerator,
    StrategyConfig,
)
from spectraquant.intelligence.research_lab.experiment_runner import (
    ExperimentRunner,
    ExperimentResult,
)
from spectraquant.intelligence.research_lab.evaluator import (
    Evaluator,
    EvaluationReport,
)
from spectraquant.intelligence.research_lab.research_memory import (
    ResearchMemory,
)
from spectraquant.intelligence.research_lab.deployment_guardian import (
    DeploymentGuardian,
    DeploymentDecision,
)
from spectraquant.intelligence.research_lab.research_loop import (
    ResearchLoop,
    run_research_cycle,
)

__all__ = [
    # hypothesis_engine
    "HypothesisEngine",
    "Hypothesis",
    # strategy_generator
    "StrategyGenerator",
    "StrategyConfig",
    # experiment_runner
    "ExperimentRunner",
    "ExperimentResult",
    # evaluator
    "Evaluator",
    "EvaluationReport",
    # research_memory
    "ResearchMemory",
    # deployment_guardian
    "DeploymentGuardian",
    "DeploymentDecision",
    # research_loop
    "ResearchLoop",
    "run_research_cycle",
]
