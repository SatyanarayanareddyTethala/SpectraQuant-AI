"""Tests for the intelligence/research_lab autonomous research system."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_memory_path(tmp_path: Path) -> str:
    return str(tmp_path / "research_memory.json")


@pytest.fixture()
def tmp_experiment_dir(tmp_path: Path) -> str:
    d = tmp_path / "experiments"
    d.mkdir()
    return str(d)


@pytest.fixture()
def sample_metrics() -> Dict[str, Any]:
    return {
        "failure_rate": 0.60,
        "regime_failures": {"TRENDING": 5, "PANIC": 3},
        "news_shock_count": 4,
        "overconfidence_count": 6,
        "regime_shift_count": 4,
        "avg_slippage_bps": 35.0,
        "dominant_regime": "TRENDING",
        "in_sample_sharpe": 2.0,
        "out_of_sample_sharpe": 0.3,
    }


# ===========================================================================
# Hypothesis Engine
# ===========================================================================

class TestHypothesisEngine:
    def test_generates_hypotheses_from_metrics(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        assert len(hypotheses) > 0

    def test_hypothesis_has_required_fields(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        for hyp in hypotheses:
            assert hyp.hypothesis_id
            assert hyp.trigger_reason
            assert hyp.affected_regime
            assert hyp.suggested_feature_change
            assert hyp.timestamp

    def test_no_duplicate_hypotheses(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        first = engine.generate(sample_metrics)
        second = engine.generate(sample_metrics)
        assert len(second) == 0, "Should not generate duplicates"

    def test_to_dict_and_from_dict(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import (
            HypothesisEngine,
            Hypothesis,
        )

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        assert hypotheses
        h = hypotheses[0]
        d = h.to_dict()
        h2 = Hypothesis.from_dict(d)
        assert h2.hypothesis_id == h.hypothesis_id
        assert h2.trigger_reason == h.trigger_reason

    def test_empty_metrics_yields_no_hypotheses(self, tmp_memory_path: str) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate({})
        assert len(hypotheses) == 0


# ===========================================================================
# Strategy Generator
# ===========================================================================

class TestStrategyGenerator:
    def test_generates_strategies_from_hypotheses(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine
        from spectraquant.intelligence.research_lab.strategy_generator import StrategyGenerator

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        assert hypotheses

        gen = StrategyGenerator()
        strategies = gen.generate(hypotheses)
        assert len(strategies) == len(hypotheses)

    def test_strategy_links_to_hypothesis(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine
        from spectraquant.intelligence.research_lab.strategy_generator import StrategyGenerator

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        gen = StrategyGenerator()
        strategies = gen.generate(hypotheses)

        hyp_ids = {h.hypothesis_id for h in hypotheses}
        for s in strategies:
            assert s.hypothesis_id in hyp_ids
            assert s.strategy_name
            assert s.features_used
            assert s.signal_logic

    def test_strategy_config_serialization(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine
        from spectraquant.intelligence.research_lab.strategy_generator import (
            StrategyGenerator,
            StrategyConfig,
        )

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        gen = StrategyGenerator()
        strategies = gen.generate(hypotheses)
        assert strategies
        s = strategies[0]
        d = s.to_dict()
        s2 = StrategyConfig.from_dict(d)
        assert s2.strategy_name == s.strategy_name
        assert s2.hypothesis_id == s.hypothesis_id

    def test_no_strategies_for_empty_hypotheses(self) -> None:
        from spectraquant.intelligence.research_lab.strategy_generator import StrategyGenerator

        gen = StrategyGenerator()
        strategies = gen.generate([])
        assert strategies == []


# ===========================================================================
# Experiment Runner
# ===========================================================================

class TestExperimentRunner:
    def test_runs_experiment_returns_result(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str, tmp_experiment_dir: str
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine
        from spectraquant.intelligence.research_lab.strategy_generator import StrategyGenerator
        from spectraquant.intelligence.research_lab.experiment_runner import ExperimentRunner

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        strategies = StrategyGenerator().generate(hypotheses[:1])

        runner = ExperimentRunner(output_dir=tmp_experiment_dir)
        results = runner.run(strategies)
        assert len(results) == 1
        r = results[0]
        assert r.experiment_id
        assert r.strategy_name
        assert isinstance(r.sharpe, float)
        assert isinstance(r.max_drawdown, float)
        assert isinstance(r.win_rate, float)
        assert 0.0 <= r.win_rate <= 1.0
        assert r.max_drawdown >= 0.0

    def test_result_serialization(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str, tmp_experiment_dir: str
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine
        from spectraquant.intelligence.research_lab.strategy_generator import StrategyGenerator
        from spectraquant.intelligence.research_lab.experiment_runner import (
            ExperimentRunner,
            ExperimentResult,
        )

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        strategies = StrategyGenerator().generate(hypotheses[:1])
        runner = ExperimentRunner(output_dir=tmp_experiment_dir)
        results = runner.run(strategies)
        r = results[0]
        d = r.to_dict()
        r2 = ExperimentResult.from_dict(d)
        assert r2.experiment_id == r.experiment_id
        assert r2.sharpe == r.sharpe

    def test_experiment_report_saved_to_disk(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str, tmp_experiment_dir: str
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine
        from spectraquant.intelligence.research_lab.strategy_generator import StrategyGenerator
        from spectraquant.intelligence.research_lab.experiment_runner import ExperimentRunner

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        strategies = StrategyGenerator().generate(hypotheses[:1])
        runner = ExperimentRunner(output_dir=tmp_experiment_dir)
        results = runner.run(strategies)
        assert results[0].experiment_dir
        assert Path(results[0].experiment_dir).exists()


# ===========================================================================
# Evaluator
# ===========================================================================

class TestEvaluator:
    def test_rejects_low_sharpe_strategy(
        self, sample_metrics: Dict[str, Any], tmp_memory_path: str, tmp_experiment_dir: str
    ) -> None:
        from spectraquant.intelligence.research_lab.experiment_runner import ExperimentResult
        from spectraquant.intelligence.research_lab.evaluator import Evaluator

        result = ExperimentResult(
            experiment_id="test_001",
            strategy_name="weak_strategy",
            hypothesis_id="abc123",
            sharpe=0.1,
            max_drawdown=0.05,
            win_rate=0.50,
            stability_score=0.8,
            regime_robustness=0.1,
            n_trades=20,
            fold_sharpes=[0.1, 0.05, 0.15],
        )
        evaluator = Evaluator(baseline_sharpe=0.5)
        reports = evaluator.evaluate([result])
        assert len(reports) == 1
        assert not reports[0].accepted
        assert reports[0].rejection_reasons

    def test_accepts_strong_strategy(self) -> None:
        from spectraquant.intelligence.research_lab.experiment_runner import ExperimentResult
        from spectraquant.intelligence.research_lab.evaluator import Evaluator

        result = ExperimentResult(
            experiment_id="test_002",
            strategy_name="strong_strategy",
            hypothesis_id="def456",
            sharpe=1.8,
            max_drawdown=0.08,
            win_rate=0.58,
            stability_score=0.9,
            regime_robustness=0.5,
            n_trades=50,
            fold_sharpes=[1.5, 1.8, 2.1],
        )
        evaluator = Evaluator(baseline_sharpe=0.5)
        reports = evaluator.evaluate([result])
        assert len(reports) == 1
        assert reports[0].accepted
        assert 0.0 <= reports[0].generalization_score <= 1.0
        assert 0.0 <= reports[0].overfit_probability <= 1.0
        assert 0.0 <= reports[0].confidence <= 1.0

    def test_evaluation_report_serialization(self) -> None:
        from spectraquant.intelligence.research_lab.experiment_runner import ExperimentResult
        from spectraquant.intelligence.research_lab.evaluator import Evaluator, EvaluationReport

        result = ExperimentResult(
            experiment_id="test_003",
            strategy_name="test_strategy",
            hypothesis_id="ghi789",
            sharpe=1.2,
            max_drawdown=0.10,
            win_rate=0.52,
            stability_score=0.75,
            regime_robustness=0.3,
            n_trades=30,
            fold_sharpes=[1.0, 1.2, 1.4],
        )
        evaluator = Evaluator(baseline_sharpe=0.5)
        reports = evaluator.evaluate([result])
        d = reports[0].to_dict()
        r2 = EvaluationReport.from_dict(d)
        assert r2.experiment_id == reports[0].experiment_id


# ===========================================================================
# Research Memory
# ===========================================================================

class TestResearchMemory:
    def test_store_and_retrieve_hypothesis(
        self, tmp_memory_path: str, sample_metrics: Dict[str, Any]
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine
        from spectraquant.intelligence.research_lab.research_memory import ResearchMemory

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        assert hypotheses

        memory = ResearchMemory(path=tmp_memory_path)
        for h in hypotheses:
            memory.store_hypothesis(h)
        memory.save()

        memory2 = ResearchMemory(path=tmp_memory_path)
        assert len(memory2.get_hypotheses()) == len(hypotheses)

    def test_idempotent_hypothesis_storage(
        self, tmp_memory_path: str, sample_metrics: Dict[str, Any]
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine
        from spectraquant.intelligence.research_lab.research_memory import ResearchMemory

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        memory = ResearchMemory(path=tmp_memory_path)
        for h in hypotheses:
            memory.store_hypothesis(h)
            memory.store_hypothesis(h)  # duplicate
        assert len(memory.get_hypotheses()) == len(hypotheses)

    def test_summary(self, tmp_memory_path: str) -> None:
        from spectraquant.intelligence.research_lab.research_memory import ResearchMemory

        memory = ResearchMemory(path=tmp_memory_path)
        summary = memory.summary()
        assert "n_hypotheses" in summary
        assert "n_experiments" in summary
        assert "n_successes" in summary
        assert "n_failures" in summary

    def test_is_hypothesis_known(
        self, tmp_memory_path: str, sample_metrics: Dict[str, Any]
    ) -> None:
        from spectraquant.intelligence.research_lab.hypothesis_engine import HypothesisEngine
        from spectraquant.intelligence.research_lab.research_memory import ResearchMemory

        engine = HypothesisEngine(memory_path=tmp_memory_path)
        hypotheses = engine.generate(sample_metrics)
        memory = ResearchMemory(path=tmp_memory_path)
        assert not memory.is_hypothesis_known(hypotheses[0].hypothesis_id)
        memory.store_hypothesis(hypotheses[0])
        assert memory.is_hypothesis_known(hypotheses[0].hypothesis_id)


# ===========================================================================
# Deployment Guardian
# ===========================================================================

class TestDeploymentGuardian:
    def test_rejects_failed_strategy(self) -> None:
        from spectraquant.intelligence.research_lab.evaluator import EvaluationReport
        from spectraquant.intelligence.research_lab.deployment_guardian import DeploymentGuardian

        report = EvaluationReport(
            experiment_id="exp_001",
            strategy_name="bad_strategy",
            hypothesis_id="abc",
            accepted=False,
            rejection_reasons=["Sharpe too low"],
            generalization_score=0.2,
            overfit_probability=0.8,
            confidence=0.1,
            sharpe=0.2,
            max_drawdown=0.25,
            win_rate=0.35,
            stability_score=0.3,
            regime_robustness=-0.5,
            baseline_sharpe=0.5,
        )
        guardian = DeploymentGuardian()
        decision = guardian.review(report)
        assert not decision.approved
        assert decision.mode == "shadow_mode"
        assert decision.capital_fraction == 0.0

    def test_approves_paper_trade_for_moderate_confidence(self) -> None:
        from spectraquant.intelligence.research_lab.evaluator import EvaluationReport
        from spectraquant.intelligence.research_lab.deployment_guardian import DeploymentGuardian

        report = EvaluationReport(
            experiment_id="exp_002",
            strategy_name="ok_strategy",
            hypothesis_id="def",
            accepted=True,
            rejection_reasons=[],
            generalization_score=0.55,
            overfit_probability=0.2,
            confidence=0.45,
            sharpe=0.9,
            max_drawdown=0.10,
            win_rate=0.52,
            stability_score=0.7,
            regime_robustness=0.2,
            baseline_sharpe=0.5,
        )
        guardian = DeploymentGuardian()
        decision = guardian.review(report)
        assert decision.approved
        assert decision.mode in ("paper_trade", "limited_capital")

    def test_decision_serialization(self) -> None:
        from spectraquant.intelligence.research_lab.evaluator import EvaluationReport
        from spectraquant.intelligence.research_lab.deployment_guardian import DeploymentGuardian

        report = EvaluationReport(
            experiment_id="exp_003",
            strategy_name="test_strategy",
            hypothesis_id="ghi",
            accepted=True,
            rejection_reasons=[],
            generalization_score=0.7,
            overfit_probability=0.1,
            confidence=0.75,
            sharpe=1.5,
            max_drawdown=0.08,
            win_rate=0.60,
            stability_score=0.85,
            regime_robustness=0.4,
            baseline_sharpe=0.5,
        )
        guardian = DeploymentGuardian()
        decision = guardian.review(report)
        d = decision.to_dict()
        assert "approved" in d
        assert "mode" in d
        assert "capital_fraction" in d


# ===========================================================================
# Research Loop (end-to-end)
# ===========================================================================

class TestResearchLoop:
    def test_full_cycle_runs_without_error(
        self,
        sample_metrics: Dict[str, Any],
        tmp_memory_path: str,
        tmp_experiment_dir: str,
    ) -> None:
        from spectraquant.intelligence.research_lab import run_research_cycle

        result = run_research_cycle(
            metrics=sample_metrics,
            memory_path=tmp_memory_path,
            experiment_dir=tmp_experiment_dir,
        )
        assert result.cycle_id
        assert result.timestamp
        assert result.n_hypotheses >= 0
        assert result.n_strategies >= 0
        assert result.n_experiments >= 0
        assert isinstance(result.errors, list)

    def test_cycle_result_serialization(
        self,
        sample_metrics: Dict[str, Any],
        tmp_memory_path: str,
        tmp_experiment_dir: str,
    ) -> None:
        from spectraquant.intelligence.research_lab import run_research_cycle

        result = run_research_cycle(
            metrics=sample_metrics,
            memory_path=tmp_memory_path,
            experiment_dir=tmp_experiment_dir,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "cycle_id" in d
        assert "n_hypotheses" in d

    def test_memory_persisted_after_cycle(
        self,
        sample_metrics: Dict[str, Any],
        tmp_memory_path: str,
        tmp_experiment_dir: str,
    ) -> None:
        from spectraquant.intelligence.research_lab import run_research_cycle

        run_research_cycle(
            metrics=sample_metrics,
            memory_path=tmp_memory_path,
            experiment_dir=tmp_experiment_dir,
        )
        assert Path(tmp_memory_path).exists()

    def test_status_and_history(
        self,
        sample_metrics: Dict[str, Any],
        tmp_memory_path: str,
        tmp_experiment_dir: str,
    ) -> None:
        from spectraquant.intelligence.research_lab import ResearchLoop

        loop = ResearchLoop(
            memory_path=tmp_memory_path,
            experiment_dir=tmp_experiment_dir,
        )
        loop.run(metrics=sample_metrics)
        status = loop.status()
        assert "n_hypotheses" in status
        history = loop.history()
        assert "hypotheses" in history
        assert "experiments" in history
        assert "evaluations" in history
