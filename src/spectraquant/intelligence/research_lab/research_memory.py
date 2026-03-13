"""Research Memory — long-term brain for the autonomous research system.

Stores hypotheses, experiments, successes, failures, and market context
as a structured JSON file.  Prevents repeating failed ideas and acts as
a scientist's notebook.

Persistence path : ``data/intelligence/research_memory.json``
"""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from spectraquant.intelligence.research_lab.hypothesis_engine import Hypothesis
from spectraquant.intelligence.research_lab.experiment_runner import ExperimentResult
from spectraquant.intelligence.research_lab.evaluator import EvaluationReport

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "data/intelligence/research_memory.json"

_EMPTY_MEMORY: Dict[str, Any] = {
    "version": "1",
    "created_at": "",
    "updated_at": "",
    "hypotheses": [],
    "experiments": [],
    "evaluations": [],
    "successes": [],
    "failures": [],
    "market_context": [],
}


class ResearchMemory:
    """Persistent research memory backed by a JSON file.

    Parameters
    ----------
    path : str
        Path to the memory JSON file.
    """

    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_hypothesis(self, hyp: Hypothesis) -> None:
        """Append a hypothesis record (idempotent by hypothesis_id)."""
        existing_ids = {h["hypothesis_id"] for h in self._data["hypotheses"]}
        if hyp.hypothesis_id in existing_ids:
            logger.debug("Hypothesis %s already in memory; skipping", hyp.hypothesis_id)
            return
        self._data["hypotheses"].append(hyp.to_dict())
        self._touch()
        logger.debug("Stored hypothesis %s", hyp.hypothesis_id)

    def store_experiment(self, result: ExperimentResult) -> None:
        """Append an experiment result."""
        self._data["experiments"].append(result.to_dict())
        self._touch()
        logger.debug("Stored experiment %s", result.experiment_id)

    def store_evaluation(self, report: EvaluationReport) -> None:
        """Append an evaluation report; also bucket into successes/failures."""
        self._data["evaluations"].append(report.to_dict())
        if report.accepted:
            self._data["successes"].append(report.experiment_id)
        else:
            self._data["failures"].append(report.experiment_id)
        self._touch()
        logger.debug(
            "Stored evaluation %s (accepted=%s)", report.experiment_id, report.accepted
        )

    def store_market_context(self, context: Dict[str, Any]) -> None:
        """Append a market context snapshot."""
        context["stored_at"] = datetime.now(tz=timezone.utc).isoformat()
        self._data["market_context"].append(context)
        # Keep only the last 30 context entries to bound size
        self._data["market_context"] = self._data["market_context"][-30:]
        self._touch()

    def is_hypothesis_known(self, hypothesis_id: str) -> bool:
        """Return True if the hypothesis is already in memory."""
        return any(
            h["hypothesis_id"] == hypothesis_id for h in self._data["hypotheses"]
        )

    def is_experiment_failed(self, strategy_name: str) -> bool:
        """Return True if a strategy with this name previously failed evaluation."""
        failed_exp_ids = set(self._data.get("failures", []))
        for ev in self._data.get("evaluations", []):
            if ev.get("strategy_name") == strategy_name and ev.get("experiment_id") in failed_exp_ids:
                return True
        return False

    def get_hypotheses(self) -> List[Dict[str, Any]]:
        return list(self._data["hypotheses"])

    def get_experiments(self) -> List[Dict[str, Any]]:
        return list(self._data["experiments"])

    def get_evaluations(self) -> List[Dict[str, Any]]:
        return list(self._data["evaluations"])

    def get_successes(self) -> List[str]:
        return list(self._data["successes"])

    def get_failures(self) -> List[str]:
        return list(self._data["failures"])

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary dict."""
        return {
            "n_hypotheses": len(self._data["hypotheses"]),
            "n_experiments": len(self._data["experiments"]),
            "n_evaluations": len(self._data["evaluations"]),
            "n_successes": len(self._data["successes"]),
            "n_failures": len(self._data["failures"]),
            "last_updated": self._data.get("updated_at", ""),
        }

    def save(self) -> None:
        """Persist the in-memory state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._path, "w") as fh:
                json.dump(self._data, fh, indent=2)
            logger.debug("Research memory saved to %s", self._path)
        except OSError as exc:
            logger.error("Failed to save research memory: %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path) as fh:
                    data = json.load(fh)
                # Ensure all keys are present (backwards compat)
                for key, default in _EMPTY_MEMORY.items():
                    data.setdefault(key, default)
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load research memory (%s); starting fresh", exc)

        memory = copy.deepcopy(_EMPTY_MEMORY)
        memory["created_at"] = datetime.now(tz=timezone.utc).isoformat()
        return memory

    def _touch(self) -> None:
        self._data["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
