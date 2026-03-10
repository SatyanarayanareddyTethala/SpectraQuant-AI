"""Run tracker for SpectraQuant-AI-V3.

A :class:`RunTracker` records per-run metadata for a single experiment run
and persists it via a :class:`~spectraquant_v3.experiments.result_store.ResultStore`.

Typical usage::

    tracker = RunTracker(
        experiment_id="exp_001",
        strategy_id="crypto_momentum_v1",
        dataset_version="2024-Q1",
        config=cfg,
    )
    tracker.record_metrics({"sharpe": 1.3, "cagr": 0.18, "max_drawdown": -0.12})
    tracker.record_artefact("signals_report", "/reports/runs/run_abc/signals.json")
    tracker.save(store)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from spectraquant_v3.experiments.result_store import ResultStore


class RunTracker:
    """Tracks a single experiment run.

    Args:
        experiment_id:   Unique experiment identifier (e.g. ``"exp_001"``).
        strategy_id:     Strategy that was run (e.g. ``"crypto_momentum_v1"``).
        dataset_version: Optional dataset label for reproducibility.
        config:          Pipeline config dict (used to compute a config hash).
    """

    def __init__(
        self,
        experiment_id: str,
        strategy_id: str,
        dataset_version: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.strategy_id = strategy_id
        self.dataset_version = dataset_version
        self.config = config or {}

        self.run_timestamp = datetime.now(timezone.utc).isoformat()
        self.config_hash = self._hash_config(self.config)
        self.metrics: dict[str, Any] = {}
        self.artefact_paths: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge *metrics* into the tracked metrics dict.

        Standard keys (all optional): ``sharpe``, ``cagr``, ``volatility``,
        ``max_drawdown``, ``turnover``, ``win_rate``.

        Args:
            metrics: Dict of metric name → value.
        """
        self.metrics.update(metrics)

    def record_artefact(self, key: str, path: str) -> None:
        """Record a file artefact path for this run.

        Args:
            key:  Logical name (e.g. ``"signals_report"``).
            path: Absolute or relative path to the file.
        """
        self.artefact_paths[key] = path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, store: ResultStore) -> dict[str, Any]:
        """Persist the run record to *store*.

        Writes:
        - ``config.json``  — config snapshot + hash + run metadata
        - ``metrics.json`` — performance metrics

        Args:
            store: :class:`~spectraquant_v3.experiments.result_store.ResultStore`.

        Returns:
            A dict with the paths of all written files.
        """
        config_doc: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "strategy_id": self.strategy_id,
            "dataset_version": self.dataset_version,
            "config_hash": self.config_hash,
            "run_timestamp": self.run_timestamp,
            "metrics_payload": dict(self.metrics),
            "artefact_paths": self.artefact_paths,
        }
        paths: dict[str, Any] = {}
        paths["config"] = str(store.write_config(self.experiment_id, config_doc))
        paths["metrics"] = str(store.write_metrics(self.experiment_id, self.metrics))
        return paths

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_config(config: dict[str, Any]) -> str:
        """Return a short SHA-256 hex digest of a stable config serialisation."""

        def _normalise(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    str(k): _normalise(v)
                    for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
                }
            if isinstance(value, set):
                normalised_items = [_normalise(v) for v in value]
                return sorted(
                    normalised_items,
                    key=lambda item: json.dumps(
                        item,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                )
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [_normalise(v) for v in value]
            return value

        normalised = _normalise(config)
        serialised = json.dumps(
            normalised,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation of this run record."""
        return {
            "experiment_id": self.experiment_id,
            "strategy_id": self.strategy_id,
            "dataset_version": self.dataset_version,
            "config_hash": self.config_hash,
            "run_timestamp": self.run_timestamp,
            "metrics": dict(self.metrics),
            "artefact_paths": dict(self.artefact_paths),
        }
