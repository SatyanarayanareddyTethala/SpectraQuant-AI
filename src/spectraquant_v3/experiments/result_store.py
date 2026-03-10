"""Experiment result storage for SpectraQuant-AI-V3.

Persists experiment results to the ``reports/experiments/`` directory tree.

Each experiment occupies its own subdirectory::

    reports/
      experiments/
        exp_001/
          config.json
          metrics.json
          dataset_manifest.json
          backtest_summary.json

The :class:`ResultStore` is intentionally simple – it writes plain JSON files
so that results are readable by humans and standard tooling without any
database dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultStore:
    """Read and write experiment result files.

    Args:
        base_dir: Root directory for experiments.  Defaults to
                  ``reports/experiments`` relative to the current working
                  directory.  Absolute paths are accepted.
    """

    _DEFAULT_BASE = "reports/experiments"

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is None:
            base_dir = self._DEFAULT_BASE
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Experiment directory helpers
    # ------------------------------------------------------------------

    def experiment_dir(self, experiment_id: str) -> Path:
        """Return the directory path for *experiment_id* (does not create it)."""
        return self.base_dir / experiment_id

    def ensure_experiment_dir(self, experiment_id: str) -> Path:
        """Return and create the directory for *experiment_id*."""
        d = self.experiment_dir(experiment_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def write_config(self, experiment_id: str, config: dict[str, Any]) -> Path:
        """Persist the experiment configuration snapshot.

        Args:
            experiment_id: Unique experiment identifier.
            config:        Serialisable dict.

        Returns:
            Path to the written file.
        """
        return self._write_json(experiment_id, "config.json", config)

    def write_metrics(self, experiment_id: str, metrics: dict[str, Any]) -> Path:
        """Persist the experiment performance metrics.

        Expected keys: ``sharpe``, ``cagr``, ``volatility``,
        ``max_drawdown``, ``turnover``, ``win_rate`` (all optional).

        Returns:
            Path to the written file.
        """
        return self._write_json(experiment_id, "metrics.json", metrics)

    def write_dataset_manifest(
        self, experiment_id: str, manifest: dict[str, Any]
    ) -> Path:
        """Persist the dataset manifest (symbols, date range, version).

        Returns:
            Path to the written file.
        """
        return self._write_json(experiment_id, "dataset_manifest.json", manifest)

    def write_backtest_summary(
        self, experiment_id: str, summary: dict[str, Any]
    ) -> Path:
        """Persist a backtest performance summary.

        Returns:
            Path to the written file.
        """
        return self._write_json(experiment_id, "backtest_summary.json", summary)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def read_config(self, experiment_id: str) -> dict[str, Any]:
        """Load the configuration snapshot for *experiment_id*."""
        return self._read_json(experiment_id, "config.json")

    def read_metrics(self, experiment_id: str) -> dict[str, Any]:
        """Load the performance metrics for *experiment_id*."""
        return self._read_json(experiment_id, "metrics.json")

    def read_dataset_manifest(self, experiment_id: str) -> dict[str, Any]:
        """Load the dataset manifest for *experiment_id*."""
        return self._read_json(experiment_id, "dataset_manifest.json")

    def read_backtest_summary(self, experiment_id: str) -> dict[str, Any]:
        """Load the backtest summary for *experiment_id*."""
        return self._read_json(experiment_id, "backtest_summary.json")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_experiments(self) -> list[str]:
        """Return sorted experiment IDs found under ``base_dir``.

        Only directories that contain at least a ``config.json`` are returned.
        """
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_json(
        self, experiment_id: str, filename: str, data: dict[str, Any]
    ) -> Path:
        d = self.ensure_experiment_dir(experiment_id)
        path = d / filename
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def _read_json(self, experiment_id: str, filename: str) -> dict[str, Any]:
        path = self.experiment_dir(experiment_id) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"No '{filename}' found for experiment '{experiment_id}' "
                f"(expected at {path})."
            )
        return json.loads(path.read_text(encoding="utf-8"))
