"""Portfolio result for SpectraQuant-AI-V3 strategy portfolio layer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PortfolioResult:
    """Aggregated result from a multi-strategy portfolio run.

    Attributes:
        portfolio_id:      Unique identifier for this portfolio.
        strategy_ids:      List of constituent strategy IDs.
        weights:           Dict mapping strategy_id → portfolio weight.
        metrics:           Aggregate performance metrics.
        strategy_results:  Per-strategy result dicts.
        artifact_paths:    Paths to persisted output files.
    """

    portfolio_id: str
    strategy_ids: list[str]
    weights: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    strategy_results: dict[str, Any] = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "portfolio_id": self.portfolio_id,
            "strategy_ids": self.strategy_ids,
            "weights": self.weights,
            "metrics": self.metrics,
            "strategy_results": self.strategy_results,
            "artifact_paths": self.artifact_paths,
        }

    def write(self, output_dir: str | Path) -> Path:
        """Write result JSON to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.portfolio_id}_result.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
