"""Health report for SpectraQuant-AI-V3 monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HealthReport:
    """Summary health report for a pipeline run.

    Attributes:
        run_id:        Identifier of the pipeline run.
        status:        Overall status: ``"ok"``, ``"degraded"``, or ``"failed"``.
        checks:        List of individual check result dicts.
        alerts:        List of alert message strings.
        metrics:       Quantitative health metrics.
    """

    run_id: str
    status: str = "ok"
    checks: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_check(
        self,
        name: str,
        passed: bool,
        message: str = "",
        value: Any = None,
    ) -> None:
        """Append a health check result."""
        self.checks.append(
            {"name": name, "passed": passed, "message": message, "value": value}
        )
        if not passed:
            self.alerts.append(f"[FAIL] {name}: {message}")
            if self.status == "ok":
                self.status = "degraded"

    def mark_failed(self, reason: str) -> None:
        """Unconditionally mark this report as failed."""
        self.status = "failed"
        self.alerts.append(f"[CRITICAL] {reason}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "checks": self.checks,
            "alerts": self.alerts,
            "metrics": self.metrics,
        }

    def write(self, output_dir: str | Path) -> Path:
        """Write the health report to a JSON file."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.run_id}_health.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
