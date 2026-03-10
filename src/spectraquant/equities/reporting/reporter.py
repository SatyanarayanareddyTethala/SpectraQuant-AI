"""Equity pipeline reporter.

Writes QA matrix, run manifest, and allocation reports.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EquityRunReport:
    """Complete report from a single equity pipeline run."""

    run_id: str
    asset_class: str = "equity"
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    symbols_requested: list[str] = field(default_factory=list)
    symbols_loaded: list[str] = field(default_factory=list)
    symbols_failed: list[str] = field(default_factory=list)
    qa_matrix: list[dict[str, Any]] = field(default_factory=list)
    allocation_weights: dict[str, float] = field(default_factory=dict)
    blocked_assets: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EquityReporter:
    """Write equity pipeline run reports to disk.

    Args:
        reports_dir: Directory for writing reports.
    """

    def __init__(self, reports_dir: str | Path = "reports/equities") -> None:
        self._reports_dir = Path(reports_dir)

    def write(self, report: EquityRunReport) -> Path:
        """Serialise *report* to a JSON file.

        Returns:
            Path to the written file.
        """
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = self._reports_dir / f"equity_run_{report.run_id}_{ts}.json"
        data = asdict(report)
        out_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Equity run report written to %s", out_path)
        return out_path

    def write_qa_matrix(
        self,
        qa: dict[str, dict[str, Any]],
        run_id: str,
    ) -> Path:
        """Write the per-symbol QA matrix to a separate file."""
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = self._reports_dir / f"equity_qa_{run_id}_{ts}.json"
        out_path.write_text(json.dumps(qa, indent=2, default=str))
        logger.info("Equity QA matrix written to %s (%d symbols)", out_path, len(qa))
        return out_path
