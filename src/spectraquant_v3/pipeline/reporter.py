"""Pipeline reporter for SpectraQuant-AI-V3.

Writes summary artefacts after each pipeline run:
- ``signals_summary.json``  – per-symbol signal scores and statuses.
- ``allocation_summary.json`` – target weights and blocked symbols.
- ``run_report.json``        – high-level run statistics.

The reporter is stateless – pass it the outputs from all pipeline stages
and it serialises them to disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectraquant_v3.core.schema import AllocationRow, SignalRow
from spectraquant_v3.pipeline.meta_policy import PolicyDecision


class PipelineReporter:
    """Writes JSON summary artefacts for a pipeline run.

    Args:
        run_id:     Parent run identifier.
        output_dir: Directory to write all artefact files.
        asset_class: Asset class string (``'crypto'`` or ``'equity'``).
    """

    def __init__(
        self,
        run_id: str,
        output_dir: str | Path,
        asset_class: str,
    ) -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.asset_class = asset_class
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Individual artefact writers
    # ------------------------------------------------------------------

    def write_signals(self, signals: list[SignalRow]) -> Path:
        """Write signal summary to ``signals_summary_<run_id>.json``.

        Returns:
            Path of the written file.
        """
        path = self.output_dir / f"signals_summary_{self.run_id}.json"
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "asset_class": self.asset_class,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_signals": len(signals),
            "ok_count": sum(1 for s in signals if s.status == "OK"),
            "no_signal_count": sum(1 for s in signals if s.status == "NO_SIGNAL"),
            "error_count": sum(1 for s in signals if s.status == "ERROR"),
            "signals": [asdict(s) for s in signals],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def write_policy_decisions(self, decisions: list[PolicyDecision]) -> Path:
        """Write meta-policy decisions to ``policy_decisions_<run_id>.json``."""
        path = self.output_dir / f"policy_decisions_{self.run_id}.json"
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "asset_class": self.asset_class,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_symbols": len(decisions),
            "passed_count": sum(1 for d in decisions if d.passed),
            "blocked_count": sum(1 for d in decisions if not d.passed),
            "decisions": [asdict(d) for d in decisions],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def write_allocation(self, rows: list[AllocationRow]) -> Path:
        """Write allocation summary to ``allocation_summary_<run_id>.json``."""
        path = self.output_dir / f"allocation_summary_{self.run_id}.json"
        active = [r for r in rows if not r.blocked]
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "asset_class": self.asset_class,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_symbols": len(rows),
            "active_positions": len(active),
            "blocked_positions": len(rows) - len(active),
            "total_weight": round(sum(r.target_weight for r in active), 6),
            "allocations": [asdict(r) for r in rows],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def write_run_report(
        self,
        universe_size: int,
        signals: list[SignalRow],
        decisions: list[PolicyDecision],
        allocations: list[AllocationRow],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write high-level run report to ``run_report_<run_id>.json``.

        Args:
            universe_size: Number of symbols in the trading universe.
            signals:       All signal rows produced this run.
            decisions:     Meta-policy decisions.
            allocations:   Allocation rows.
            extra:         Optional additional metadata to embed.

        Returns:
            Path of the written file.
        """
        path = self.output_dir / f"run_report_{self.run_id}.json"
        active = [r for r in allocations if not r.blocked]
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "asset_class": self.asset_class,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "universe_size": universe_size,
            "signals_ok": sum(1 for s in signals if s.status == "OK"),
            "signals_no_signal": sum(1 for s in signals if s.status == "NO_SIGNAL"),
            "signals_error": sum(1 for s in signals if s.status == "ERROR"),
            "policy_passed": sum(1 for d in decisions if d.passed),
            "policy_blocked": sum(1 for d in decisions if not d.passed),
            "active_positions": len(active),
            "total_weight": round(sum(r.target_weight for r in active), 6),
        }
        if extra:
            payload.update(extra)
        path.write_text(json.dumps(payload, indent=2))
        return path

    # ------------------------------------------------------------------
    # Convenience: write all artefacts at once
    # ------------------------------------------------------------------

    def write_all(
        self,
        universe_symbols: list[str],
        signals: list[SignalRow],
        decisions: list[PolicyDecision],
        allocations: list[AllocationRow],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Write signals, policy decisions, allocation, and run report.

        Returns:
            Dict mapping artefact name to written path.
        """
        return {
            "signals": self.write_signals(signals),
            "policy_decisions": self.write_policy_decisions(decisions),
            "allocation": self.write_allocation(allocations),
            "run_report": self.write_run_report(
                universe_size=len(universe_symbols),
                signals=signals,
                decisions=decisions,
                allocations=allocations,
                extra=extra,
            ),
        }
