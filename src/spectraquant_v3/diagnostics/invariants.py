"""Run-level invariant checks for deterministic orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from spectraquant_v3.core.failures import FailureCode


@dataclass(frozen=True)
class InvariantViolation:
    code: FailureCode
    message: str
    stage: str


def _extract_timestamp_series(frame: pd.DataFrame) -> pd.Series | None:
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.to_datetime(frame.index, utc=True, errors="coerce")
    return None


def validate_result_invariants(result: dict[str, Any]) -> list[InvariantViolation]:
    """Validate post-run invariants over pipeline result payload."""
    violations: list[InvariantViolation] = []

    universe = result.get("universe") or []
    if not universe:
        violations.append(
            InvariantViolation(
                code=FailureCode.EMPTY_UNIVERSE,
                stage="universe",
                message="Pipeline returned success payload with an empty universe.",
            )
        )

    signals = result.get("signals") or []
    signal_symbols = {getattr(s, "canonical_symbol", "") for s in signals}
    if universe and signal_symbols and signal_symbols - set(universe):
        violations.append(
            InvariantViolation(
                code=FailureCode.DATE_ALIGNMENT,
                stage="signals",
                message="Signal payload includes symbols outside declared universe.",
            )
        )

    allocations = result.get("allocations") or []
    active = [a for a in allocations if not getattr(a, "blocked", True)]
    if active:
        total_abs = sum(abs(float(getattr(a, "target_weight", 0.0))) for a in active)
        if total_abs == 0.0:
            violations.append(
                InvariantViolation(
                    code=FailureCode.ALLOCATION_ZERO,
                    stage="allocation",
                    message="All active allocations are zero weight.",
                )
            )

    dataset = result.get("dataset")
    if isinstance(dataset, dict):
        for symbol, frame in dataset.items():
            if isinstance(frame, pd.DataFrame) and frame.empty:
                violations.append(
                    InvariantViolation(
                        code=FailureCode.EMPTY_DATASET,
                        stage="features",
                        message=f"Dataset for symbol '{symbol}' is empty.",
                    )
                )

            if isinstance(frame, pd.DataFrame):
                series = _extract_timestamp_series(frame)
                if series is not None and not series.empty:
                    if series.isna().any():
                        violations.append(
                            InvariantViolation(
                                code=FailureCode.TIMESTAMP_RANGE,
                                stage="features",
                                message=f"Dataset for symbol '{symbol}' has unparsable timestamps.",
                            )
                        )
                    elif series.min().year < 2000:
                        violations.append(
                            InvariantViolation(
                                code=FailureCode.TIMESTAMP_RANGE,
                                stage="features",
                                message=(
                                    f"Dataset for symbol '{symbol}' contains suspicious early timestamp "
                                    f"{series.min().isoformat()} (possible epoch/1970 bug)."
                                ),
                            )
                        )

    artefact_paths = result.get("artefact_paths") or {}
    if result.get("status") == "success":
        required = {"signals", "allocation", "run_report", "policy_decisions", "diagnostics_summary"}
        missing = sorted(k for k in required if k not in artefact_paths)
        if missing:
            violations.append(
                InvariantViolation(
                    code=FailureCode.ARTIFACT_MISSING,
                    stage="reporting",
                    message=f"Successful run missing required artefacts: {missing}",
                )
            )

    return violations


def list_run_artifacts(run_dir: str | Path) -> list[dict[str, Any]]:
    """Index run artifacts for machine consumption."""
    root = Path(run_dir)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        records.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix,
            }
        )
    return records


def compare_run_reports(healthy_run_report: str | Path, failed_run_report: str | Path) -> dict[str, Any]:
    """Utility to compare healthy vs failed run summary metrics."""
    healthy = json.loads(Path(healthy_run_report).read_text())
    failed = json.loads(Path(failed_run_report).read_text())

    keys = [
        "universe_size",
        "signals_ok",
        "signals_no_signal",
        "signals_error",
        "policy_passed",
        "policy_blocked",
        "active_positions",
        "total_weight",
    ]
    delta = {
        key: {
            "healthy": healthy.get(key),
            "failed": failed.get(key),
            "delta": (failed.get(key, 0) - healthy.get(key, 0)) if isinstance(healthy.get(key), (int, float)) and isinstance(failed.get(key), (int, float)) else None,
        }
        for key in keys
    }
    return {
        "healthy_run_id": healthy.get("run_id", ""),
        "failed_run_id": failed.get("run_id", ""),
        "metrics": delta,
    }
