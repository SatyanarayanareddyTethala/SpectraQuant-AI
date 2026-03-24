from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from spectraquant_v3.diagnostics.invariants import compare_run_reports, validate_result_invariants


def test_invariant_detects_1970_timestamp_and_missing_artifacts() -> None:
    dataset = {
        "BTC": pd.DataFrame(
            {
                "timestamp": ["1970-01-01T00:00:00Z", "1970-01-02T00:00:00Z"],
                "close": [1.0, 2.0],
            }
        )
    }

    result = {
        "status": "success",
        "universe": ["BTC"],
        "signals": [],
        "allocations": [],
        "dataset": dataset,
        "artefact_paths": {"run_report": "x.json"},
    }
    violations = validate_result_invariants(result)
    codes = {v.code.value for v in violations}
    assert "TIMESTAMP_RANGE" in codes
    assert "ARTIFACT_MISSING" in codes


def test_compare_run_reports_returns_metric_delta(tmp_path: Path) -> None:
    healthy_path = tmp_path / "healthy.json"
    failed_path = tmp_path / "failed.json"

    healthy_path.write_text(
        json.dumps(
            {
                "run_id": "healthy01",
                "universe_size": 10,
                "signals_ok": 8,
                "signals_no_signal": 2,
                "signals_error": 0,
                "policy_passed": 7,
                "policy_blocked": 3,
                "active_positions": 7,
                "total_weight": 0.95,
            }
        )
    )
    failed_path.write_text(
        json.dumps(
            {
                "run_id": "failed01",
                "universe_size": 10,
                "signals_ok": 0,
                "signals_no_signal": 10,
                "signals_error": 0,
                "policy_passed": 0,
                "policy_blocked": 10,
                "active_positions": 0,
                "total_weight": 0.0,
            }
        )
    )

    report = compare_run_reports(healthy_path, failed_path)
    assert report["healthy_run_id"] == "healthy01"
    assert report["failed_run_id"] == "failed01"
    assert report["metrics"]["signals_ok"]["delta"] == -8
