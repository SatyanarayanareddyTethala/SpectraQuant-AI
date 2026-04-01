from __future__ import annotations

from pathlib import Path

import pytest

from spectraquant_v3.service.models import StageEvent
from spectraquant_v3.service.run_registry import InvalidTransitionError, RunRegistry


def _event() -> StageEvent:
    from datetime import datetime, timezone

    return StageEvent(stage="submission", status="started", at=datetime.now(timezone.utc), message="", details={})


def test_state_machine_blocks_invalid_transitions(tmp_path: Path) -> None:
    reg = RunRegistry(tmp_path / "registry.sqlite")
    rec = reg.create_run(
        run_id="r1",
        idempotency_key="idem-state-mach-001",
        asset_class="equity",
        execution_mode="research",
        run_mode="normal",
        dry_run=False,
    )
    assert rec.state == "queued"
    reg.transition_state("r1", to_state="running")
    reg.transition_state("r1", to_state="success", result={"ok": True})
    with pytest.raises(InvalidTransitionError):
        reg.transition_state("r1", to_state="failed", error_code="X", error_message="late")


def test_terminal_run_rejects_event_append(tmp_path: Path) -> None:
    reg = RunRegistry(tmp_path / "registry.sqlite")
    reg.create_run(
        run_id="r2",
        idempotency_key="idem-state-mach-002",
        asset_class="equity",
        execution_mode="research",
        run_mode="normal",
        dry_run=False,
    )
    reg.transition_state("r2", to_state="failed", error_code="ERR", error_message="boom")
    with pytest.raises(InvalidTransitionError):
        reg.append_event("r2", _event())


def test_cancellation_from_queue_is_terminal(tmp_path: Path) -> None:
    reg = RunRegistry(tmp_path / "registry.sqlite")
    reg.create_run(
        run_id="r3",
        idempotency_key="idem-state-mach-003",
        asset_class="crypto",
        execution_mode="research",
        run_mode="normal",
        dry_run=False,
    )
    rec = reg.request_cancellation("r3", reason="user")
    assert rec.state == "cancelled"
    assert rec.cancellation_requested_at is not None
