"""Shared run orchestration used by both API and CLI entrypoints."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectraquant_v3.core import config as config_mod
from spectraquant_v3.core.enums import RunMode
from spectraquant_v3.core.errors import (
    ConfigValidationError,
    EmptyUniverseError,
    ManifestValidationError,
    SpectraQuantError,
)
from spectraquant_v3.core.failures import FailureCode, FailureDetail
from spectraquant_v3.core.manifest import RunManifest
from spectraquant_v3.diagnostics.invariants import list_run_artifacts, validate_result_invariants
from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline
from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline
from spectraquant_v3.service.models import RunSubmissionRequest, StageEvent
from spectraquant_v3.service.run_registry import RunRegistry

_RUN_LOCKS: dict[str, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()


@dataclass
class OrchestratedRun:
    run_id: str
    state: str
    error: FailureDetail | None
    result: dict[str, Any]


def _event(stage: str, status: str, message: str = "", **details: Any) -> StageEvent:
    return StageEvent(
        stage=stage,
        status=status,
        at=datetime.now(timezone.utc),
        message=message,
        details=details,
    )


def _classify_error(exc: Exception) -> FailureDetail:
    if isinstance(exc, ConfigValidationError):
        return FailureDetail(FailureCode.CONFIG_ERROR, str(exc), stage="config", retriable=False)
    if isinstance(exc, EmptyUniverseError):
        return FailureDetail(FailureCode.EMPTY_UNIVERSE, str(exc), stage="universe", retriable=False)
    if isinstance(exc, ManifestValidationError):
        return FailureDetail(FailureCode.MANIFEST_INVALID, str(exc), stage="reporting", retriable=False)
    if isinstance(exc, SpectraQuantError):
        return FailureDetail(FailureCode.PIPELINE_FAILURE, str(exc), stage="pipeline", retriable=False)
    return FailureDetail(FailureCode.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}", stage="internal", retriable=True)


def _load_config(req: RunSubmissionRequest) -> dict[str, Any]:
    cfg_loader = config_mod.get_crypto_config if req.asset_class == "crypto" else config_mod.get_equity_config
    cfg = cfg_loader(req.config_dir or None)

    # Hard boundary: live execution cannot run via research mode or dry-run paths.
    configured_execution_mode = str(cfg.get("execution", {}).get("mode", "paper")).lower()
    if req.execution_mode == "live":
        if configured_execution_mode != "live":
            raise ConfigValidationError(
                "LIVE execution requested but config execution.mode is not 'live'."
            )
        if req.run_mode == "test" or req.dry_run:
            raise ConfigValidationError(
                "LIVE execution cannot use run_mode=test or dry_run=true."
            )
    if req.execution_mode == "research" and configured_execution_mode == "live":
        raise ConfigValidationError(
            "Research execution request refuses config with execution.mode=live."
        )

    return cfg


def _acquire_lock(lock_key: str) -> threading.Lock | None:
    with _LOCK_GUARD:
        lock = _RUN_LOCKS.setdefault(lock_key, threading.Lock())
    if not lock.acquire(blocking=False):
        return None
    return lock


def execute_submission(req: RunSubmissionRequest, registry: RunRegistry) -> OrchestratedRun:
    run_id = req.run_id or str(uuid.uuid4())[:8]
    rec = registry.create_run(
        run_id=run_id,
        idempotency_key=req.idempotency_key,
        asset_class=req.asset_class,
        execution_mode=req.execution_mode,
        run_mode=req.run_mode,
        dry_run=req.dry_run,
    )
    if rec.run_id != run_id:
        return OrchestratedRun(run_id=rec.run_id, state=rec.state, error=None, result=rec.result)

    lock_key = f"{req.asset_class}:{req.execution_mode}"
    lock = _acquire_lock(lock_key)
    if lock is None:
        failure = FailureDetail(
            code=FailureCode.RUN_LOCKED,
            message=f"Another run for '{lock_key}' is already in progress.",
            stage="submission",
            retriable=True,
        )
        registry.update_state(run_id, state="failed", error_code=failure.code.value, error_message=failure.message)
        return OrchestratedRun(run_id=run_id, state="failed", error=failure, result={})

    registry.update_state(run_id, state="running")
    registry.append_event(run_id, _event("submission", "started", "Run accepted"))

    try:
        registry.append_event(run_id, _event("config", "started"))
        cfg = _load_config(req)
        registry.append_event(run_id, _event("config", "ok", execution_mode=req.execution_mode))

        run_mode = RunMode(req.run_mode)
        registry.append_event(run_id, _event("pipeline", "started"))
        if req.asset_class == "crypto":
            result = run_crypto_pipeline(cfg, run_mode=run_mode, dry_run=req.dry_run, run_id=run_id)
        else:
            result = run_equity_pipeline(cfg, run_mode=run_mode, dry_run=req.dry_run, run_id=run_id)
        registry.append_event(run_id, _event("pipeline", "ok", pipeline_status=result.get("status", "")))

        registry.append_event(run_id, _event("invariants", "started"))
        violations = validate_result_invariants(result)
        if violations:
            failure = FailureDetail(
                code=violations[0].code,
                message=violations[0].message,
                stage=violations[0].stage,
                retriable=False,
            )
            registry.append_event(
                run_id,
                _event(
                    "invariants",
                    "failed",
                    message="Invariant validation failed",
                    violations=[v.__dict__ for v in violations],
                ),
            )
            registry.update_state(
                run_id,
                state="failed",
                error_code=failure.code.value,
                error_message=failure.message,
                result={"violations": [v.__dict__ for v in violations]},
            )
            return OrchestratedRun(run_id=run_id, state="failed", error=failure, result={"violations": [v.__dict__ for v in violations]})

        registry.append_event(run_id, _event("invariants", "ok"))

        run_dir = _resolve_run_dir(req.asset_class, run_id, result)
        artifacts = list_run_artifacts(run_dir)
        envelope = {
            "pipeline": result,
            "artifacts": artifacts,
            "run_dir": str(run_dir),
        }
        registry.update_state(run_id, state="success", result=envelope)
        return OrchestratedRun(run_id=run_id, state="success", error=None, result=envelope)

    except Exception as exc:  # noqa: BLE001
        failure = _classify_error(exc)
        registry.append_event(run_id, _event(failure.stage or "pipeline", "failed", message=failure.message))
        registry.update_state(
            run_id,
            state="failed",
            error_code=failure.code.value,
            error_message=failure.message,
        )
        return OrchestratedRun(run_id=run_id, state="failed", error=failure, result={})
    finally:
        lock.release()


def _resolve_run_dir(asset_class: str, run_id: str, result: dict[str, Any]) -> Path:
    artefact_paths = result.get("artefact_paths") or {}
    if artefact_paths:
        first_path = Path(next(iter(artefact_paths.values())))
        return first_path.parent

    reports_root = Path("reports")
    parent = "crypto" if asset_class == "crypto" else "equities"
    return reports_root / parent / run_id


def load_manifest_for_run(run_record: dict[str, Any]) -> dict[str, Any]:
    """Load persisted manifest payload for diagnostics endpoint."""
    run_id = run_record["run_id"]
    asset_class = run_record["asset_class"]
    manifest_parent = Path("reports") / ("crypto" if asset_class == "crypto" else "equities") / run_id
    files = sorted(manifest_parent.glob("run_manifest_*.json"))
    if not files:
        return {}
    manifest = RunManifest.from_file(files[-1])
    return {
        "run_id": manifest.run_id,
        "asset_class": manifest.asset_class.value,
        "run_mode": manifest.run_mode.value,
        "status": manifest.status.value,
        "started_at": manifest.started_at,
        "completed_at": manifest.completed_at,
        "stages": manifest.stages,
        "errors": manifest.errors,
    }
