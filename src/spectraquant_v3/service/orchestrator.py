"""Shared run orchestration used by both API and CLI entrypoints."""

from __future__ import annotations

import hashlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectraquant_v3.core import config as config_mod
from spectraquant_v3.core.enums import RunMode
from spectraquant_v3.core.errors import ConfigValidationError, EmptyUniverseError, ManifestValidationError, SpectraQuantError
from spectraquant_v3.core.failures import FailureCode, FailureDetail
from spectraquant_v3.core.manifest import RunManifest
from spectraquant_v3.diagnostics.invariants import list_run_artifacts, validate_result_invariants
from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline
from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline
from spectraquant_v3.service.locking import InMemoryLockManager, LockManager
from spectraquant_v3.service.models import RunSubmissionRequest, StageEvent
from spectraquant_v3.service.run_registry import InvalidTransitionError, RunRegistry
from spectraquant_v3.service.worker import InMemoryWorker, RunTask


@dataclass
class OrchestratedRun:
    run_id: str
    state: str
    error: FailureDetail | None
    result: dict[str, Any]


def _event(stage: str, status: str, message: str = "", **details: Any) -> StageEvent:
    return StageEvent(stage=stage, status=status, at=datetime.now(timezone.utc), message=message, details=details)




def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)

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
    configured_execution_mode = str(cfg.get("execution", {}).get("mode", "paper")).lower()
    if req.execution_mode == "live":
        if configured_execution_mode != "live":
            raise ConfigValidationError("LIVE execution requested but config execution.mode is not 'live'.")
        if req.run_mode == "test" or req.dry_run:
            raise ConfigValidationError("LIVE execution cannot use run_mode=test or dry_run=true.")
    if req.execution_mode == "research" and configured_execution_mode == "live":
        raise ConfigValidationError("Research execution request refuses config with execution.mode=live.")
    return cfg


class ControlPlaneOrchestrator:
    def __init__(
        self,
        registry: RunRegistry,
        *,
        lock_manager: LockManager | None = None,
        timeout_seconds: int = 900,
    ) -> None:
        self.registry = registry
        self.lock_manager = lock_manager or InMemoryLockManager()
        self.timeout_seconds = timeout_seconds
        self._requests: dict[str, RunSubmissionRequest] = {}
        self._worker = InMemoryWorker(self._handle_task)

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self._worker.stop()

    def submit(self, req: RunSubmissionRequest, *, actor: str = "system") -> OrchestratedRun:
        run_id = req.run_id or str(uuid.uuid4())[:8]
        rec = self.registry.create_run(
            run_id=run_id,
            idempotency_key=req.idempotency_key,
            asset_class=req.asset_class,
            execution_mode=req.execution_mode,
            run_mode=req.run_mode,
            dry_run=req.dry_run,
        )
        if rec.run_id != run_id:
            return OrchestratedRun(run_id=rec.run_id, state=rec.state, error=None, result=rec.result)

        self._requests[run_id] = req
        self.registry.append_event(run_id, _event("submission", "started", "Run queued"))
        self.registry.audit(actor=actor, action="submit_run", run_id=run_id, execution_mode=req.execution_mode, outcome="accepted", details={"asset_class": req.asset_class})
        self._worker.submit(RunTask(run_id=run_id))
        return OrchestratedRun(run_id=run_id, state="queued", error=None, result={})

    def cancel(self, run_id: str, *, actor: str = "system", reason: str = "") -> OrchestratedRun:
        rec = self.registry.request_cancellation(run_id, reason=reason)
        self.registry.append_event(run_id, _event("submission", "failed", "Cancellation requested", reason=reason), allow_terminal=True)
        self.registry.audit(actor=actor, action="cancel_run", run_id=run_id, execution_mode=rec.execution_mode, outcome="requested", details={"state": rec.state})
        return OrchestratedRun(run_id=run_id, state=rec.state, error=None, result=rec.result)

    def _handle_task(self, task: RunTask) -> None:
        req = self._requests.get(task.run_id)
        if req is None:
            return
        rec = self.registry.get_run(task.run_id)
        if rec is None or rec.state in {"cancelled", "failed", "success", "timed_out"}:
            return
        if rec.state == "cancelling":
            self.registry.transition_state(task.run_id, to_state="cancelled", terminal_reason="cancelled-before-start")
            return

        lock_key = f"{req.asset_class}:{req.execution_mode}"
        lease = self.lock_manager.try_acquire(lock_key)
        if lease is None:
            self.registry.transition_state(
                task.run_id,
                to_state="failed",
                error_code=FailureCode.RUN_LOCKED.value,
                error_message=f"Another run for '{lock_key}' is already in progress.",
                terminal_reason="run-lock-contention",
            )
            self.registry.append_event(task.run_id, _event("submission", "failed", "Run lock contention"), allow_terminal=True)
            return

        try:
            try:
                self.registry.transition_state(task.run_id, to_state="running")
            except InvalidTransitionError:
                return
            self.registry.append_event(task.run_id, _event("submission", "ok", "Run started"))

            self.registry.append_event(task.run_id, _event("config", "started"))
            cfg = _load_config(req)
            self.registry.append_event(task.run_id, _event("config", "ok", execution_mode=req.execution_mode))

            if self.registry.get_run(task.run_id).state == "cancelling":
                self.registry.transition_state(task.run_id, to_state="cancelled", terminal_reason="cancelled-pre-pipeline")
                self.registry.append_event(task.run_id, _event("pipeline", "failed", "Run cancelled before pipeline execution"), allow_terminal=True)
                return

            run_mode = RunMode(req.run_mode)
            self.registry.append_event(task.run_id, _event("pipeline", "started"))
            with ThreadPoolExecutor(max_workers=1) as pool:
                fn = run_crypto_pipeline if req.asset_class == "crypto" else run_equity_pipeline
                fut = pool.submit(fn, cfg, run_mode, req.dry_run, task.run_id)
                try:
                    result = fut.result(timeout=self.timeout_seconds)
                except TimeoutError:
                    self.registry.append_event(task.run_id, _event("pipeline", "failed", "Pipeline timed out"))
                    self.registry.transition_state(
                        task.run_id,
                        to_state="timed_out",
                        error_code=FailureCode.INTERNAL_ERROR.value,
                        error_message=f"Run exceeded timeout of {self.timeout_seconds}s",
                        terminal_reason="timeout",
                    )
                    return
            self.registry.append_event(task.run_id, _event("pipeline", "ok", pipeline_status=result.get("status", "")))

            self.registry.append_event(task.run_id, _event("invariants", "started"))
            violations = validate_result_invariants(result)
            if violations:
                failure = FailureDetail(code=violations[0].code, message=violations[0].message, stage=violations[0].stage, retriable=False)
                payload = _json_safe({"violations": [v.__dict__ for v in violations]})
                self.registry.append_event(task.run_id, _event("invariants", "failed", message="Invariant validation failed", violations=payload["violations"]))
                self.registry.transition_state(
                    task.run_id,
                    to_state="failed",
                    error_code=failure.code.value,
                    error_message=failure.message,
                    result=payload,
                    terminal_reason="invariant-violation",
                )
                return
            self.registry.append_event(task.run_id, _event("invariants", "ok"))

            if self.registry.get_run(task.run_id).state == "cancelling":
                self.registry.transition_state(task.run_id, to_state="cancelled", terminal_reason="cancelled-post-pipeline")
                self.registry.append_event(task.run_id, _event("pipeline", "failed", "Run cancelled"), allow_terminal=True)
                return

            run_dir = _resolve_run_dir(req.asset_class, task.run_id, result)
            artifacts = list_run_artifacts(run_dir)
            envelope = _json_safe({"pipeline": result, "artifacts": artifacts, "run_dir": str(run_dir)})
            self.registry.transition_state(task.run_id, to_state="success", result=envelope, terminal_reason="completed")
        except (InvalidTransitionError, KeyError):
            raise
        except Exception as exc:  # noqa: BLE001
            failure = _classify_error(exc)
            self.registry.append_event(task.run_id, _event(failure.stage or "pipeline", "failed", message=failure.message))
            try:
                self.registry.transition_state(
                    task.run_id,
                    to_state="failed",
                    error_code=failure.code.value,
                    error_message=failure.message,
                    terminal_reason="exception",
                )
            except InvalidTransitionError:
                pass
        finally:
            self.lock_manager.release(lease)


def _resolve_run_dir(asset_class: str, run_id: str, result: dict[str, Any]) -> Path:
    artefact_paths = result.get("artefact_paths") or {}
    if artefact_paths:
        first_path = Path(next(iter(artefact_paths.values())))
        return first_path.parent
    reports_root = Path("reports")
    parent = "crypto" if asset_class == "crypto" else "equities"
    return reports_root / parent / run_id


def load_manifest_for_run(run_record: dict[str, Any]) -> dict[str, Any]:
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


def validate_live_approval(req: RunSubmissionRequest) -> None:
    if req.execution_mode != "live":
        return
    expected = os.getenv("SQ_LIVE_APPROVAL_TOKEN", "")
    if not expected or not req.approval_token:
        raise ConfigValidationError("LIVE execution requires approval token.")
    digest_in = hashlib.sha256(req.approval_token.encode("utf-8")).hexdigest()
    digest_exp = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    if digest_in != digest_exp:
        raise ConfigValidationError("Invalid LIVE approval token.")
