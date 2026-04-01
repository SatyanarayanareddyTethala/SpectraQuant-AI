"""FastAPI control plane for SpectraQuant V3 orchestration."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException

from spectraquant_v3.core.failures import FailureCode
from spectraquant_v3.diagnostics.invariants import list_run_artifacts
from spectraquant_v3.service.models import ApiEnvelope, RunSubmissionRequest
from spectraquant_v3.service.orchestrator import ControlPlaneOrchestrator, load_manifest_for_run, validate_live_approval
from spectraquant_v3.service.run_registry import RunRegistry


def _load_key_permissions() -> dict[str, set[str]]:
    raw = os.getenv("SQ_CONTROL_PLANE_KEYS", "")
    if not raw:
        return {"dev-research-key": {"research", "paper"}, "dev-live-key": {"research", "paper", "live"}}
    mapping: dict[str, set[str]] = {}
    for pair in raw.split(";"):
        if not pair.strip() or ":" not in pair:
            continue
        key, scopes = pair.split(":", 1)
        mapping[key.strip()] = {s.strip() for s in scopes.split(",") if s.strip()}
    return mapping


def create_app(registry_path: str | Path = "reports/control_plane/run_registry.sqlite") -> FastAPI:
    registry = RunRegistry(registry_path)
    orchestrator = ControlPlaneOrchestrator(registry)
    key_permissions = _load_key_permissions()

    app = FastAPI(title="SpectraQuant V3 Control Plane", version="1.1.0")

    @app.on_event("startup")
    def _startup() -> None:
        orchestrator.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        orchestrator.stop()

    def _require_key(x_api_key: str | None = Header(default=None)) -> str:
        if not x_api_key:
            raise HTTPException(status_code=401, detail={"code": FailureCode.UNAUTHORIZED.value, "message": "Missing X-API-Key"})
        if x_api_key not in key_permissions:
            raise HTTPException(status_code=403, detail={"code": FailureCode.FORBIDDEN.value, "message": "Unknown API key"})
        return x_api_key

    @app.get("/health", response_model=ApiEnvelope)
    def health() -> ApiEnvelope:
        return ApiEnvelope(ok=True, data={"status": "ok", "service": "spectraquant-v3-control-plane"})

    @app.get("/doctor", response_model=ApiEnvelope)
    def doctor() -> ApiEnvelope:
        checks = {
            "registry_db_exists": Path(registry_path).exists(),
            "reports_dir_exists": Path("reports").exists(),
            "config_base_exists": Path("config/v3/base.yaml").exists(),
            "config_crypto_exists": Path("config/v3/crypto.yaml").exists(),
            "config_equities_exists": Path("config/v3/equities.yaml").exists(),
        }
        ok = all(checks.values())
        return ApiEnvelope(ok=ok, data={"checks": checks})

    @app.post("/runs", response_model=ApiEnvelope)
    def submit_run(request: RunSubmissionRequest, x_api_key: str | None = Header(default=None)) -> ApiEnvelope:
        api_key = _require_key(x_api_key)
        scopes = key_permissions[api_key]
        if request.execution_mode not in scopes:
            registry.audit(actor=api_key, action="submit_run", run_id=request.run_id or "pending", execution_mode=request.execution_mode, outcome="denied", details={"reason": "insufficient_scope"})
            return ApiEnvelope(ok=False, error={"code": FailureCode.FORBIDDEN.value, "message": f"API key lacks scope for execution_mode={request.execution_mode}", "retriable": False})
        try:
            validate_live_approval(request)
        except Exception as exc:  # noqa: BLE001
            registry.audit(actor=api_key, action="submit_run", run_id=request.run_id or "pending", execution_mode=request.execution_mode, outcome="denied", details={"reason": str(exc)})
            return ApiEnvelope(ok=False, error={"code": FailureCode.CONFIG_ERROR.value, "message": str(exc), "stage": "config", "retriable": False})

        outcome = orchestrator.submit(request, actor=api_key)
        return ApiEnvelope(ok=True, data={"run_id": outcome.run_id, "state": outcome.state, "result": outcome.result})

    @app.post("/runs/{run_id}/cancel", response_model=ApiEnvelope)
    def cancel_run(run_id: str, x_api_key: str | None = Header(default=None)) -> ApiEnvelope:
        api_key = _require_key(x_api_key)
        outcome = orchestrator.cancel(run_id, actor=api_key)
        return ApiEnvelope(ok=True, data={"run_id": run_id, "state": outcome.state})

    @app.get("/runs", response_model=ApiEnvelope)
    def list_runs(limit: int = 50, x_api_key: str | None = Header(default=None)) -> ApiEnvelope:
        _require_key(x_api_key)
        rows = [row.model_dump(mode="json") for row in registry.list_runs(limit=limit)]
        return ApiEnvelope(ok=True, data={"runs": rows})

    @app.get("/runs/{run_id}", response_model=ApiEnvelope)
    def get_run(run_id: str, x_api_key: str | None = Header(default=None)) -> ApiEnvelope:
        _require_key(x_api_key)
        row = registry.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail={"code": FailureCode.INVALID_REQUEST.value, "message": f"Unknown run_id '{run_id}'"})

        manifest = load_manifest_for_run(row.model_dump())
        events = [event.model_dump(mode="json") for event in registry.list_events(run_id)]
        payload = row.model_dump(mode="json")
        payload["events"] = events
        payload["manifest"] = manifest
        return ApiEnvelope(ok=True, data=payload)

    @app.get("/runs/{run_id}/artifacts", response_model=ApiEnvelope)
    def get_run_artifacts(run_id: str, x_api_key: str | None = Header(default=None)) -> ApiEnvelope:
        _require_key(x_api_key)
        row = registry.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail={"code": FailureCode.INVALID_REQUEST.value, "message": f"Unknown run_id '{run_id}'"})

        run_dir = row.result.get("run_dir") if isinstance(row.result, dict) else None
        if not run_dir:
            parent = "crypto" if row.asset_class == "crypto" else "equities"
            run_dir = str(Path("reports") / parent / run_id)
        artifacts = list_run_artifacts(run_dir)
        return ApiEnvelope(ok=True, data={"run_id": run_id, "run_dir": run_dir, "artifacts": artifacts})

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("spectraquant_v3.service.api:app", host="0.0.0.0", port=8000, reload=False)
