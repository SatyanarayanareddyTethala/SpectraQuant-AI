"""FastAPI control plane for SpectraQuant V3 orchestration."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from spectraquant_v3.core.failures import FailureCode
from spectraquant_v3.diagnostics.invariants import list_run_artifacts
from spectraquant_v3.service.models import ApiEnvelope, RunSubmissionRequest
from spectraquant_v3.service.orchestrator import execute_submission, load_manifest_for_run
from spectraquant_v3.service.run_registry import RunRegistry


def create_app(registry_path: str | Path = "reports/control_plane/run_registry.sqlite") -> FastAPI:
    registry = RunRegistry(registry_path)
    app = FastAPI(title="SpectraQuant V3 Control Plane", version="1.0.0")

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
    def submit_run(request: RunSubmissionRequest) -> ApiEnvelope:
        outcome = execute_submission(request, registry)
        if outcome.error is not None:
            return ApiEnvelope(
                ok=False,
                error={
                    "code": outcome.error.code.value,
                    "message": outcome.error.message,
                    "stage": outcome.error.stage,
                    "retriable": outcome.error.retriable,
                    "run_id": outcome.run_id,
                },
            )
        return ApiEnvelope(ok=True, data={"run_id": outcome.run_id, "state": outcome.state, "result": outcome.result})

    @app.get("/runs", response_model=ApiEnvelope)
    def list_runs(limit: int = 50) -> ApiEnvelope:
        rows = [row.model_dump(mode="json") for row in registry.list_runs(limit=limit)]
        return ApiEnvelope(ok=True, data={"runs": rows})

    @app.get("/runs/{run_id}", response_model=ApiEnvelope)
    def get_run(run_id: str) -> ApiEnvelope:
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
    def get_run_artifacts(run_id: str) -> ApiEnvelope:
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
    """Run control-plane API with uvicorn."""
    import uvicorn

    uvicorn.run("spectraquant_v3.service.api:app", host="0.0.0.0", port=8000, reload=False)
