# SpectraQuant V3 ↔ n8n Integration Blueprint

## Architecture

- **n8n owns orchestration**: schedule, approvals, retries, alerts, incident routing.
- **SpectraQuant V3 owns quant logic**: universe, ingestion QA, signals, policy, allocation.
- Control plane service (`spectraquant_v3.service.api`) provides stable machine interfaces.

## Endpoint Map

Base service: `uvicorn spectraquant_v3.service.api:app --host 0.0.0.0 --port 8000`

- `GET /health` — liveness
- `GET /doctor` — environment/config readiness
- `POST /runs` — submit run (idempotent by key)
- `GET /runs` — list recent runs
- `GET /runs/{run_id}` — run status, events, manifest
- `GET /runs/{run_id}/artifacts` — indexed artifacts

## Request/Response Contracts

### Submit run

```json
{
  "asset_class": "equity",
  "execution_mode": "research",
  "run_mode": "normal",
  "dry_run": false,
  "idempotency_key": "n8n-2026-03-24-equity-open"
}
```

Success envelope:

```json
{
  "ok": true,
  "data": {
    "run_id": "abc123ef",
    "state": "success",
    "result": {
      "pipeline": {"status": "success"},
      "artifacts": [],
      "run_dir": "reports/equities/abc123ef"
    }
  },
  "error": {}
}
```

Failure envelope:

```json
{
  "ok": false,
  "data": {},
  "error": {
    "code": "EMPTY_UNIVERSE",
    "message": "...",
    "stage": "universe",
    "retriable": false,
    "run_id": "abc123ef"
  }
}
```

## Recommended n8n Workflows

1. **Pre-flight**: Cron → `/doctor` → fail fast if checks false.
2. **Research run**: Schedule → Approval (optional) → `POST /runs` (`execution_mode=research`).
3. **Paper execution**: Trigger from successful research run and policy gates.
4. **Live execution gate**: Manual approval + risk checklist + `execution_mode=live`.
5. **Incident response**: On `ok=false`, branch by `error.code`; create incident ticket, attach `/runs/{run_id}` and `/artifacts`.
6. **Self-heal loop**: Retry only when `retriable=true`; hard stop otherwise.

## Failure Branching Model

- `RUN_LOCKED`: retry with exponential backoff.
- `CONFIG_ERROR`, `RUN_MODE_GUARD`, `MANIFEST_INVALID`: stop + human intervention.
- `EMPTY_UNIVERSE`, `EMPTY_DATASET`, `DATE_ALIGNMENT`, `TIMESTAMP_RANGE`: data-quality incident; run diagnostics.
- `INTERNAL_ERROR`: one automatic retry then page on-call.

## Operational Guardrails

- Use unique idempotency keys per business run (`<workflow>-<date>-<asset>-<window>`).
- Keep `execution_mode=live` on dedicated n8n workflow with explicit approvals.
- Never reuse research workflow credentials for live broker actions.
- Persist run payloads from `/runs/{run_id}` for audit trails.
