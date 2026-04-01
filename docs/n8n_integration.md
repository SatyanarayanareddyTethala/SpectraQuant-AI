# SpectraQuant V3 ↔ n8n Integration Blueprint

## Architecture

- **n8n owns orchestration**: schedule, approvals, retries, alerts, incident routing.
- **SpectraQuant V3 owns quant logic**: universe, ingestion QA, signals, policy, allocation.
- Control plane service (`spectraquant_v3.service.api`) provides stable machine interfaces.
- **Execution model is now async**: `POST /runs` only queues a run; background worker executes and persists lifecycle events.

## Endpoint Map

Base service: `uvicorn spectraquant_v3.service.api:app --host 0.0.0.0 --port 8000`

- `GET /health` — liveness
- `GET /doctor` — environment/config readiness
- `POST /runs` — submit run (idempotent by key, async queued)
- `POST /runs/{run_id}/cancel` — request cancellation
- `GET /runs` — list recent runs
- `GET /runs/{run_id}` — run status, events, manifest
- `GET /runs/{run_id}/artifacts` — indexed artifacts

All control-plane endpoints require `X-API-Key`.

## Async Queue + Worker Design

1. API validates auth/scope/approval token and writes queued run in registry.
2. API returns immediately with `state=queued`.
3. Worker dequeues task, acquires orchestration lock, transitions to `running`.
4. Worker executes config → pipeline → invariant checks.
5. Worker transitions run to terminal (`success`, `failed`, `cancelled`, `timed_out`) and persists result.

### Run State Machine

Legal states:
- `queued`
- `running`
- `cancelling`
- `cancelled` (terminal)
- `success` (terminal)
- `failed` (terminal)
- `timed_out` (terminal)

Allowed transitions are strictly enforced in registry.

## Auth + Approval Model

### API Keys + Scopes

`SQ_CONTROL_PLANE_KEYS` format:

```text
a-key:research,paper;b-key:research,paper,live
```

- Scope must include requested `execution_mode`.
- Unauthorized/missing keys are rejected at API layer.

### Live Approval Gate

- Set `SQ_LIVE_APPROVAL_TOKEN` in control-plane environment.
- `POST /runs` with `execution_mode=live` must include `approval_token`.
- Invalid/missing approval token is denied and audited.

### Immutable Audit Ledger

Sensitive submission/cancellation attempts are inserted into `audit_log` (append-only table with update/delete triggers blocked).

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

Success envelope (submission accepted):

```json
{
  "ok": true,
  "data": {
    "run_id": "abc123ef",
    "state": "queued",
    "result": {}
  },
  "error": {}
}
```

## Cancellation Semantics

- `POST /runs/{run_id}/cancel` marks queued runs as `cancelled` immediately.
- Running runs transition to `cancelling`, then worker finalizes as `cancelled` at safe checkpoint.
- Terminal runs ignore cancellation mutation.

## Recommended n8n Polling + Retry Logic

1. Submit run (`POST /runs`) and persist `run_id`.
2. Poll `GET /runs/{run_id}` every 5–15s with capped backoff.
3. Stop polling when state enters terminal set.
4. Retry submission only when API-level error says `retriable=true`.
5. For `RUN_LOCKED` or transient `INTERNAL_ERROR`, use exponential backoff with jitter.
6. For non-retriable states (`CONFIG_ERROR`, invariants, scope/approval denials), route to incident workflow.

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
