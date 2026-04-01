from __future__ import annotations

import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from spectraquant_v3.service.api import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("SQ_CONTROL_PLANE_KEYS", raising=False)
    app = create_app(tmp_path / "registry.sqlite")
    with TestClient(app) as tc:
        yield tc


def _auth_headers(key: str = "dev-research-key") -> dict[str, str]:
    return {"X-API-Key": key}


def _wait_for_terminal(client: TestClient, run_id: str, timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/runs/{run_id}", headers=_auth_headers()).json()["data"]
        if payload["state"] in {"success", "failed", "cancelled", "timed_out"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("run did not reach terminal state")


def test_health_and_doctor(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    doctor = client.get("/doctor")
    assert doctor.status_code == 200
    assert "checks" in doctor.json()["data"]


def test_run_submission_status_and_artifacts(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "reports" / "equities" / "run123"
    run_dir.mkdir(parents=True)
    report = run_dir / "run_report_run123.json"
    report.write_text('{"run_id":"run123","universe_size":1}')

    monkeypatch.setattr("spectraquant_v3.service.orchestrator.config_mod.get_equity_config", lambda *_: {"run": {}, "cache": {}, "qa": {}, "execution": {"mode": "paper"}, "portfolio": {}, "equities": {}}, raising=False)

    def fake_pipeline(cfg, run_mode, dry_run, run_id):
        return {
            "run_id": run_id,
            "status": "success",
            "universe": ["INFY.NS"],
            "signals": [type("S", (), {"canonical_symbol": "INFY.NS"})()],
            "decisions": [],
            "allocations": [type("A", (), {"blocked": False, "target_weight": 1.0})()],
            "artefact_paths": {
                "signals": str(run_dir / f"signals_summary_{run_id}.json"),
                "allocation": str(run_dir / f"allocation_summary_{run_id}.json"),
                "run_report": str(run_dir / f"run_report_{run_id}.json"),
                "policy_decisions": str(run_dir / f"policy_decisions_{run_id}.json"),
                "diagnostics_summary": str(run_dir / f"diagnostics_summary_{run_id}.json"),
            },
        }

    monkeypatch.setattr("spectraquant_v3.service.orchestrator.run_equity_pipeline", fake_pipeline, raising=False)

    payload = {
        "asset_class": "equity",
        "execution_mode": "research",
        "run_mode": "normal",
        "dry_run": False,
        "idempotency_key": "idem-run-submission-01",
        "run_id": "run123",
    }

    submit = client.post("/runs", json=payload, headers=_auth_headers())
    assert submit.status_code == 200
    assert submit.json()["ok"] is True
    assert submit.json()["data"]["run_id"] == "run123"
    assert submit.json()["data"]["state"] == "queued"

    status_payload = _wait_for_terminal(client, "run123")
    assert status_payload["state"] == "success"
    assert isinstance(status_payload["events"], list)

    artifacts = client.get("/runs/run123/artifacts", headers=_auth_headers())
    assert artifacts.status_code == 200
    assert artifacts.json()["ok"] is True


def test_idempotent_submission_reuses_existing_run(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("spectraquant_v3.service.orchestrator.config_mod.get_crypto_config", lambda *_: {"run": {}, "cache": {}, "qa": {}, "execution": {"mode": "paper"}, "portfolio": {}, "crypto": {}}, raising=False)

    def fake_pipeline(cfg, run_mode, dry_run, run_id):
        return {
            "run_id": run_id,
            "status": "success",
            "universe": ["BTC"],
            "signals": [type("S", (), {"canonical_symbol": "BTC"})()],
            "decisions": [],
            "allocations": [type("A", (), {"blocked": False, "target_weight": 1.0})()],
            "artefact_paths": {
                "signals": f"reports/crypto/{run_id}/signals_summary_{run_id}.json",
                "allocation": f"reports/crypto/{run_id}/allocation_summary_{run_id}.json",
                "run_report": f"reports/crypto/{run_id}/run_report_{run_id}.json",
                "policy_decisions": f"reports/crypto/{run_id}/policy_decisions_{run_id}.json",
                "diagnostics_summary": f"reports/crypto/{run_id}/diagnostics_summary_{run_id}.json",
            },
        }

    monkeypatch.setattr("spectraquant_v3.service.orchestrator.run_crypto_pipeline", fake_pipeline, raising=False)

    payload = {
        "asset_class": "crypto",
        "execution_mode": "research",
        "run_mode": "normal",
        "dry_run": False,
        "idempotency_key": "idem-crypto-001",
        "run_id": "runA",
    }

    first = client.post("/runs", json=payload, headers=_auth_headers()).json()
    second = client.post("/runs", json={**payload, "run_id": "runB"}, headers=_auth_headers()).json()
    assert first["data"]["run_id"] == second["data"]["run_id"]


def test_live_mode_guardrail_and_auth_scope(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQ_LIVE_APPROVAL_TOKEN", "approve-me")
    monkeypatch.setattr("spectraquant_v3.service.orchestrator.config_mod.get_equity_config", lambda *_: {"run": {}, "cache": {}, "qa": {}, "execution": {"mode": "live"}, "portfolio": {}, "equities": {}}, raising=False)

    payload = {
        "asset_class": "equity",
        "execution_mode": "live",
        "run_mode": "normal",
        "dry_run": False,
        "idempotency_key": "idem-live-unsafe-01",
    }

    resp = client.post("/runs", json=payload, headers=_auth_headers("dev-research-key")).json()
    assert resp["ok"] is False
    assert resp["error"]["code"] == "FORBIDDEN"


def test_cancel_run_transitions_to_cancelled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("spectraquant_v3.service.orchestrator.config_mod.get_crypto_config", lambda *_: {"run": {}, "cache": {}, "qa": {}, "execution": {"mode": "paper"}, "portfolio": {}, "crypto": {}}, raising=False)

    def slow_pipeline(cfg, run_mode, dry_run, run_id):
        time.sleep(0.25)
        return {
            "run_id": run_id,
            "status": "success",
            "universe": ["BTC"],
            "signals": [type("S", (), {"canonical_symbol": "BTC"})()],
            "decisions": [],
            "allocations": [type("A", (), {"blocked": False, "target_weight": 1.0})()],
            "artefact_paths": {
                "signals": f"reports/crypto/{run_id}/signals_summary_{run_id}.json",
                "allocation": f"reports/crypto/{run_id}/allocation_summary_{run_id}.json",
                "run_report": f"reports/crypto/{run_id}/run_report_{run_id}.json",
                "policy_decisions": f"reports/crypto/{run_id}/policy_decisions_{run_id}.json",
                "diagnostics_summary": f"reports/crypto/{run_id}/diagnostics_summary_{run_id}.json",
            },
        }

    monkeypatch.setattr("spectraquant_v3.service.orchestrator.run_crypto_pipeline", slow_pipeline, raising=False)

    payload = {
        "asset_class": "crypto",
        "execution_mode": "research",
        "run_mode": "normal",
        "dry_run": False,
        "idempotency_key": "idem-cancel-001",
        "run_id": "cancel01",
    }
    client.post("/runs", json=payload, headers=_auth_headers())
    client.post("/runs/cancel01/cancel", headers=_auth_headers())

    status_payload = _wait_for_terminal(client, "cancel01")
    assert status_payload["state"] == "cancelled"
