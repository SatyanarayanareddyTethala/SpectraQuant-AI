from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from spectraquant_v3.service.api import create_app


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = create_app(tmp_path / "registry.sqlite")
    return TestClient(app)


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
            "signals": [],
            "decisions": [],
            "allocations": [],
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

    submit = client.post("/runs", json=payload)
    assert submit.status_code == 200
    assert submit.json()["ok"] is True
    assert submit.json()["data"]["run_id"] == "run123"

    status = client.get("/runs/run123")
    assert status.status_code == 200
    assert status.json()["data"]["state"] == "success"
    assert isinstance(status.json()["data"]["events"], list)

    artifacts = client.get("/runs/run123/artifacts")
    assert artifacts.status_code == 200
    assert artifacts.json()["ok"] is True


def test_idempotent_submission_reuses_existing_run(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("spectraquant_v3.service.orchestrator.config_mod.get_crypto_config", lambda *_: {"run": {}, "cache": {}, "qa": {}, "execution": {"mode": "paper"}, "portfolio": {}, "crypto": {}}, raising=False)

    def fake_pipeline(cfg, run_mode, dry_run, run_id):
        return {
            "run_id": run_id,
            "status": "success",
            "universe": ["BTC"],
            "signals": [],
            "decisions": [],
            "allocations": [],
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

    first = client.post("/runs", json=payload).json()
    second = client.post("/runs", json={**payload, "run_id": "runB"}).json()
    assert first["data"]["run_id"] == second["data"]["run_id"]


def test_live_mode_guardrail_blocks_unsafe_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("spectraquant_v3.service.orchestrator.config_mod.get_equity_config", lambda *_: {"run": {}, "cache": {}, "qa": {}, "execution": {"mode": "paper"}, "portfolio": {}, "equities": {}}, raising=False)

    payload = {
        "asset_class": "equity",
        "execution_mode": "live",
        "run_mode": "test",
        "dry_run": True,
        "idempotency_key": "idem-live-unsafe-01",
    }

    resp = client.post("/runs", json=payload).json()
    assert resp["ok"] is False
    assert resp["error"]["code"] == "CONFIG_ERROR"
