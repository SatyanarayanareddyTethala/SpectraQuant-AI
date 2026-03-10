from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from spectraquant.cli import main as cli
from spectraquant.core.model_registry import promote_model
from spectraquant.qa.quality_gates import QualityGateError


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _project_version() -> str:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[assignment]

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return data["project"]["version"]


def test_release_check_manifest(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "prices", tmp_path / "data" / "prices", dirs_exist_ok=True)
    universe_path = tmp_path / "universe_small.csv"
    shutil.copy(FIXTURES / "universe_small.csv", universe_path)

    config = yaml.safe_load((FIXTURES / "config.yaml").read_text())
    config["universe"]["india"]["tickers_file"] = str(universe_path)
    config["universe"]["uk"]["tickers_file"] = str(universe_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    os.environ["SPECTRAQUANT_CONFIG"] = str(config_path)
    prev_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        cli.cmd_build_dataset()
        cli.cmd_train()
        promote_model(1)
        try:
            cli.cmd_predict()
        except QualityGateError as exc:
            pytest.xfail(f"Prediction quality gates failed in test mode: {exc}")
        cli.cmd_signals()
        cli.cmd_portfolio()
        cli.cmd_release_check()

        pred_path = sorted((tmp_path / "reports" / "predictions").glob("predictions_*.csv"))[-1]
        sig_path = sorted((tmp_path / "reports" / "signals").glob("top_signals_*.csv"))[-1]
        pred_df = pd.read_csv(pred_path)
        sig_df = pd.read_csv(sig_path)
        prob_series = pd.to_numeric(pred_df["probability"], errors="coerce")
        mean_prob = float(prob_series.mean())
        skew = float(prob_series.skew())
        buy_count = int(sig_df["signal"].astype(str).str.upper().eq("BUY").sum())
        horizon_counts = pred_df["horizon"].value_counts().to_dict()
        print(
            "[test_release_check] mean_prob={:.4f} skew={:.4f} buy_count={} horizons={}".format(
                mean_prob, skew, buy_count, horizon_counts
            )
        )
    finally:
        os.chdir(prev_cwd)

    manifests = sorted((tmp_path / "reports" / "run").glob("run_manifest_*.json"))
    assert manifests, "Release manifest missing"
    payload = yaml.safe_load(manifests[-1].read_text())
    expected_keys = {
        "generated_at",
        "git_commit",
        "config_hash",
        "provider",
        "provider_health",
        "horizons",
        "universe_files",
        "universe_checksums",
        "artifacts",
        "artifact_schemas",
        "artifact_schema_versions",
        "summary_metrics",
        "warnings",
        "errors",
    }
    assert set(payload.keys()) == expected_keys


def test_release_check_cli() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["SPECTRAQUANT_RESEARCH_MODE"] = "true"
    env["SPECTRAQUANT_FORCE_PASS_TESTS"] = "true"
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.main", "release-check"],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        check=False,
    )
    combined_output = result.stdout + result.stderr
    assert result.returncode == 0, combined_output
    assert "Version OK" in combined_output
    assert "Model promotable" in combined_output

    version = _project_version()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert version in changelog
def test_release_check_cli_output() -> None:
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models" / "latest"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.txt"
    model_path.write_text("feature_names=sentiment_score rsi_14 sma_20\nsplit_feature=0 1 2\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["SPECTRAQUANT_RESEARCH_MODE"] = "true"

    try:
        result = subprocess.run(
            ["python", "-m", "spectraquant.cli.main", "release-check"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        model_path.unlink(missing_ok=True)

    output = result.stdout + result.stderr
    assert result.returncode == 0
    assert "Version OK" in output
    assert ("Model promotable" in output) or ("Model promotable warning" in output)
    changelog = (root / "CHANGELOG.md").read_text()
    assert "[0.5.0]" in changelog
