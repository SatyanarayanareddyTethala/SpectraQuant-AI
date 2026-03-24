from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from spectraquant_v3.cli.main import app


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_V3_CONFIG_DIR = REPO_ROOT / "local-config" / "v3"


def test_local_v3_config_contains_required_files() -> None:
    for fname in ("base.yaml", "crypto.yaml", "equities.yaml"):
        assert (LOCAL_V3_CONFIG_DIR / fname).is_file()


def test_doctor_accepts_local_v3_config_dir() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--config-dir", str(LOCAL_V3_CONFIG_DIR)])

    assert result.exit_code == 0
    assert "All checks passed." in result.stdout
