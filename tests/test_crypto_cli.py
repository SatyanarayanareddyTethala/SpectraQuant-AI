"""Tests for crypto CLI integration."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
_CLI_ENV = {
    **os.environ,
    "PYTHONPATH": _SRC_DIR + (os.pathsep + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""),
}


def test_crypto_commands_in_app_help() -> None:
    """Crypto commands appear in app.py --help output."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    for cmd in ("crypto-run", "crypto-stream", "onchain-scan", "agents-run", "allocate"):
        assert cmd in output, f"'{cmd}' missing from help output"


def test_crypto_commands_in_main_help() -> None:
    """Crypto commands appear in main.py --help / USAGE output."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.main", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    for cmd in ("crypto-run", "crypto-stream", "onchain-scan", "agents-run", "allocate"):
        assert cmd in output, f"'{cmd}' missing from main.py help"


def test_crypto_run_no_typeerror_via_app() -> None:
    """crypto-run via app.py must not raise TypeError for missing cfg.

    With ``--test-mode``, crypto is auto-enabled so the pipeline succeeds.
    """
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "crypto-run", "--dry-run", "--test-mode"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    combined = result.stdout + result.stderr
    # Must NOT see TypeError about missing positional argument
    assert "TypeError" not in combined
    # With --test-mode, crypto is auto-enabled; pipeline should succeed
    assert result.returncode == 0, f"Unexpected failure:\n{combined}"


def test_crypto_run_no_typeerror_via_main() -> None:
    """crypto-run via main.py must not raise TypeError for missing cfg.

    With ``--test-mode``, crypto is auto-enabled so the pipeline succeeds.
    """
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.main", "crypto-run", "--dry-run", "--test-mode"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    combined = result.stdout + result.stderr
    assert "TypeError" not in combined
    # With --test-mode, crypto is auto-enabled; pipeline should succeed
    assert result.returncode == 0, f"Unexpected failure:\n{combined}"


def test_crypto_run_not_unknown_command() -> None:
    """crypto-run must not be treated as unknown command."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "crypto-run"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    combined = result.stdout + result.stderr
    # Should NOT show "Usage:" only (which indicates unknown command)
    # Instead should recognize the command and try to run it
    assert "Running command: crypto-run" in combined


def test_register_crypto_commands_zero_arg() -> None:
    """All registered crypto handlers must be zero-arg callables."""
    from spectraquant.cli.commands.crypto import register_crypto_commands

    commands: dict = {}
    register_crypto_commands(commands)

    expected = {"crypto-run", "crypto-stream", "onchain-scan", "agents-run", "allocate", "crypto-ingest-dataset"}
    assert expected == set(commands.keys())
    for name, handler in commands.items():
        assert callable(handler), f"{name} is not callable"


def test_doctor_still_works_after_crypto_changes() -> None:
    """doctor command must still work after crypto integration."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.main", "doctor"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "SpectraQuant Environment Doctor" in combined
