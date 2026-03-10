"""Smoke tests for CLI commands."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure subprocesses can find the spectraquant package via PYTHONPATH
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
_CLI_ENV = {**os.environ, "PYTHONPATH": _SRC_DIR + (os.pathsep + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")}


def test_cli_help_flag() -> None:
    """Test that --help flag works."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert result.returncode == 0
    assert "Usage:" in result.stdout or "Usage:" in result.stderr


def test_cli_h_flag() -> None:
    """Test that -h flag works."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "-h"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert result.returncode == 0
    assert "Usage:" in result.stdout or "Usage:" in result.stderr


def test_cli_no_args_shows_usage() -> None:
    """Test that running without args shows usage."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert "Usage:" in result.stdout or "Usage:" in result.stderr


def test_cli_unknown_command_shows_usage() -> None:
    """Test that unknown command shows usage."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "unknown-command"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert "Usage:" in result.stdout or "Usage:" in result.stderr


def test_cli_command_discovery() -> None:
    """Test that all expected commands are discoverable."""
    expected_commands = [
        "download",
        "news-scan",
        "features",
        "build-dataset",
        "train",
        "predict",
        "signals",
        "score",
        "portfolio",
        "execute",
        "eval",
        "retrain",
        "refresh",
        "promote-model",
        "list-models",
        "feature-pruning",
        "model-compare",
        "stress-test",
        "regime-stress",
        "explain-portfolio",
        "compare-runs",
        "doctor",
        "health-check",
        "release-check",
    ]
    
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    output = result.stdout + result.stderr
    
    for command in expected_commands:
        assert command in output, f"Command '{command}' not found in help output"


def test_cli_universe_subcommand_help() -> None:
    """Test that universe subcommand requires subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "universe"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    output = result.stdout + result.stderr
    assert "universe" in output.lower()


def test_cli_main_module_entry_point() -> None:
    """Test that old main module entry point still works."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.main", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert result.returncode == 0
    assert "Usage:" in result.stdout or "Usage:" in result.stderr


def test_cli_app_module_entry_point() -> None:
    """Test that new app module entry point works."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert result.returncode == 0
    assert "Usage:" in result.stdout or "Usage:" in result.stderr


def test_cli_research_mode_flag() -> None:
    """Test that --research flag is recognized."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "--research", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert result.returncode == 0


def test_cli_use_sentiment_flag() -> None:
    """Test that --use-sentiment flag is recognized."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "--use-sentiment", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert result.returncode == 0


def test_cli_test_mode_flag() -> None:
    """Test that --test-mode flag is recognized."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "--test-mode", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert result.returncode == 0


def test_cli_dry_run_flag() -> None:
    """Test that --dry-run flag is recognized."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "--dry-run", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert result.returncode == 0


def test_cli_universe_flag() -> None:
    """Test that --universe flag is recognized."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "--universe=nifty50", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    
    assert result.returncode == 0
