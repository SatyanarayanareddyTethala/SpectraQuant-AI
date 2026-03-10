"""Tests for equity CLI commands."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
_CLI_ENV = {
    **os.environ,
    "PYTHONPATH": _SRC_DIR + (
        os.pathsep + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""
    ),
}


def test_equity_commands_in_app_help() -> None:
    """Equity commands appear in app.py --help output."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.app", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    for cmd in ("equity-run", "equity-download", "equity-universe", "equity-signals"):
        assert cmd in output, f"'{cmd}' missing from app.py --help output"


def test_equity_commands_in_main_help() -> None:
    """Equity commands appear in main.py USAGE string."""
    result = subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.main", "--help"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    for cmd in ("equity-run", "equity-download", "equity-universe", "equity-signals"):
        assert cmd in output, f"'{cmd}' missing from main.py USAGE"


def test_equity_commands_are_separate_from_crypto() -> None:
    """Crypto and equity commands must be distinct (no shared entry point)."""
    from spectraquant.cli.commands.equities import register_equity_commands
    from spectraquant.cli.commands.crypto import register_crypto_commands

    crypto_cmds: dict = {}
    equity_cmds: dict = {}
    register_crypto_commands(crypto_cmds)
    register_equity_commands(equity_cmds)

    overlap = set(crypto_cmds) & set(equity_cmds)
    assert overlap == set(), f"Commands overlap between crypto and equity: {overlap}"


def test_equity_namespace_does_not_import_crypto() -> None:
    """Nothing in the equities namespace may import from spectraquant.crypto."""
    import ast
    import importlib.util

    equities_root = (
        Path(__file__).resolve().parents[1] / "src" / "spectraquant" / "equities"
    )
    violations: list[str] = []
    for py_file in equities_root.rglob("*.py"):
        source = py_file.read_text()
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("spectraquant.crypto"):
                        violations.append(f"{py_file}: imports {node.module}")

    assert violations == [], (
        "Equity namespace must not import from crypto namespace:\n"
        + "\n".join(violations)
    )


def test_crypto_namespace_does_not_import_equities() -> None:
    """Nothing in the crypto namespace may import from spectraquant.equities."""
    import ast

    crypto_root = (
        Path(__file__).resolve().parents[1] / "src" / "spectraquant" / "crypto"
    )
    violations: list[str] = []
    for py_file in crypto_root.rglob("*.py"):
        source = py_file.read_text()
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("spectraquant.equities"):
                    violations.append(f"{py_file}: imports {node.module}")

    assert violations == [], (
        "Crypto namespace must not import from equity namespace:\n"
        + "\n".join(violations)
    )
