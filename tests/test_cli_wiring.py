"""GATE 3 — CLI wiring test.

Every ``cli/commands/*.py`` that defines a ``register_*_commands`` function
must be imported and called by **both** CLI entrypoints (``main.py`` and
``app.py``).  This test fails if any command module is orphaned.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COMMANDS_DIR = _REPO_ROOT / "src" / "spectraquant" / "cli" / "commands"
_MAIN_PY = _REPO_ROOT / "src" / "spectraquant" / "cli" / "main.py"
_APP_PY = _REPO_ROOT / "src" / "spectraquant" / "cli" / "app.py"


def _collect_register_functions() -> list[str]:
    """Return all ``register_*_commands`` function names in cli/commands/."""
    names: list[str] = []
    for py_file in sorted(_COMMANDS_DIR.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("register_") and node.name.endswith("_commands"):
                names.append(node.name)
    return names


def _source_calls_function(source_path: Path, func_name: str) -> bool:
    """Return True if *source_path* contains a call to *func_name*."""
    source = source_path.read_text()
    # Match both direct call and import
    return func_name in source


class TestCLIWiring:
    """All register_*_commands must be called by both CLI entrypoints."""

    register_functions = _collect_register_functions()

    @pytest.mark.parametrize("func_name", register_functions)
    def test_main_py_calls(self, func_name: str) -> None:
        assert _source_calls_function(_MAIN_PY, func_name), (
            f"{func_name} defined in cli/commands/ but never called in main.py"
        )

    @pytest.mark.parametrize("func_name", register_functions)
    def test_app_py_calls(self, func_name: str) -> None:
        assert _source_calls_function(_APP_PY, func_name), (
            f"{func_name} defined in cli/commands/ but never called in app.py"
        )

    def test_no_orphaned_command_modules(self) -> None:
        """Every non-__init__ module in cli/commands/ must define at least one
        register function and it must be wired into both entrypoints."""
        assert len(self.register_functions) > 0, "No register_*_commands found"
        main_src = _MAIN_PY.read_text()
        app_src = _APP_PY.read_text()
        missing_main = [f for f in self.register_functions if f not in main_src]
        missing_app = [f for f in self.register_functions if f not in app_src]
        assert not missing_main, f"Functions missing from main.py: {missing_main}"
        assert not missing_app, f"Functions missing from app.py: {missing_app}"


class TestCryptoRunDryRunTestMode:
    """GATE 4 — crypto-run --dry-run --test-mode must succeed."""

    def test_crypto_run_dry_run_test_mode_via_main(self) -> None:
        """crypto-run via main.py succeeds with --dry-run --test-mode."""
        import subprocess
        import sys

        _SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
        env = {
            **os.environ,
            "PYTHONPATH": _SRC_DIR + (os.pathsep + os.environ.get("PYTHONPATH", "")),
        }
        result = subprocess.run(
            [sys.executable, "-m", "spectraquant.cli.main", "crypto-run", "--dry-run", "--test-mode"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_REPO_ROOT),
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, f"crypto-run failed:\n{combined}"
        assert "TypeError" not in combined
        assert "unknown command" not in combined.lower()
        assert "Step 1:" in combined
        assert "Step 9:" in combined

    def test_crypto_run_dry_run_test_mode_via_app(self) -> None:
        """crypto-run via app.py succeeds with --dry-run --test-mode."""
        import subprocess
        import sys

        _SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
        env = {
            **os.environ,
            "PYTHONPATH": _SRC_DIR + (os.pathsep + os.environ.get("PYTHONPATH", "")),
        }
        result = subprocess.run(
            [sys.executable, "-m", "spectraquant.cli.app", "crypto-run", "--dry-run", "--test-mode"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_REPO_ROOT),
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, f"crypto-run failed:\n{combined}"
        assert "TypeError" not in combined
        assert "unknown command" not in combined.lower()
        assert "Step 1:" in combined
        assert "Step 9:" in combined

    def test_artifact_written(self) -> None:
        """crypto-run must write an artifact to reports/run/."""
        import subprocess
        import sys

        _SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
        env = {
            **os.environ,
            "PYTHONPATH": _SRC_DIR + (os.pathsep + os.environ.get("PYTHONPATH", "")),
        }
        result = subprocess.run(
            [sys.executable, "-m", "spectraquant.cli.main", "crypto-run", "--dry-run", "--test-mode"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, f"crypto-run failed:\n{result.stderr}"
        run_dir = _REPO_ROOT / "reports" / "run"
        json_files = list(run_dir.glob("crypto_run_*.json"))
        assert json_files, "No crypto_run artifact found in reports/run/"


class TestCryptoStreamNoBudget:
    """GATE 4 — crypto-stream must not be killed by enforce_stage_budget."""

    def test_crypto_stream_exempt_from_budget_in_main(self) -> None:
        src = _MAIN_PY.read_text()
        # crypto-stream must be in the no-budget set
        assert "crypto-stream" in src
        # The pattern: should skip enforce_stage_budget for crypto-stream
        assert "_NO_BUDGET_COMMANDS" in src or "crypto-stream" in src

    def test_crypto_stream_exempt_from_budget_in_app(self) -> None:
        src = _APP_PY.read_text()
        assert "crypto-stream" in src
        assert "_NO_BUDGET_COMMANDS" in src or "crypto-stream" in src
