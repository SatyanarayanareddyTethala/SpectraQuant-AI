#!/usr/bin/env python3
"""Release validation checks for SpectraQuant."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
CHANGELOG_PATH = ROOT_DIR / "CHANGELOG.md"
PACKAGE_INIT = ROOT_DIR / "src" / "spectraquant" / "__init__.py"
MODELS_LATEST = ROOT_DIR / "models" / "latest" / "model.txt"


@dataclass
class ValidationResult:
    ok: bool
    message: str


def _load_version_from_pyproject() -> str:
    in_project = False
    for line in PYPROJECT_PATH.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project and stripped.startswith("version"):
            _, _, value = stripped.partition("=")
            return value.strip().strip('"').strip("'")
    return ""


def _load_version_from_init() -> str:
    for line in PACKAGE_INIT.read_text().splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def check_version_consistency(expected_version: str | None = None) -> ValidationResult:
    pyproject_version = _load_version_from_pyproject()
    init_version = _load_version_from_init()
    if not pyproject_version or not init_version:
        return ValidationResult(False, "Version missing in pyproject.toml or src/spectraquant/__init__.py.")
    if pyproject_version != init_version:
        return ValidationResult(
            False, f"Version mismatch: pyproject.toml={pyproject_version} __init__.py={init_version}."
        )
    if expected_version and pyproject_version != expected_version:
        return ValidationResult(
            False, f"Version mismatch: expected {expected_version}, found {pyproject_version}."
        )
    return ValidationResult(True, f"Version consistency OK ({pyproject_version}).")


def check_changelog(version: str | None = None) -> ValidationResult:
    if not CHANGELOG_PATH.exists():
        return ValidationResult(False, "CHANGELOG.md is missing.")
    content = CHANGELOG_PATH.read_text()
    if version and f"[{version}]" not in content:
        return ValidationResult(False, f"CHANGELOG.md missing entry for [{version}].")
    if "## " not in content:
        return ValidationResult(False, "CHANGELOG.md appears empty or missing headings.")
    return ValidationResult(True, "Changelog OK.")


def check_models_promotable() -> ValidationResult:
    if not MODELS_LATEST.exists():
        return ValidationResult(False, f"Model artifact missing: {MODELS_LATEST}.")
    if MODELS_LATEST.stat().st_size == 0:
        return ValidationResult(False, f"Model artifact is empty: {MODELS_LATEST}.")
    return ValidationResult(True, f"Latest model artifact OK ({MODELS_LATEST}).")


def run_checks(expected_version: str | None) -> list[ValidationResult]:
    results = [
        check_version_consistency(expected_version),
        check_changelog(expected_version),
        check_models_promotable(),
    ]
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run release validation checks.")
    parser.add_argument("--expected-version", dest="expected_version", help="Expected release version.")
    args = parser.parse_args()

    results = run_checks(args.expected_version)
    failures = [res for res in results if not res.ok]
    for res in results:
        status = "OK" if res.ok else "FAIL"
        print(f"[{status}] {res.message}")
    if failures:
        print(f"{len(failures)} validation checks failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
