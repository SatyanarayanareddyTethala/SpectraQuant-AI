"""Validation to ensure .gitignore rules are safe for production."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def check_gitignore_safety(ignore_synthetic_folders: bool = False) -> None:
    gitignore_path = Path(".gitignore")
    assert gitignore_path.exists(), ".gitignore file is missing"

    lines = [line.strip() for line in gitignore_path.read_text(encoding="utf-8").splitlines()]
    ignore_patterns = {line for line in lines if line and not line.startswith(("#", "!"))}

    def _contains(pattern: str) -> bool:
        return pattern in ignore_patterns

    assert not _contains("config.yaml"), "config.yaml must not be ignored"
    assert not _contains("src/"), "Source code directory must not be ignored"

    synthetic_ignored = _contains("synthetic") or _contains("data/")
    if not synthetic_ignored:
        message = "Synthetic data folders should be ignored to avoid accidental commits"
        if ignore_synthetic_folders:
            logger.warning("%s. Remediation: add folders to .gitignore / remove from repo.", message)
        else:
            raise AssertionError(message)

    assert _contains("reports/"), "reports/ should be ignored"
    assert _contains("models/"), "models/ should be ignored"
    assert _contains("data/processed") or _contains("data/"), "data/processed should be ignored"

    logger.info(".gitignore safety validated; required patterns present and critical files visible.")
