"""Filesystem validation for expected SpectraQuant artifacts."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _assert_dir_exists(path: Path, non_empty: bool = False) -> None:
    assert path.exists() and path.is_dir(), f"Required directory missing: {path}"
    if non_empty:
        contents = list(path.iterdir())
        assert contents, f"Directory {path} is empty but expected files"
    logger.info("✓ Folder present: %s", path)


def check_expected_outputs() -> None:
    """Verify core output locations are present and populated."""

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    pred_dir = Path("reports/predictions")
    signals_dir = Path("reports/signals")
    portfolio_dir = Path("reports/portfolio")

    _assert_dir_exists(raw_dir, non_empty=True)
    _assert_dir_exists(processed_dir)
    _assert_dir_exists(models_dir)
    assert any(models_dir.iterdir()), f"No model files found in {models_dir}"
    logger.info("✓ Model files present under %s", models_dir)
    _assert_dir_exists(pred_dir)
    _assert_dir_exists(signals_dir)
    _assert_dir_exists(portfolio_dir)
