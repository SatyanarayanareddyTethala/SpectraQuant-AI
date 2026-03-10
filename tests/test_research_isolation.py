from __future__ import annotations

from pathlib import Path

from spectraquant.qa.research_isolation import check_no_research_imports


def test_production_does_not_import_research() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "spectraquant"
    check_no_research_imports(root)
