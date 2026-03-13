from __future__ import annotations

from pathlib import Path


README_PATH = Path(__file__).resolve().parents[2] / "README.md"


def test_readme_does_not_document_removed_v3_examples():
    text = README_PATH.read_text(encoding="utf-8")

    assert "sqv3 research run" not in text
    assert "sqv3 experiment run" not in text
    assert "sqv3 experiment results" not in text
    assert "sqv3 strategy run --strategy" not in text
    assert "sqv3 strategy portfolio --strategy" not in text


def test_readme_documents_current_v3_examples():
    text = README_PATH.read_text(encoding="utf-8")

    assert "sqv3 research dataset --asset-class crypto" in text
    assert "sqv3 strategy run crypto_momentum_v1" in text
    assert "sqv3 experiment compare <id_1> <id_2>" in text
    assert "sqv3 feature-store query --feature-name" in text
