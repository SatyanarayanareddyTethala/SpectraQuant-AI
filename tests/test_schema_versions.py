from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectraquant.core.schema import SCHEMA_COLUMNS, schema_version_for


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _assert_columns(name: str, path: Path) -> None:
    df = pd.read_csv(path)
    expected = SCHEMA_COLUMNS[name]
    if name == "portfolio_weights":
        assert set(expected).issubset(df.columns)
    else:
        assert list(df.columns) == expected
    assert (df["schema_version"] == schema_version_for(name)).all()


def test_schema_versions_predictions() -> None:
    _assert_columns("predictions", FIXTURES / "expected" / "predictions.csv")


def test_schema_versions_signals() -> None:
    _assert_columns("signals", FIXTURES / "expected" / "signals.csv")


def test_schema_versions_portfolio_returns() -> None:
    _assert_columns("portfolio_returns", FIXTURES / "expected" / "portfolio_returns.csv")


def test_schema_versions_portfolio_weights() -> None:
    _assert_columns("portfolio_weights", FIXTURES / "expected" / "portfolio_weights.csv")
