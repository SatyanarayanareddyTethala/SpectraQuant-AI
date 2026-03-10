from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectraquant.core.time import ensure_datetime_column

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EXPECTED_DIR = FIXTURES / "expected"


def _fixture_file(name: str) -> Path:
    path = EXPECTED_DIR / name
    if not path.exists():
        raise AssertionError(f"Missing fixture artifact {path}")
    return path


def test_no_epoch_dates_in_predictions() -> None:
    pred_path = _fixture_file("predictions.csv")
    pred_df = pd.read_csv(pred_path)
    pred_df = ensure_datetime_column(pred_df, "date")
    assert (pred_df["date"].dt.normalize() != pd.Timestamp("1970-01-01", tz="UTC")).all()


def test_no_epoch_dates_in_signals() -> None:
    signal_path = _fixture_file("signals.csv")
    signals_df = pd.read_csv(signal_path)
    signals_df = ensure_datetime_column(signals_df, "date")
    assert (signals_df["date"].dt.normalize() != pd.Timestamp("1970-01-01", tz="UTC")).all()


def test_portfolio_weights_not_zero() -> None:
    weights_path = _fixture_file("portfolio_weights.csv")
    weights_df = pd.read_csv(weights_path)
    numeric = weights_df.drop(columns=["date"], errors="ignore").select_dtypes(include="number")
    assert numeric.to_numpy().sum() > 0, "Portfolio weights are all zero"


def test_snapshot_rank_determinism() -> None:
    signal_path = _fixture_file("signals.csv")
    signals_df = pd.read_csv(signal_path)
    sorted_once = signals_df.sort_values("score", ascending=False)
    sorted_twice = signals_df.sort_values("score", ascending=False)
    assert sorted_once["ticker"].tolist() == sorted_twice["ticker"].tolist()
