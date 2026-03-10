from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectraquant.core.schema import validate_predictions, validate_signals
from spectraquant.core.time import ensure_datetime_column, is_intraday_horizon
from spectraquant.data.normalize import normalize_price_columns


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EXPECTED_DIR = FIXTURES / "expected"
PRICES_DIR = FIXTURES / "prices"
INTRADAY_DIR = PRICES_DIR / "intraday"


def _fixture_file(name: str) -> Path:
    path = EXPECTED_DIR / name
    if not path.exists():
        raise AssertionError(f"Missing fixture artifact {path}")
    return path


def test_predictions_invariants() -> None:
    pred_path = _fixture_file("predictions.csv")
    pred_df = pd.read_csv(pred_path)
    pred_df = validate_predictions(pred_df)
    assert pred_df["model_version"].notna().all()
    assert pred_df["factor_set_version"].notna().all()

    for (ticker, horizon), pred_rows in pred_df.groupby(["ticker", "horizon"], dropna=False):
        if is_intraday_horizon(horizon):
            price_path = INTRADAY_DIR / f"{ticker}_1m.csv"
        else:
            price_path = PRICES_DIR / f"{ticker}.parquet"
            if not price_path.exists():
                price_path = PRICES_DIR / f"{ticker}.csv"
        assert price_path.exists(), f"Missing price history for {ticker} ({horizon})"
        if price_path.suffix == ".parquet":
            price_df = pd.read_parquet(price_path)
        else:
            price_df = pd.read_csv(price_path)
        price_df = normalize_price_columns(price_df, ticker)
        price_df = ensure_datetime_column(price_df, "date")
        latest_date = price_df["date"].max()
        pred_date = pred_rows["date"].max()
        assert pred_date == latest_date


def test_signals_invariants() -> None:
    signal_path = _fixture_file("signals.csv")
    signals_df = pd.read_csv(signal_path)
    signals_df = validate_signals(signals_df)
    assert signals_df["date"].dt.tz is not None


def test_portfolio_invariants() -> None:
    returns_path = _fixture_file("portfolio_returns.csv")
    returns_df = pd.read_csv(returns_path)
    returns_df = ensure_datetime_column(returns_df, "date")
    assert returns_df["date"].dt.tz is not None
