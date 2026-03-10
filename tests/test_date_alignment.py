from __future__ import annotations

# replaced hardcoded ticker
from pathlib import Path

import pandas as pd

from spectraquant.core.time import resolve_prediction_date_for_horizon
from spectraquant.qa.output_check import write_date_alignment_report


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _load_prices() -> dict[str, pd.DataFrame]:
    price_data = {}
    for ticker in ["TICKER1.NS", "TICKER2.L"]:
        price_data[ticker] = pd.read_csv(FIXTURES / "prices" / f"{ticker}.csv")
    return price_data


def _load_intraday() -> dict[str, pd.DataFrame]:
    price_data = {}
    for ticker in ["TICKER1.NS", "TICKER2.L"]:
        price_data[ticker] = pd.read_csv(
            FIXTURES / "prices" / "intraday" / f"{ticker}_1m.csv"
        )
    return price_data


def test_date_alignment_report(tmp_path: Path) -> None:
    pred_df = pd.read_csv(FIXTURES / "expected" / "predictions.csv")
    price_data = _load_prices()
    intraday_data = _load_intraday()
    path = write_date_alignment_report(pred_df, price_data, intraday_data, output_dir=tmp_path)
    report = pd.read_csv(path)
    assert (report["status"] == "PASS").all()


def test_intraday_alignment_boundary() -> None:
    price_data = _load_prices()
    intraday_data = _load_intraday()
    bad = intraday_data["TICKER1.NS"].copy()
    bad.loc[bad.index[-1], "date"] = "2024-01-02T09:17:30Z"
    intraday_data["TICKER1.NS"] = bad
    aligned = resolve_prediction_date_for_horizon("TICKER1.NS", "5m", price_data, intraday_data)
    assert aligned.minute % 5 == 0
    assert aligned.second == 0
