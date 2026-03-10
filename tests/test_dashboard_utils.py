from __future__ import annotations

import pandas as pd
import pytest

from dashboard.utils.diagnostics import (
    DATE_INDEX_NOT_DATETIME,
    NO_SIGNAL_AFTER_FILTER,
    SIGNAL_RETURN_MISALIGNMENT,
    TIMEZONE_MISMATCH,
)
from dashboard.utils.simulator import simulate_portfolio_from_signals
from dashboard.utils.time_index import align_on_time_index, normalize_time_index


def test_normalize_time_index_rejects_integer_index() -> None:
    df = pd.DataFrame({"value": [1, 2]}, index=pd.RangeIndex(2))
    with pytest.raises(Exception) as exc:
        normalize_time_index(df)
    diagnostic = getattr(exc.value, "diagnostic", None)
    assert diagnostic is not None
    assert diagnostic.code == DATE_INDEX_NOT_DATETIME


def test_align_on_time_index_reports_timezone_mismatch() -> None:
    left = pd.DataFrame({"value": [1, 2]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    right = pd.DataFrame(
        {"value": [3, 4]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
    )
    _, _, diagnostics = align_on_time_index(left, right)
    assert any(diag.code == TIMEZONE_MISMATCH for diag in diagnostics)


def test_simulate_portfolio_reports_misalignment() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "signal": ["BUY"],
            "date": ["2020-01-01"],
            "horizon": ["1d"],
        }
    )
    returns = pd.DataFrame({"date": ["2021-01-01"], "return": [0.01]})
    outcome = simulate_portfolio_from_signals(signals, returns, None, horizon="1d", min_overlap=1)
    assert any(diag.code == SIGNAL_RETURN_MISALIGNMENT for diag in outcome.diagnostics)


def test_simulate_portfolio_reports_all_zero_weights() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "signal": ["BUY"],
            "date": ["2024-01-01"],
            "horizon": ["1d"],
            "score": [10.0],
        }
    )
    returns = pd.DataFrame({"date": ["2024-01-01"], "return": [0.01]})
    outcome = simulate_portfolio_from_signals(
        signals, returns, None, horizon="1d", alpha_threshold=50.0, min_overlap=1
    )
    assert any(diag.code == NO_SIGNAL_AFTER_FILTER for diag in outcome.diagnostics)
