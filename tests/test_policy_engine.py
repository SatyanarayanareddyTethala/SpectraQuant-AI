from __future__ import annotations

# replaced hardcoded ticker
import pandas as pd
import pytest

from spectraquant.core.policy import PolicyViolation, enforce_policy, enforce_turnover_policy


def test_policy_max_positions() -> None:
    config = {"portfolio": {"policies": {"max_positions": 1}}}
    with pytest.raises(PolicyViolation, match="max_positions"):
        enforce_policy(["TICKER1.NS", "TICKER2.L"], config)


def test_policy_max_turnover() -> None:
    config = {"portfolio": {"policies": {"max_turnover": 0.1}}}
    prior = pd.Series({"TICKER1.NS": 0.0, "TICKER2.L": 0.0})
    current = pd.Series({"TICKER1.NS": 0.2, "TICKER2.L": 0.0})
    with pytest.raises(PolicyViolation, match="max_turnover"):
        enforce_turnover_policy(current, prior, config)


def test_policy_auto_repair() -> None:
    config = {"portfolio": {"policies": {"max_positions": 1}, "policy": {"auto_repair": True}}}
    tickers, repairs = enforce_policy(["TICKER2.L", "TICKER1.NS"], config)
    assert tickers == ["TICKER1.NS"]
    assert repairs
