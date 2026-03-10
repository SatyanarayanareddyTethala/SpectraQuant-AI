import pandas as pd

from spectraquant.dataset.labels import compute_forward_returns


def test_forward_returns_leakage_protection():
    close = pd.Series([100, 101, 102, 103, 104, 105], index=pd.date_range("2023-01-01", periods=6))
    fwd = compute_forward_returns(close, horizon=2)
    assert fwd.iloc[0] == (close.iloc[2] / close.iloc[0] - 1)
    assert fwd.iloc[-1] != fwd.iloc[-1]
    assert fwd.iloc[-2] != fwd.iloc[-2]
