import pandas as pd

from spectraquant.eval.walkforward import _generate_splits


def test_walkforward_split_order():
    dates = pd.date_range("2023-01-01", periods=60, freq="B", tz="UTC")
    splits = _generate_splits(pd.Series(dates))
    assert splits
    for train_dates, test_dates in splits:
        assert max(train_dates) < min(test_dates)
        assert set(train_dates).isdisjoint(set(test_dates))
