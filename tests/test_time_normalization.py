from __future__ import annotations

import pandas as pd
import pytest

from spectraquant.core.time import normalize_time_index


def test_normalize_time_index_from_date_column() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01"],
            "value": [1, 2],
        }
    )
    normalized = normalize_time_index(df, context="unit-test")
    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert str(normalized.index.tz) == "UTC"
    assert normalized.index.is_monotonic_increasing


def test_normalize_time_index_rejects_numeric_index() -> None:
    df = pd.DataFrame({"value": [1, 2]}, index=[0, 1])
    with pytest.raises(ValueError, match="unit-test"):
        normalize_time_index(df, context="unit-test")
