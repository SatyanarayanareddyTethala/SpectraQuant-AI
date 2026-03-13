from __future__ import annotations

import pandas as pd

from spectraquant.dataset.panel import build_price_feature_panel


def test_build_price_feature_panel_returns_cross_sectional_rows() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    price_data = {
        "AAA.NS": pd.DataFrame({"close": [10, 10.2, 10.4, 10.1, 10.5, 10.7, 10.9, 11.0]}, index=dates),
        "BBB.NS": pd.DataFrame({"close": [20, 20.1, 20.2, 20.1, 20.3, 20.2, 20.4, 20.6]}, index=dates),
    }

    panel = build_price_feature_panel(price_data)

    assert not panel.empty
    assert set(["date", "ticker", "Close", "ret_1d", "ret_5d", "sma_5", "vol_5", "label"]).issubset(panel.columns)
    assert set(panel["ticker"].unique()) == {"AAA.NS", "BBB.NS"}
