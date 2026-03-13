from __future__ import annotations

import pandas as pd

from spectraquant.dataset.panel import build_price_feature_panel


def test_build_price_feature_panel_returns_cross_sectional_rows() -> None:
    dates = pd.date_range("2024-01-01", periods=25, freq="D", tz="UTC")
    price_data = {
        "AAA.NS": pd.DataFrame({"close": [10 + (i * 0.1) for i in range(25)]}, index=dates),
        "BBB.NS": pd.DataFrame({"close": [20 + ((-1) ** i) * 0.2 + i * 0.05 for i in range(25)]}, index=dates),
    }

    panel = build_price_feature_panel(price_data)

    assert not panel.empty
    assert set(["date", "ticker", "Close", "ret_1d", "ret_5d", "sma_5", "vol_5", "rsi_14", "label"]).issubset(panel.columns)
    assert set(panel["ticker"].unique()) == {"AAA.NS", "BBB.NS"}


def test_panel_builder_preserves_nan_labels_for_tail_rows() -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
    price_data = {"AAA.NS": pd.DataFrame({"close": [10 + i for i in range(12)]}, index=dates)}

    panel = build_price_feature_panel(price_data)
    tail = panel[panel["ticker"] == "AAA.NS"].tail(5)

    assert tail["label"].isna().all()
