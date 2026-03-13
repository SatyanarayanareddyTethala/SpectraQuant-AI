from __future__ import annotations

import pandas as pd

from spectraquant_v3.feature_store import FeatureStore


def test_feature_store_write_and_list(tmp_path) -> None:
    store = FeatureStore(tmp_path / "feature_store")
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({"ret_1d": [0.1, 0.2, 0.3]}, index=dates)

    meta = store.write_feature_frame(
        df=df,
        feature_name="returns",
        feature_version="v1",
        symbol="INFY.NS",
        asset_class="equity",
        source_run_id="test",
    )

    assert meta.row_count == 3
    listed = store.list_feature_sets(asset_class="equity", feature_name="returns")
    assert len(listed) == 1
    assert listed[0].symbol == "INFY.NS"
