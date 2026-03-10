import pandas as pd
from spectraquant.alpha.experts import EXPERT_REGISTRY

REQ = {"ticker","date","horizon","score","signal","confidence","expected_return","risk_estimate","expert_name","schema_version"}


def test_expert_output_schema():
    prices = {"T.NS": pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60, tz="UTC"), "close": range(60), "volume": [1000]*60})}
    cfg = {"predictions": {"daily_horizons": ["1d", "5d"]}}
    for name, fn in EXPERT_REGISTRY.items():
        df = fn(prices, {"T.NS": {}}, pd.DataFrame(), cfg)
        assert REQ.issubset(df.columns), name
        assert df["expert_name"].eq(name).all()
