import pandas as pd

from spectraquant.features.ohlcv_features import compute_ohlcv_features


def test_compute_ohlcv_features_output():
    dates = pd.date_range("2023-01-01", periods=80, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "open": 100 + pd.Series(range(80)).values,
            "high": 101 + pd.Series(range(80)).values,
            "low": 99 + pd.Series(range(80)).values,
            "close": 100 + pd.Series(range(80)).values,
            "volume": 1000 + pd.Series(range(80)).values,
        },
        index=dates,
    )

    features = compute_ohlcv_features(df)
    expected_cols = {
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "logret_1d",
        "roc_10",
        "momentum_20d",
        "sma_10",
        "sma_20",
        "sma_50",
        "close_sma20",
        "sma10_sma50",
        "vol_20",
        "atr_14",
        "hl_range",
        "z_20",
        "bollinger_width",
        "body",
        "upper_wick",
        "lower_wick",
        "gap",
        "vol_z_20",
        "volume_change_1d",
    }
    assert expected_cols.issubset(features.columns)
    assert features.notna().all().all()
