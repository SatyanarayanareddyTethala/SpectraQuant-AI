import pandas as pd

from spectraquant.regime.simple_regime import compute_regime, REGIME_LABELS


def test_regime_labels_stable():
    dates = pd.date_range("2023-01-01", periods=100, freq="B", tz="UTC")
    close = pd.Series(range(100, 200), index=dates)
    df = pd.DataFrame({"close": close}, index=dates)
    regime = compute_regime(df)
    assert regime.notna().all()
    assert set(regime.unique()).issubset(set(REGIME_LABELS))
