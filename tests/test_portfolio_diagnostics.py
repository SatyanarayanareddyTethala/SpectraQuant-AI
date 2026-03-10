import pandas as pd

from spectraquant.portfolio.simulator import simulate_portfolio


def test_portfolio_zero_weight_diagnostics():
    signals_df = pd.DataFrame(
        {
            "date": ["2023-01-02", "2023-01-02"],
            "ticker": ["AAA", "BBB"],
            "ensemble_score": [0.9, 0.8],
        }
    )

    dates = pd.date_range("2022-12-01", periods=30, freq="B", tz="UTC")
    price_data = {
        "AAA": pd.DataFrame(
            {"close": range(100, 130), "volume": [10] * 30},
            index=dates,
        ),
        "BBB": pd.DataFrame(
            {"close": range(200, 230), "volume": [10] * 30},
            index=dates,
        ),
    }
    config = {
        "portfolio": {
            "rebalance": "monthly",
            "weighting": "equal",
            "top_k": 2,
            "liquidity_min_volume": 1e6,
        }
    }

    result = simulate_portfolio(signals_df, price_data, config)
    diagnostics = result.get("diagnostics", [])
    assert diagnostics


def test_portfolio_zero_weight_reason_for_point_in_time_mode():
    signals_df = pd.DataFrame(
        {
            "date": ["2023-01-02"],
            "AAA": ["SELL"],
            "BBB": ["SELL"],
        }
    )

    dates = pd.date_range("2022-12-01", periods=30, freq="B", tz="UTC")
    price_data = {
        "AAA": pd.DataFrame({"close": range(100, 130)}, index=dates),
        "BBB": pd.DataFrame({"close": range(200, 230)}, index=dates),
    }
    config = {"portfolio": {"rebalance": "monthly"}, "signals_point_in_time": True}

    result = simulate_portfolio(signals_df, price_data, config)
    diagnostics = result.get("diagnostics", [])
    assert any(
        d.get("reason") == "all_zero_weights:signals_point_in_time_no_historical_coverage"
        for d in diagnostics
    )
