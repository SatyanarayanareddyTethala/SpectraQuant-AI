from __future__ import annotations

import pandas as pd

from spectraquant.cli.main import _generate_signals_from_predictions


def test_intraday_signal_thresholds_and_score_scaling() -> None:
    pred_df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "date": ["2026-01-01", "2026-01-01", "2026-01-01"],
            "horizon": ["5m", "5m", "5m"],
            "regime": ["neutral", "neutral", "neutral"],
            "score": [65.0, 30.0, 55.0],
        }
    )
    config = {
        "intraday": {
            "signal_thresholds": {"buy": 0.6, "sell": 0.4},
            "top_n": 5,
        }
    }

    signals = _generate_signals_from_predictions(pred_df, config)
    signal_map = signals.set_index("ticker")["signal"].to_dict()

    assert signal_map["AAA"] == "BUY"
    assert signal_map["BBB"] == "SELL"
    assert signal_map["CCC"] == "HOLD"


def test_intraday_top_n_is_applied_per_horizon() -> None:
    pred_df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "date": ["2026-01-01"] * 6,
            "horizon": ["5m", "5m", "5m", "30m", "30m", "30m"],
            "regime": ["neutral"] * 6,
            "score": [0.90, 0.80, 0.70, 0.95, 0.60, 0.50],
        }
    )
    config = {
        "intraday": {
            "signal_thresholds": {"buy": 0.6, "sell": 0.4},
            "top_n": 2,
        }
    }

    signals = _generate_signals_from_predictions(pred_df, config)

    assert len(signals) == 4
    counts = signals["horizon"].value_counts().to_dict()
    assert counts["5m"] == 2
    assert counts["30m"] == 2
