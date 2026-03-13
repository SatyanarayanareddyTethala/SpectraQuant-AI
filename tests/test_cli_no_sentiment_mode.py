from __future__ import annotations

import logging

import pandas as pd

from spectraquant.cli import main
from spectraquant.qa import quality_gates


def test_parse_cli_overrides_supports_no_sentiment() -> None:
    args, use_sentiment, *_rest, verbose, no_sentiment = main._parse_cli_overrides(
        ["build-dataset", "--no-sentiment", "--verbose"]
    )
    assert args == ["build-dataset"]
    assert use_sentiment is False
    assert verbose is True
    assert no_sentiment is True


def test_dataset_builder_does_not_load_sentiment_functions_when_disabled(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    price_data = {
        "AAA.NS": pd.DataFrame({"close": [10 + i * 0.2 for i in range(20)]}, index=dates),
        "BBB.NS": pd.DataFrame({"close": [20 + i * 0.1 for i in range(20)]}, index=dates),
    }

    monkeypatch.setattr(main, "_resolve_tickers_with_meta", lambda cfg: (("AAA.NS", "BBB.NS"), {"raw_count": 2}))
    monkeypatch.setattr(main, "_collect_price_data", lambda tickers: price_data)
    monkeypatch.setattr(main, "run_quality_gates_dataset", lambda df, cfg: None)
    monkeypatch.setattr(main, "register_default_factors", lambda: None)
    monkeypatch.setattr(main, "get_factor_metadata", lambda: [])
    monkeypatch.setattr(main, "get_factor_set_hash", lambda: "hash")
    monkeypatch.setattr(main, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(main, "DATASET_CSV", tmp_path / "dataset.csv")
    monkeypatch.setattr(main, "DATASET_PARQUET", tmp_path / "dataset.parquet")
    monkeypatch.setattr(main, "DATASET_METADATA", tmp_path / "dataset_metadata.json")

    def _boom():
        raise AssertionError("sentiment functions should not load when disabled")

    monkeypatch.setattr(main, "_sentiment_functions", _boom)

    cfg = {"sentiment": {"enabled": False}, "dataset": {"use_panel_builder": True}}
    out = main._build_dataset_from_prices(cfg)

    assert not out.empty
    assert set(out["ticker"].unique()) == {"AAA.NS", "BBB.NS"}


def test_quality_gate_close_source_logs_only_in_debug(caplog) -> None:
    df = pd.DataFrame({"Adj Close": [1.0, 1.1], "Close": [1.0, 1.1]})

    with caplog.at_level(logging.INFO):
        quality_gates.get_canonical_price_column(df, {"qa": {"price_return_source": "auto"}})

    assert "Using adjusted close" not in caplog.text


def test_dataset_builder_uses_accurate_fallback_progress_label(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
    price_data = {"AAA.NS": pd.DataFrame({"close": [10 + i for i in range(12)]}, index=dates)}

    monkeypatch.setattr(main, "_resolve_tickers_with_meta", lambda cfg: (("AAA.NS",), {"raw_count": 1}))
    monkeypatch.setattr(main, "_collect_price_data", lambda tickers: price_data)
    monkeypatch.setattr(main, "run_quality_gates_dataset", lambda df, cfg: None)
    monkeypatch.setattr(main, "run_quality_gates_price_frame", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "register_default_factors", lambda: None)
    monkeypatch.setattr(main, "get_factor_metadata", lambda: [])
    monkeypatch.setattr(main, "get_factor_set_hash", lambda: "hash")
    monkeypatch.setattr(main, "record_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(main, "DATASET_CSV", tmp_path / "dataset.csv")
    monkeypatch.setattr(main, "DATASET_PARQUET", tmp_path / "dataset.parquet")
    monkeypatch.setattr(main, "DATASET_METADATA", tmp_path / "dataset_metadata.json")
    monkeypatch.setattr(main, "_prepare_price_frame", lambda df: df)
    monkeypatch.setattr(main, "_sanitize_price_frame_for_dataset", lambda df, ticker, cfg: (df, {}))

    seen = {"label": None}

    def _capture_progress(items, description, enabled):
        seen["label"] = description
        for item in items:
            yield item

    monkeypatch.setattr(main, "progress_iter", _capture_progress)
    monkeypatch.setattr(main, "build_price_feature_panel", lambda _price_data: pd.DataFrame())

    cfg = {"sentiment": {"enabled": False}, "dataset": {"use_panel_builder": True}, "qa": {"min_price_rows": 2, "min_non_null_ratio": 0.0, "min_eligible_tickers": 1}}
    out = main._build_dataset_from_prices(cfg)

    assert not out.empty
    assert seen["label"] == "Building per-ticker dataset"
