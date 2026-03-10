import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spectraquant.cli.main as cli_main
from spectraquant.core.perf import _rss_mb_from_ru_maxrss, enforce_stage_budget
from spectraquant.qa.quality_gates import QualityGateError, run_quality_gates_price_frame


def test_rss_conversion_branches() -> None:
    assert _rss_mb_from_ru_maxrss(1024 * 1024, "Darwin") == pytest.approx(1.0)
    assert _rss_mb_from_ru_maxrss(1024, "Linux") == pytest.approx(1.0)


def test_perf_budget_does_not_mask_exception() -> None:
    config = {"perf": {"max_seconds": 0.0, "max_mb": 0.0}, "research_mode": False}
    with pytest.raises(ValueError):
        with enforce_stage_budget("predict", config):
            raise ValueError("boom")


def test_quality_gate_logs_failures(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [0.0, 0.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
    )
    caplog.set_level(logging.ERROR)
    with pytest.raises(QualityGateError):
        run_quality_gates_price_frame(df, ticker="TEST.NS", exchange="NSE", interval="1d", cfg={})
    assert "Quality gate failures" in caplog.text


def test_dataset_build_drops_short_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    good_index = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    bad_index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    base_columns = {
        "Open": np.linspace(100, 160, num=len(good_index)),
        "High": np.linspace(101, 161, num=len(good_index)),
        "Low": np.linspace(99, 159, num=len(good_index)),
        "Close": np.linspace(100, 160, num=len(good_index)),
        "Volume": np.linspace(1000, 2000, num=len(good_index)),
    }
    good_df = pd.DataFrame(base_columns, index=good_index)
    bad_df = pd.DataFrame(
        {
            "Open": np.linspace(100, 104, num=len(bad_index)),
            "High": np.linspace(101, 105, num=len(bad_index)),
            "Low": np.linspace(99, 103, num=len(bad_index)),
            "Close": np.linspace(100, 104, num=len(bad_index)),
            "Volume": np.linspace(1000, 1004, num=len(bad_index)),
        },
        index=bad_index,
    )

    monkeypatch.setattr(
        cli_main, "_resolve_tickers_with_meta", lambda _config: (("GOOD.NS", "BAD.NS"), {"raw_count": 2})
    )
    monkeypatch.setattr(cli_main, "_collect_price_data", lambda _tickers: {"GOOD.NS": good_df, "BAD.NS": bad_df})
    monkeypatch.setattr(cli_main, "DATASET_CSV", tmp_path / "dataset.csv")
    monkeypatch.setattr(cli_main, "DATASET_PARQUET", tmp_path / "dataset.parquet")
    monkeypatch.setattr(cli_main, "DATASET_METADATA", tmp_path / "dataset_metadata.json")
    monkeypatch.setattr(cli_main, "PROCESSED_DIR", tmp_path / "processed")

    config = {
        "qa": {"min_price_rows": 50, "min_non_null_ratio": 0.9},
        "test_mode": {"enabled": True},
        "sentiment": {"enabled": False},
    }

    dataset = cli_main._build_dataset_from_prices(config)
    assert dataset["ticker"].unique().tolist() == ["GOOD.NS"]


def _make_price_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 160, num=len(index)),
            "High": np.linspace(101, 161, num=len(index)),
            "Low": np.linspace(99, 159, num=len(index)),
            "Close": np.linspace(100, 160, num=len(index)),
            "Volume": np.linspace(1000, 2000, num=len(index)),
        },
        index=index,
    )


def test_dataset_allows_small_universe_in_test_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tickers = tuple(f"TICK{i}.NS" for i in range(5))
    index = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    base_df = _make_price_df(index)

    monkeypatch.setattr(
        cli_main,
        "_resolve_tickers_with_meta",
        lambda _config: (tickers, {"raw_count": len(tickers), "source": "explicit"}),
    )
    monkeypatch.setattr(
        cli_main, "_collect_price_data", lambda _tickers: {ticker: base_df.copy() for ticker in tickers}
    )
    monkeypatch.setattr(cli_main, "DATASET_CSV", tmp_path / "dataset.csv")
    monkeypatch.setattr(cli_main, "DATASET_PARQUET", tmp_path / "dataset.parquet")
    monkeypatch.setattr(cli_main, "DATASET_METADATA", tmp_path / "dataset_metadata.json")
    monkeypatch.setattr(cli_main, "PROCESSED_DIR", tmp_path / "processed")

    config = {
        "qa": {"min_price_rows": 30, "min_non_null_ratio": 0.9, "min_eligible_tickers": 10},
        "test_mode": {"enabled": True},
        "sentiment": {"enabled": False},
        "universe": {"tickers": list(tickers)},
    }

    dataset = cli_main._build_dataset_from_prices(config)
    assert sorted(dataset["ticker"].unique().tolist()) == sorted(tickers)


def test_dataset_small_universe_respects_floor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tickers = ("ONLY1.NS", "ONLY2.NS")
    index = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    base_df = _make_price_df(index)

    monkeypatch.setattr(
        cli_main,
        "_resolve_tickers_with_meta",
        lambda _config: (tickers, {"raw_count": len(tickers), "source": "explicit"}),
    )
    monkeypatch.setattr(
        cli_main, "_collect_price_data", lambda _tickers: {ticker: base_df.copy() for ticker in tickers}
    )
    monkeypatch.setattr(cli_main, "DATASET_CSV", tmp_path / "dataset.csv")
    monkeypatch.setattr(cli_main, "DATASET_PARQUET", tmp_path / "dataset.parquet")
    monkeypatch.setattr(cli_main, "DATASET_METADATA", tmp_path / "dataset_metadata.json")
    monkeypatch.setattr(cli_main, "PROCESSED_DIR", tmp_path / "processed")

    config = {
        "qa": {"min_price_rows": 30, "min_non_null_ratio": 0.9, "min_eligible_tickers": 10},
        "test_mode": {"enabled": False},
        "sentiment": {"enabled": False},
        "universe": {"tickers": list(tickers)},
    }

    with pytest.raises(RuntimeError):
        cli_main._build_dataset_from_prices(config)


def test_dataset_small_universe_allows_above_floor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tickers = tuple(f"SMALL{i}.NS" for i in range(5))
    index = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    base_df = _make_price_df(index)

    monkeypatch.setattr(
        cli_main,
        "_resolve_tickers_with_meta",
        lambda _config: (tickers, {"raw_count": len(tickers), "source": "explicit"}),
    )
    monkeypatch.setattr(
        cli_main, "_collect_price_data", lambda _tickers: {ticker: base_df.copy() for ticker in tickers}
    )
    monkeypatch.setattr(cli_main, "DATASET_CSV", tmp_path / "dataset.csv")
    monkeypatch.setattr(cli_main, "DATASET_PARQUET", tmp_path / "dataset.parquet")
    monkeypatch.setattr(cli_main, "DATASET_METADATA", tmp_path / "dataset_metadata.json")
    monkeypatch.setattr(cli_main, "PROCESSED_DIR", tmp_path / "processed")

    config = {
        "qa": {"min_price_rows": 30, "min_non_null_ratio": 0.9, "min_eligible_tickers": 10},
        "test_mode": {"enabled": False},
        "sentiment": {"enabled": False},
        "universe": {"tickers": list(tickers)},
    }

    dataset = cli_main._build_dataset_from_prices(config)
    assert sorted(dataset["ticker"].unique().tolist()) == sorted(tickers)
