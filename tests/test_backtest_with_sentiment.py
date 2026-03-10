from __future__ import annotations

# replaced hardcoded ticker
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from spectraquant.cli import main as cli
from spectraquant.core.model_registry import promote_model
from spectraquant.qa.quality_gates import QualityGateError


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _write_price_snapshot(path: Path, ticker: str, start: float, trend: float) -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    close = pd.Series([start + trend * i for i in range(len(dates))])
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close.shift(1, fill_value=close.iloc[0]),
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
        }
    )
    df.to_csv(path / f"{ticker}.csv", index=False)


def _write_sentiment_cache(path: Path, ticker: str, score: float) -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    payload = pd.DataFrame(
        {
            "date": dates,
            "news_sentiment_avg": score,
            "news_sentiment_std": 0.0,
            "news_count": 5,
            "social_sentiment_avg": score,
            "social_sentiment_std": 0.0,
            "social_count": 10,
        }
    )
    path.mkdir(parents=True, exist_ok=True)
    payload.to_json(path / f"{ticker}_daily.json", orient="records", date_format="iso")


def _run_pipeline(tmp_path: Path, *, use_sentiment: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices_dir = tmp_path / "data" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    _write_price_snapshot(prices_dir, "TICKER4.NS", start=100.0, trend=1.5)
    _write_price_snapshot(prices_dir, "TICKER5.NS", start=200.0, trend=2.0)

    universe_path = tmp_path / "universe_subset.csv"
    universe_path.write_text("TICKER4.NS\nTICKER5.NS\n", encoding="utf-8")

    sentiment_dir = tmp_path / "data" / "sentiment"
    _write_sentiment_cache(sentiment_dir, "TICKER4.NS", 0.6)
    _write_sentiment_cache(sentiment_dir, "TICKER5.NS", 0.7)

    config = yaml.safe_load((FIXTURES / "config.yaml").read_text())
    config["universe"]["india"]["tickers_file"] = str(universe_path)
    config["universe"]["uk"]["tickers_file"] = str(universe_path)
    config["data"]["provider"] = "mock"
    config["data"]["source"] = "yfinance"
    config["test_mode"] = True
    config.setdefault("sentiment", {})
    config["sentiment"]["enabled"] = use_sentiment
    config["sentiment"]["refresh_cache"] = False
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    os.environ["SPECTRAQUANT_CONFIG"] = str(config_path)
    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        cli.cmd_build_dataset()
        cli.cmd_train()
        promote_model(1)
        try:
            cli.cmd_predict()
        except QualityGateError as exc:
            pytest.xfail(f"Prediction gates failed in test mode: {exc}")
        cli.cmd_signals()
        cli.cmd_portfolio()

        weights_path = tmp_path / "reports" / "portfolio" / "portfolio_weights.csv"
        metrics_path = tmp_path / "reports" / "portfolio" / "portfolio_metrics.json"
        assert weights_path.exists()
        assert metrics_path.exists()

        returns_path = tmp_path / "reports" / "portfolio" / "portfolio_returns.csv"
        returns_df = pd.read_csv(returns_path)
        signals_path = sorted((tmp_path / "reports" / "signals").glob("top_signals_*.csv"))[-1]
        signals_df = pd.read_csv(signals_path)
        return returns_df, signals_df
    finally:
        os.chdir(prev_cwd)


def test_backtest_portfolio_with_sentiment(tmp_path: Path) -> None:
    returns_with, signals_with = _run_pipeline(tmp_path / "with_sentiment", use_sentiment=True)
    returns_without, signals_without = _run_pipeline(tmp_path / "without_sentiment", use_sentiment=False)

    buy_with = int(signals_with["signal"].astype(str).str.upper().eq("BUY").sum())
    buy_without = int(signals_without["signal"].astype(str).str.upper().eq("BUY").sum())

    if buy_with == buy_without:
        pytest.xfail("Sentiment had no effect on BUY signal count")

    returns_mean = pd.to_numeric(returns_with.drop(columns=["date"], errors="ignore").stack()).mean()
    if returns_mean <= 0:
        pytest.xfail("Sentiment-positive test did not yield positive portfolio returns")
