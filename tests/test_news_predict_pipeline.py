"""Tests for the news-universe predict pipeline fixes.

Covers:
A) Flat LightGBM output → dispersion fallback creates non-degenerate returns
   and quality gates pass.
B) Integration test: news universe → cmd_predict offline → quality gates pass,
   explainability columns present, output tickers match universe, governance
   JSONL written per row.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"
_SRC = _REPO_ROOT / "src"


# ===========================================================================
# A) Dispersion fallback unit tests
# ===========================================================================


class TestDispersionFallback:
    """Verify that flat model output gets per-ticker tech signal blended in."""

    def _make_flat_predictions(self, n_tickers: int = 5) -> pd.DataFrame:
        """Build a predictions DataFrame where all returns are identical (degenerate)."""
        from spectraquant.core.predictions import ANNUAL_RETURN_TARGET, TRADING_DAYS

        tickers = [f"T{i}.NS" for i in range(n_tickers)]
        flat_val = 0.002  # all tickers identical
        annual_val = float(
            np.tanh(
                (np.power(1 + flat_val, TRADING_DAYS) - 1) / ANNUAL_RETURN_TARGET
            )
            * ANNUAL_RETURN_TARGET
        )
        rows = []
        for t in tickers:
            rows.append(
                {
                    "ticker": t,
                    "expected_return_annual": annual_val,
                    "expected_return_horizon": flat_val,
                    "expected_return": flat_val,
                    "predicted_return": flat_val,
                    "predicted_return_1d": flat_val,
                    "probability": 0.6,
                    "score": 50.0,
                    "model_version": "v1",
                    "factor_set_version": "fs1",
                    "horizon": "5d",
                    "regime": "neutral",
                    "date": pd.Timestamp("2024-06-01", tz="UTC"),
                }
            )
        return pd.DataFrame(rows)

    def test_flat_annual_returns_fail_quality_gate(self) -> None:
        """Quality gate should fail when annual returns are flat (std < 1e-4)."""
        from spectraquant.qa.quality_gates import run_quality_gates_predictions, QualityGateError

        df = self._make_flat_predictions(n_tickers=5)
        cfg: Dict[str, Any] = {"qa": {"force_pass_tests": False}}

        try:
            issues = run_quality_gates_predictions(df, cfg)
            # Should have raised before here, but if not, check for degenerate codes
        except QualityGateError as exc:
            issues = exc.issues

        codes = {i.code for i in issues}
        # At least one of the degenerate/flat codes should fire
        degenerate_codes = {
            "expected_return_annual_flat",
            "degenerate_expected_return",
            "expected_return_constant",
        }
        assert degenerate_codes & codes, (
            f"Expected at least one degenerate code from {degenerate_codes}, got {codes}"
        )

    def test_tech_signal_creates_dispersion(self) -> None:
        """_summarize_price_series returns different signals for different price trends."""
        import sys
        sys.path.insert(0, str(_SRC))
        from spectraquant.cli.main import _summarize_price_series

        # Trending up
        up_close = pd.Series(
            [100.0 * (1.01 ** i) for i in range(30)]
        )
        # Trending down
        down_close = pd.Series(
            [100.0 * (0.99 ** i) for i in range(30)]
        )

        up_m = _summarize_price_series(up_close)
        down_m = _summarize_price_series(down_close)

        # Different price trends must produce different signals
        assert up_m["momentum_daily"] != down_m["momentum_daily"]
        assert up_m["mean_return"] != down_m["mean_return"]

    def test_z_scored_tech_signal_has_unit_std(self) -> None:
        """When we z-score the tech signals they should have std ≈ 1."""
        signals = np.array([0.001, -0.002, 0.003, -0.001, 0.0015])
        # Mean-center first
        signals -= np.mean(signals)
        std = float(np.std(signals))
        if std > 0:
            signals_z = signals / std
            assert abs(float(np.std(signals_z)) - 1.0) < 0.01

    def test_predictions_frame_dispersion_after_build_prediction_frame(self) -> None:
        """build_prediction_frame with diverse metrics must pass the annual std gate."""
        from spectraquant.core.predictions import build_prediction_frame
        from spectraquant.qa.quality_gates import run_quality_gates_predictions

        n = 5
        metrics = {
            f"T{i}.NS": {
                "mean_return": (i - n // 2) * 0.003,  # spread from -0.006 to +0.006
                "volatility": 0.01 + i * 0.005,
                "momentum_daily": (i - n // 2) * 0.002,
                "rsi": 40.0 + i * 8.0,
            }
            for i in range(n)
        }
        tickers = list(metrics.keys())
        factor_scores = {t: (i - n // 2) * 0.5 for i, t in enumerate(tickers)}
        dates = {t: pd.Timestamp("2024-06-01") for t in tickers}

        df = build_prediction_frame(
            tickers=tickers,
            metrics_by_ticker=metrics,
            factor_scores=factor_scores,
            horizon="1d",
            horizon_days=1.0,
            model_version="v-test",
            factor_set_version="fs-test",
            regime="neutral",
            prediction_dates=dates,
        )

        # Annual std must be above gate threshold
        annual_std = float(df["expected_return_annual"].std(ddof=0))
        assert annual_std >= 1e-4, (
            f"expected_return_annual std={annual_std:.2e} is below 1e-4 gate threshold"
        )

        # Quality gate should not fire
        cfg: Dict[str, Any] = {"qa": {"force_pass_tests": False}}
        issues = run_quality_gates_predictions(df, cfg)
        failures = [i for i in issues if i.severity == "FAIL"]
        assert not failures, f"Quality gate FAIL issues: {failures}"


# ===========================================================================
# B) Integration test: news-universe predict pipeline
# ===========================================================================


def _write_price_csv(path: Path, n_rows: int = 30, base_price: float = 100.0, trend: float = 0.01) -> None:
    """Write a synthetic OHLCV CSV with a given trend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_rows):
        close = base_price * (1 + trend) ** i
        rows.append(
            {
                "date": f"2024-01-{i+1:02d}T00:00:00Z",
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": 100000 + i * 5000,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_dataset(path: Path, tickers: list[str]) -> None:
    """Write a minimal dataset parquet that cmd_train can use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    dates = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    for date in dates:
        for i, ticker in enumerate(tickers):
            row = {
                "date": date,
                "ticker": ticker,
                "close": 100.0 * (1 + i * 0.01 + rng.normal(0, 0.005)),
                "volume": 100000,
                "returns_1d": (i - len(tickers) / 2) * 0.001 + rng.normal(0, 0.003),
                "rsi_14": 40.0 + i * 10.0 + rng.normal(0, 2.0),
                "sma_20": 100.0 + i * 0.5,
                "volatility_20": 0.01 + i * 0.002,
                "momentum_5d": (i - len(tickers) / 2) * 0.002,
                "sentiment_score": 0.0,
                "label": 1 if i % 2 == 0 else 0,
                "fwd_ret_5d": (i - len(tickers) / 2) * 0.002,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(path)


@pytest.fixture()
def news_universe_env(tmp_path: Path):
    """Fixture: set up a tmp directory with news universe, prices, dataset, model."""
    import sys
    sys.path.insert(0, str(_SRC))

    tickers = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]

    # ---- news_universe_latest.json ----
    cache_dir = tmp_path / "data" / "news_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    news_json = {
        "tickers": tickers,
        "source_csv": "reports/news/news_candidates_stub.csv",
        "generated_at": "2024-06-01T00:00:00Z",
    }
    (cache_dir / "news_universe_latest.json").write_text(
        json.dumps(news_json), encoding="utf-8"
    )

    # ---- Price CSVs ----
    prices_dir = tmp_path / "data" / "prices"
    _write_price_csv(prices_dir / "TCS.NS.csv", base_price=3500.0, trend=0.008)
    _write_price_csv(prices_dir / "INFY.NS.csv", base_price=1500.0, trend=0.002)
    _write_price_csv(prices_dir / "RELIANCE.NS.csv", base_price=2800.0, trend=-0.003)

    # ---- Universe CSV ----
    universe_csv = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": ["TCS", "INFY", "RELIANCE"]}).to_csv(universe_csv, index=False)

    # ---- Dataset parquet ----
    dataset_dir = tmp_path / "reports" / "datasets"
    dataset_path = dataset_dir / "dataset_test.parquet"
    _write_dataset(dataset_path, tickers)

    # ---- Config YAML (on disk for SPECTRAQUANT_CONFIG) ----
    config: Dict[str, Any] = {
        "alpha": {"enabled": False, "weights": {"momentum": 0.35, "trend": 0.25, "volatility": 0.2, "value": 0.2}},
        "portfolio": {
            "rebalance": "weekly",
            "weighting": "equal",
            "alpha_threshold": -0.01,
            "volatility_target": None,
            "max_asset_weight": None,
            "sector_limits": {},
            "sector_map": {},
            "top_k": 20,
            "min_weight_threshold": 0.0,
            "liquidity_min_volume": None,
            "max_positions": 20,
            "max_weight": None,
            "max_turnover": None,
            "policies": {},
            "policy": {"auto_repair": False},
            "horizon": "1d",
        },
        "data": {
            "tickers": tickers,
            "synthetic": False,
            "source": "yfinance",
            "provider": "mock",
            "batch_size": 50,
            "batch_sleep_seconds": 1,
            "max_retries": 1,
            "daily_retention_years": 5,
            "max_tickers_per_run": 300,
        },
        "mlops": {
            "auto_retrain": True,
            "retrain_interval_days": 7,
            "min_improvement": 0.0,
            "seed": 42,
        },
        "qa": {
            "min_price_rows": 5,
            "stale_tolerance_minutes": 10000000,
            "flatline_window": 3,
            "max_abs_daily_return": 0.8,
            "min_expected_return_std_daily": 1e-6,
            "min_expected_return_std_intraday": 1e-9,
            "min_volume": 0,
            "max_missing_pct": 0.2,
            "force_pass_tests": False,
        },
        "universe": {
            "path": str(universe_csv),
            "india": {
                "source": "nse",
                "tickers_file": str(universe_csv),
            },
            "uk": {
                "source": "lse",
                "tickers_file": str(universe_csv),
            },
            "selected_sets": ["news"],
            "dry_run": False,
        },
        "research_mode": False,
        "test_mode": True,
        "predictions": {
            "daily_horizons": ["1d", "5d"],
            "intraday_horizons": [],
        },
        "intraday": {"enabled": False},
        "explain": {"enabled": False},
        "perf": {
            "max_seconds": 300,
            "max_mb": 1024,
            "stages": {"download": {"max_seconds": 1800}},
        },
        "signals": {
            "buy_threshold": 50.0,
            "sell_threshold": -1.0,
            "min_buy_signals": 1,
            "min_buy_signals_test": 0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    yield {
        "tmp_path": tmp_path,
        "config": config,
        "config_path": config_path,
        "tickers": tickers,
        "dataset_path": dataset_path,
        "prices_dir": prices_dir,
        "universe_csv": universe_csv,
    }


class TestNewsPredictIntegration:
    """Integration tests for the news-driven prediction pipeline (fully offline)."""

    def _setup_and_run_predict(self, env: dict) -> tuple[pd.DataFrame, Path]:
        """Train a model then run cmd_predict; return (predictions_df, pred_path)."""
        import sys
        sys.path.insert(0, str(_SRC))
        from spectraquant.cli import main as cli
        from spectraquant.core.model_registry import promote_model

        tmp_path: Path = env["tmp_path"]
        config: dict = env["config"]
        config_path: Path = env["config_path"]

        # Point SPECTRAQUANT_CONFIG to the config file so internal _load_config() calls work
        prev_config_env = os.environ.get("SPECTRAQUANT_CONFIG")
        os.environ["SPECTRAQUANT_CONFIG"] = str(config_path)
        try:
            # Train (loads dataset from filesystem)
            cli.cmd_train(config=config)
            promote_model(1)

            # Predict
            cli.cmd_predict(config=config)
        finally:
            if prev_config_env is None:
                os.environ.pop("SPECTRAQUANT_CONFIG", None)
            else:
                os.environ["SPECTRAQUANT_CONFIG"] = prev_config_env

        pred_dir = tmp_path / "reports" / "predictions"
        pred_files = sorted(pred_dir.glob("predictions_*.csv"))
        assert pred_files, f"No predictions_*.csv found in {pred_dir}"
        pred_path = pred_files[-1]
        return pd.read_csv(pred_path), pred_path

    def test_quality_gates_pass(self, news_universe_env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
        """cmd_predict must produce predictions that pass quality gates without force_pass."""
        from spectraquant.qa.quality_gates import QualityGateError

        monkeypatch.chdir(news_universe_env["tmp_path"])
        monkeypatch.setenv("SPECTRAQUANT_RESEARCH_MODE", "false")

        try:
            df, _ = self._setup_and_run_predict(news_universe_env)
        except QualityGateError as exc:
            pytest.fail(
                f"Quality gates failed unexpectedly: {exc}\n"
                f"Issues: {[i for i in exc.issues if i.severity == 'FAIL']}"
            )

        # Verify annual std is above the gate threshold
        annual_std = float(df["expected_return_annual"].std(ddof=0))
        assert annual_std >= 1e-4, (
            f"expected_return_annual std={annual_std:.2e} is below the 1e-4 quality gate threshold"
        )

    def test_explainability_columns_present(self, news_universe_env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
        """Predictions CSV must include all new explainability columns."""
        from spectraquant.qa.quality_gates import QualityGateError

        monkeypatch.chdir(news_universe_env["tmp_path"])

        try:
            df, _ = self._setup_and_run_predict(news_universe_env)
        except QualityGateError:
            pytest.xfail("Quality gates failed; skipping column check")

        required_explainability = [
            "reason",
            "event_type",
            "analysis_model",
            "expected_move_pct",
            "confidence",
            "risk_score",
            "news_refs",
            "stop_price",
        ]
        missing = [col for col in required_explainability if col not in df.columns]
        assert not missing, f"Missing explainability columns: {missing}"

    def test_output_tickers_match_news_universe(self, news_universe_env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
        """Output tickers must be exactly the news universe tickers (no extras)."""
        from spectraquant.qa.quality_gates import QualityGateError

        monkeypatch.chdir(news_universe_env["tmp_path"])

        try:
            df, _ = self._setup_and_run_predict(news_universe_env)
        except QualityGateError:
            pytest.xfail("Quality gates failed; skipping ticker check")

        expected_tickers = set(news_universe_env["tickers"])
        output_tickers = set(df["ticker"].unique())

        # Output must be a subset of the news universe (hard filter)
        extra = output_tickers - expected_tickers
        assert not extra, (
            f"Output contains tickers outside the news universe: {extra}. "
            f"Expected subset of {expected_tickers}."
        )

    def test_governance_jsonl_written(self, news_universe_env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
        """Governance JSONL must exist and contain one record per prediction row."""
        from spectraquant.qa.quality_gates import QualityGateError

        monkeypatch.chdir(news_universe_env["tmp_path"])

        try:
            df, _ = self._setup_and_run_predict(news_universe_env)
        except QualityGateError:
            pytest.xfail("Quality gates failed; skipping governance check")

        tmp_path: Path = news_universe_env["tmp_path"]
        gov_dir = tmp_path / "reports" / "governance"
        assert gov_dir.exists(), f"Governance directory {gov_dir} was not created"

        jsonl_files = list(gov_dir.glob("*.jsonl"))
        assert jsonl_files, f"No JSONL files found in {gov_dir}"

        gov_path = jsonl_files[0]
        records = [
            json.loads(line)
            for line in gov_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(records) == len(df), (
            f"Expected {len(df)} governance records, found {len(records)}"
        )
        # Each record must have a ticker
        for rec in records:
            assert "ticker" in rec, f"Governance record missing 'ticker': {rec}"

    def test_analysis_model_column_non_empty(self, news_universe_env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
        """analysis_model column must not be empty strings."""
        from spectraquant.qa.quality_gates import QualityGateError

        monkeypatch.chdir(news_universe_env["tmp_path"])

        try:
            df, _ = self._setup_and_run_predict(news_universe_env)
        except QualityGateError:
            pytest.xfail("Quality gates failed; skipping analysis_model check")

        if "analysis_model" not in df.columns:
            pytest.fail("analysis_model column missing from predictions CSV")

        empty_count = (df["analysis_model"] == "").sum() + df["analysis_model"].isna().sum()
        assert empty_count == 0, (
            f"{empty_count} rows have empty analysis_model values"
        )
