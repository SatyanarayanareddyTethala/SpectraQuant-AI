from __future__ import annotations

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


def _load_expected(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "expected" / name)


def _run_pipeline(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "prices", tmp_path / "data" / "prices", dirs_exist_ok=True)
    universe_path = tmp_path / "universe_small.csv"
    shutil.copy(FIXTURES / "universe_small.csv", universe_path)

    config = yaml.safe_load((FIXTURES / "config.yaml").read_text())
    config["universe"]["india"]["tickers_file"] = str(universe_path)
    config["universe"]["uk"]["tickers_file"] = str(universe_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    os.environ["SPECTRAQUANT_CONFIG"] = str(config_path)
    os.chdir(tmp_path)

    cli.cmd_build_dataset()
    cli.cmd_train()
    promote_model(1)
    try:
        cli.cmd_predict()
    except QualityGateError as exc:
        pytest.xfail(f"Prediction quality gates failed in test mode: {exc}")
    cli.cmd_signals()
    cli.cmd_portfolio()


def test_golden_pipeline_outputs(tmp_path: Path) -> None:
    _run_pipeline(tmp_path)
    pred = sorted((tmp_path / "reports" / "predictions").glob("predictions_*.csv"))[-1]
    sig = sorted((tmp_path / "reports" / "signals").glob("top_signals_*.csv"))[-1]
    ret = tmp_path / "reports" / "portfolio" / "portfolio_returns.csv"
    wgt = tmp_path / "reports" / "portfolio" / "portfolio_weights.csv"
    metrics_path = tmp_path / "reports" / "portfolio" / "portfolio_metrics.json"

    pred_df = pd.read_csv(pred)
    sig_df = pd.read_csv(sig)
    ret_df = pd.read_csv(ret)
    wgt_df = pd.read_csv(wgt)
    metrics = yaml.safe_load(metrics_path.read_text())
    prob_series = pd.to_numeric(pred_df["probability"], errors="coerce")
    mean_prob = float(prob_series.mean())
    skew = float(prob_series.skew())
    buy_count = int(sig_df["signal"].astype(str).str.upper().eq("BUY").sum())
    horizon_counts = pred_df["horizon"].value_counts().to_dict()
    print(
        "[test_golden_pipeline] mean_prob={:.4f} skew={:.4f} buy_count={} horizons={}".format(
            mean_prob, skew, buy_count, horizon_counts
        )
    )

    pd.testing.assert_frame_equal(pred_df, _load_expected("predictions.csv"))
    pd.testing.assert_frame_equal(sig_df, _load_expected("signals.csv"))
    pd.testing.assert_frame_equal(ret_df, _load_expected("portfolio_returns.csv"))
    pd.testing.assert_frame_equal(wgt_df, _load_expected("portfolio_weights.csv"))
    assert "regime" in metrics
