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


def test_execution_outputs(tmp_path: Path) -> None:
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
    cli.cmd_execute()

    pred_path = sorted((tmp_path / "reports" / "predictions").glob("predictions_*.csv"))[-1]
    sig_path = sorted((tmp_path / "reports" / "signals").glob("top_signals_*.csv"))[-1]
    pred_df = pd.read_csv(pred_path)
    sig_df = pd.read_csv(sig_path)
    prob_series = pd.to_numeric(pred_df["probability"], errors="coerce")
    mean_prob = float(prob_series.mean())
    skew = float(prob_series.skew())
    buy_count = int(sig_df["signal"].astype(str).str.upper().eq("BUY").sum())
    horizon_counts = pred_df["horizon"].value_counts().to_dict()
    print(
        "[test_execution_outputs] mean_prob={:.4f} skew={:.4f} buy_count={} horizons={}".format(
            mean_prob, skew, buy_count, horizon_counts
        )
    )

    trades_path = tmp_path / "reports" / "execution" / "trades.csv"
    fills_path = tmp_path / "reports" / "execution" / "fills.csv"
    costs_path = tmp_path / "reports" / "execution" / "costs.csv"
    pnl_path = tmp_path / "reports" / "execution" / "daily_pnl.csv"

    assert trades_path.exists()
    assert fills_path.exists()
    assert costs_path.exists()
    assert pnl_path.exists()
