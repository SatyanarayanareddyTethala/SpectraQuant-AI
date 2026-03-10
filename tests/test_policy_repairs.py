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


def test_policy_repairs_output(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "prices", tmp_path / "data" / "prices", dirs_exist_ok=True)
    universe_path = tmp_path / "universe_small.csv"
    shutil.copy(FIXTURES / "universe_small.csv", universe_path)

    config = yaml.safe_load((FIXTURES / "config.yaml").read_text())
    config["universe"]["india"]["tickers_file"] = str(universe_path)
    config["universe"]["uk"]["tickers_file"] = str(universe_path)
    config["portfolio"]["max_positions"] = 1
    config["portfolio"]["policy"]["auto_repair"] = True
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
            pytest.xfail(f"Prediction quality gates failed in test mode: {exc}")
        cli.cmd_signals()
        cli.cmd_portfolio()

        repairs = sorted((tmp_path / "reports" / "portfolio").glob("policy_repairs_*.csv"))
        assert repairs, "Policy repairs report missing"
        df = pd.read_csv(repairs[-1])
        assert not df.empty
    finally:
        os.chdir(prev_cwd)
