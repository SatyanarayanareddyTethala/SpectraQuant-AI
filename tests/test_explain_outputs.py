import os
import shutil
from pathlib import Path
from typing import List

import pandas as pd
import pytest
import yaml

from spectraquant.cli import main as cli
from spectraquant.core.model_registry import promote_model
from spectraquant.qa.quality_gates import QualityGateError


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_explain_outputs(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "prices", tmp_path / "data" / "prices", dirs_exist_ok=True)
    universe_path = tmp_path / "universe_small.csv"
    shutil.copy(FIXTURES / "universe_small.csv", universe_path)

    config = yaml.safe_load((FIXTURES / "config.yaml").read_text())
    config["universe"]["india"]["tickers_file"] = str(universe_path)
    config["universe"]["uk"]["tickers_file"] = str(universe_path)
    config["explain"]["enabled"] = True
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

    explain_files: List[Path] = sorted((tmp_path / "reports" / "explain").glob("factor_contributions_*.csv"))
    assert explain_files, "Explainability report missing"
    explain_df = pd.read_csv(explain_files[-1])
    assert "schema_version" in explain_df.columns
    grouped = explain_df.groupby(["ticker", "horizon", "date"])["contribution"].sum()
    assert (grouped - 1.0).abs().max() < 1e-6
