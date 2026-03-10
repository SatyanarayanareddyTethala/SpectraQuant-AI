from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

from spectraquant.cli import main as cli

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_cmd_portfolio_single_date_signals_switches_to_single_step(tmp_path: Path, caplog) -> None:
    shutil.copytree(FIXTURES / "prices", tmp_path / "data" / "prices", dirs_exist_ok=True)

    config = yaml.safe_load((FIXTURES / "config.yaml").read_text())
    config["alpha"]["enabled"] = False
    config["portfolio"]["rebalance"] = "monthly"
    config["execution"]["mode"] = "eod"
    universe_path = tmp_path / "universe_small.csv"
    universe_path.write_text("ticker\nTICKER1.NS\nTICKER2.L\n")
    config["universe"]["india"]["tickers_file"] = str(universe_path)
    config["universe"]["uk"]["tickers_file"] = str(universe_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    signals_dir = tmp_path / "reports" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    signals_df = pd.DataFrame(
        {
            "date": ["2024-01-10", "2024-01-10"],
            "ticker": ["TICKER1.NS", "TICKER2.L"],
            "signal": ["BUY", "BUY"],
            "score": [0.9, 0.8],
            "horizon": ["1d", "1d"],
        }
    )
    signals_df.to_csv(signals_dir / "top_signals_20250101_000000.csv", index=False)

    os.environ["SPECTRAQUANT_CONFIG"] = str(config_path)
    os.chdir(tmp_path)

    caplog.set_level(logging.INFO)
    cli.cmd_portfolio()

    assert "Detected point-in-time signals" in caplog.text
    assert "No BUY signals at" not in caplog.text

    weights_path = tmp_path / "reports" / "portfolio" / "portfolio_weights.csv"
    weights_df = pd.read_csv(weights_path)
    assert not weights_df.empty
    ticker_cols = [c for c in weights_df.columns if c.startswith("TICKER")]
    assert ticker_cols
    assert float(weights_df[ticker_cols].sum(axis=1).iloc[-1]) > 0
