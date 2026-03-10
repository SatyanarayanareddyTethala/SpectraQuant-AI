#!/usr/bin/env python
"""Deterministic replay harness using persisted artifacts only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraquant.cli.main import (
    cmd_build_dataset,
    cmd_execute,
    cmd_portfolio,
    cmd_predict,
    cmd_signals,
    cmd_train,
)
from spectraquant.core.model_registry import promote_model
from spectraquant.qa.hash_utils import hash_file


def _hash_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_manifest(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"replay_manifest_{payload['run_id']}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic replay harness.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--universe", required=True, help="Path to universe CSV")
    parser.add_argument("--prices-dir", default="data/prices", help="Directory with cached prices")
    parser.add_argument("--model-version", default=None, help="Optional model version")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    universe_path = Path(args.universe).resolve()
    prices_dir = Path(args.prices_dir).resolve()

    if not prices_dir.exists():
        raise FileNotFoundError(f"Prices directory missing: {prices_dir}")
    if not list(prices_dir.glob("*.csv")) and not list(prices_dir.glob("*.parquet")):
        raise FileNotFoundError("No price artifacts available for replay.")

    cfg = yaml.safe_load(config_path.read_text()) or {}
    cfg.setdefault("universe", {})
    cfg["universe"].setdefault("india", {})
    cfg["universe"].setdefault("uk", {})
    cfg["universe"]["india"]["source"] = "csv"
    cfg["universe"]["india"]["path"] = str(universe_path)
    cfg["universe"]["uk"]["tickers_file"] = str(universe_path)
    cfg["research_mode"] = False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_config = Path(tmpdir) / "config.yaml"
        tmp_config.write_text(yaml.safe_dump(cfg))
        os.environ["SPECTRAQUANT_CONFIG"] = str(tmp_config)
        import spectraquant.config as config_module
        config_module.CONFIG_PATH = tmp_config

        cmd_build_dataset()
        cmd_train()
        try:
            promote_model(1)
        except Exception:
            pass
        cmd_predict()
        cmd_signals()
        cmd_portfolio()
        try:
            cmd_execute()
        except Exception:
            pass

    manifest_dir = Path("reports/manifests")
    outputs = {}
    pred_files = sorted((Path("reports/predictions")).glob("predictions_*.csv"))
    if pred_files:
        outputs["predictions_latest"] = hash_file(pred_files[-1])
    signal_files = sorted((Path("reports/signals")).glob("top_signals_*.csv"))
    if signal_files:
        outputs["signals_latest"] = hash_file(signal_files[-1])
    portfolio_returns = Path("reports/portfolio/portfolio_returns.csv")
    portfolio_weights = Path("reports/portfolio/portfolio_weights.csv")
    if portfolio_returns.exists():
        outputs["portfolio_returns"] = hash_file(portfolio_returns)
    if portfolio_weights.exists():
        outputs["portfolio_weights"] = hash_file(portfolio_weights)
    execution_trades = Path("reports/execution/trades.csv")
    execution_fills = Path("reports/execution/fills.csv")
    execution_costs = Path("reports/execution/costs.csv")
    execution_pnl = Path("reports/execution/daily_pnl.csv")
    if execution_trades.exists():
        outputs["execution_trades"] = hash_file(execution_trades)
    if execution_fills.exists():
        outputs["execution_fills"] = hash_file(execution_fills)
    if execution_costs.exists():
        outputs["execution_costs"] = hash_file(execution_costs)
    if execution_pnl.exists():
        outputs["execution_pnl"] = hash_file(execution_pnl)

    payload = {
        "run_id": Path(config_path).stem,
        "config_hash": _hash_text(config_path.read_text()),
        "universe_hash": _hash_text(universe_path.read_text()),
        "input_prices": {
            **{p.name: hash_file(p) for p in prices_dir.glob("*.*")},
            **{
                f"intraday/{p.name}": hash_file(p)
                for p in (prices_dir / "intraday").glob("*.*")
                if (prices_dir / "intraday").exists()
            },
        },
        "outputs": outputs,
    }
    manifest = _write_manifest(manifest_dir, payload)
    print(f"Replay manifest written to {manifest}")


if __name__ == "__main__":
    main()
