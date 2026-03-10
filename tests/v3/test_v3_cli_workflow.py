from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from spectraquant_v3.cli.main import app


def _price_frame(rows: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(1, rows + 1),
            "high": range(2, rows + 2),
            "low": range(1, rows + 1),
            "close": range(2, rows + 2),
            "volume": [1000] * rows,
        },
        index=idx,
    )


def test_crypto_universe_command_operational(monkeypatch):
    cfg = {
        "crypto": {
            "symbols": ["BTC", "ETH"],
            "universe_mode": "static",
            "universe_filters": {"require_exchange_coverage": False},
            "quality_gate": {"require_tradable_mapping": False},
        }
    }

    class DummyRegistry:
        def contains(self, _sym):
            return True

        def all_symbols(self):
            return ["BTC", "ETH"]

    monkeypatch.setattr("spectraquant_v3.core.config.get_crypto_config", lambda *_: cfg, raising=False)
    monkeypatch.setattr("spectraquant_v3.crypto.symbols.registry.build_registry_from_config", lambda *_: DummyRegistry(), raising=False)

    runner = CliRunner()
    result = runner.invoke(app, ["crypto", "universe"])
    assert result.exit_code == 0
    assert "symbol included reason" in result.stdout
    assert "BTC" in result.stdout
    assert "[crypto universe] included=" in result.stdout


def test_research_dataset_supports_news_and_context(monkeypatch, tmp_path):
    cfg = {"crypto": {"symbols": ["BTC"], "prices_dir": str(tmp_path / "prices")}}

    captured = {"news": None, "context": None}

    class DummyCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def read_parquet(self, _sym):
            return _price_frame()

    class DummyEngine:
        def transform_many(self, price_map, news_map=None, context_map=None):
            captured["news"] = news_map
            captured["context"] = context_map
            return {k: v.assign(ret_1d=0.01, rsi=50, volume_ratio=1.0, atr_norm=0.01, vol_realised=0.2) for k, v in price_map.items()}

    class DummyBuilder:
        def __init__(self, output_dir, run_id):
            self.output_dir = output_dir
            self.run_id = run_id

        def build(self, **_kwargs):
            class R:
                manifest_path = "dummy.json"

                def summary(self):
                    return "ok"

            return R()

    news_dir = tmp_path / "news"
    context_dir = tmp_path / "context"
    news_dir.mkdir()
    context_dir.mkdir()
    pd.DataFrame({"sentiment": [0.2]}, index=[pd.Timestamp("2024-01-01")]).to_parquet(news_dir / "BTC.parquet")
    pd.DataFrame({"btc_close": [100.0]}, index=[pd.Timestamp("2024-01-01")]).to_parquet(context_dir / "BTC.parquet")

    monkeypatch.setattr("spectraquant_v3.core.config.get_crypto_config", lambda *_: cfg, raising=False)
    monkeypatch.setattr("spectraquant_v3.core.cache.CacheManager", DummyCache, raising=False)
    monkeypatch.setattr("spectraquant_v3.crypto.features.engine.CryptoFeatureEngine.from_config", lambda _cfg: DummyEngine(), raising=False)
    monkeypatch.setattr("spectraquant_v3.research.dataset_builder.ResearchDatasetBuilder", DummyBuilder, raising=False)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "research",
            "dataset",
            "--include-news",
            "--include-context",
            "--news-dir",
            str(news_dir),
            "--context-dir",
            str(context_dir),
        ],
    )
    assert result.exit_code == 0
    assert captured["news"] and "BTC" in captured["news"]
    assert captured["context"] and "BTC" in captured["context"]


def test_backtest_strategy_wiring_for_cross_sectional(monkeypatch, tmp_path):
    cfg = {"crypto": {"symbols": ["BTC"], "prices_dir": str(tmp_path / "prices")}}
    captured = {"strategy_id": None, "persist_strategy": None}

    class DummyCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def read_parquet(self, _sym):
            return _price_frame()

    class DummyResults:
        def summary(self):
            return "summary"

        def write(self, _output_dir):
            return str(tmp_path / "result.json")

    class DummyEngine:
        def __init__(self, **kwargs):
            captured["strategy_id"] = kwargs.get("strategy_id")

        def run(self):
            return DummyResults()

    class DummyExperimentManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_experiment(self, **kwargs):
            captured["persist_strategy"] = kwargs.get("strategy_id")

    monkeypatch.setattr("spectraquant_v3.core.config.get_crypto_config", lambda *_: cfg, raising=False)
    monkeypatch.setattr("spectraquant_v3.core.cache.CacheManager", DummyCache, raising=False)
    monkeypatch.setattr("spectraquant_v3.backtest.engine.BacktestEngine", DummyEngine, raising=False)
    monkeypatch.setattr("spectraquant_v3.experiments.experiment_manager.ExperimentManager", DummyExperimentManager, raising=False)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "--strategy",
            "crypto_cross_sectional_momentum_v1",
            "--symbols",
            "BTC",
        ],
    )
    assert result.exit_code == 0
    assert captured["strategy_id"] == "crypto_cross_sectional_momentum_v1"
    assert captured["persist_strategy"] == "crypto_cross_sectional_momentum_v1"


def test_experiment_compare_shows_expanded_metrics(monkeypatch):
    class DummyManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def compare_experiments(self, _ids):
            return [
                {
                    "experiment_id": "exp1",
                    "strategy_id": "crypto_momentum_v1",
                    "sharpe": 1.2,
                    "cagr": 0.25,
                    "max_drawdown": -0.10,
                    "volatility": 0.30,
                    "win_rate": 0.6,
                    "turnover": 0.4,
                    "total_return": 0.5,
                    "calmar": 2.0,
                    "n_steps": 12,
                }
            ]

    monkeypatch.setattr("spectraquant_v3.experiments.experiment_manager.ExperimentManager", DummyManager, raising=False)

    runner = CliRunner()
    result = runner.invoke(app, ["experiment", "compare", "exp1,exp2"])
    assert result.exit_code == 0
    assert "TURNOVER" in result.stdout
    assert "TOTAL_RET" in result.stdout
    assert "CALMAR" in result.stdout
    assert "N_STEPS" in result.stdout
