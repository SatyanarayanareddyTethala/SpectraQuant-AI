from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from spectraquant.analysis.feature_pruning import analyze_feature_pruning
from spectraquant.analysis.model_comparison import compare_models
from spectraquant.analysis.run_comparison import compare_runs
from spectraquant.explain.portfolio_rationale import build_portfolio_rationale
from spectraquant.stress.param_sensitivity import run_param_sensitivity
from spectraquant.stress.regime_performance import analyze_regime_performance


def _write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def test_feature_pruning_output_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    dates = pd.date_range("2023-01-01", periods=30, freq="D", tz="UTC")
    tickers = ["AAA", "BBB"]
    rng = np.random.default_rng(42)
    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append({
                "date": date,
                "ticker": ticker,
                "feat_a": float(rng.random()),
                "feat_b": float(rng.random()),
            })
    df = pd.DataFrame(rows)
    df["fwd_ret_5d"] = df["feat_a"] * 0.1
    df["up_5d"] = (df["feat_a"] > 0.5).astype(int)
    dataset_path = _write_parquet(df, tmp_path / "reports" / "datasets" / "dataset_test.parquet")

    X = df.reset_index()[["feat_a", "feat_b"]]
    y = df.reset_index()["up_5d"]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    artifacts = {
        "logistic_cls_5": {
            "model": model,
            "features": ["feat_a", "feat_b"],
            "target": "up_5d",
            "task": "classification",
            "horizon": 5,
        }
    }

    report = analyze_feature_pruning(dataset_path, artifacts, [5])
    assert "retained_features" in report
    assert "dropped_features" in report
    assert "importance_stats" in report
    assert "correlation_clusters" in report
    assert Path(report["output_path"]).exists()


def test_model_comparison_logic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    eval_payload = {
        "metrics": {
            "5d": {
                "walk_forward": {
                    "classification": [
                        {"sharpe": 1.2, "hit_rate": 0.55},
                        {"sharpe": 1.0, "hit_rate": 0.5},
                    ],
                    "regression": [
                        {"sharpe": 0.8, "hit_rate": 0.52},
                    ],
                }
            }
        }
    }
    eval_path = tmp_path / "reports" / "eval" / "model_eval_test.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(eval_payload))

    portfolio_payload = {"metrics": {"sharpe_ratio": 1.1, "max_drawdown": -0.2, "turnover": 0.3}}
    portfolio_path = tmp_path / "reports" / "portfolio" / "portfolio_metrics.json"
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_path.write_text(json.dumps(portfolio_payload))

    report = compare_models(eval_path, portfolio_path)
    comparison = report["comparison"]["5d"]
    assert comparison["best_single_model"]["model"] == "classification"
    assert comparison["ensemble"]["sharpe"] == pytest.approx(1.1)


def test_param_sensitivity_determinism(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    base_config = {"alpha": {"enabled": True}}
    param_grid = {"alpha.weight": [0.1, 0.2], "portfolio.top_k": [5, 10]}

    def run_pipeline(cfg: dict) -> dict:
        alpha_weight = cfg["alpha"]["weight"]
        top_k = cfg["portfolio"]["top_k"]
        return {"metrics": {"sharpe": float(alpha_weight + top_k / 100)}}

    df1 = run_param_sensitivity(base_config, param_grid, run_pipeline)
    df2 = run_param_sensitivity(base_config, param_grid, run_pipeline)
    pd.testing.assert_frame_equal(df1, df2)


def test_regime_segmentation_correctness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    dates = pd.date_range("2023-01-01", periods=4, freq="D", tz="UTC")
    returns = pd.Series([0.01, -0.02, 0.03, -0.01], index=dates)
    regimes = pd.Series(
        ["LOW_VOL_TREND", "LOW_VOL_TREND", "HIGH_VOL_CHOP", "HIGH_VOL_CHOP"], index=dates
    )
    report = analyze_regime_performance(returns, regimes)
    assert report["per_regime"]["LOW_VOL_TREND"]["hit_rate"] == pytest.approx(0.5)
    assert Path(report["output_path"]).exists()


def test_portfolio_rationale_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    date = pd.Timestamp("2023-01-05", tz="UTC")
    portfolio_df = pd.DataFrame(
        {
            "date": [date, date],
            "ticker": ["AAA", "BBB"],
            "weight": [0.6, 0.0],
            "ensemble_score": [75.0, 60.0],
        }
    )
    contrib_df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "feature": ["feat_a", "feat_b", "feat_a"],
            "contribution": [0.4, -0.1, 0.2],
        }
    )
    regimes = pd.Series(["LOW_VOL_TREND"], index=[date])
    report = build_portfolio_rationale(portfolio_df, contrib_df, regimes)
    assert "positions" in report
    assert report["positions"][0]["ticker"] == "AAA"


def test_run_comparison_correctness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_a = "20240101_000000"
    run_b = "20240102_000000"

    for run_id, top_k in [(run_a, 5), (run_b, 10)]:
        run_dir = tmp_path / "reports" / "run" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        config_path.write_text(f"portfolio:\n  top_k: {top_k}\n")
        weights_path = run_dir / f"portfolio_weights_{run_id}.csv"
        pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "AAA": [0.6],
                "BBB": [0.4],
            }
        ).to_csv(weights_path, index=False)
        manifest = {
            "paths": {
                "config": str(config_path),
                "portfolio_weights": str(weights_path),
            }
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        feature_report = {
            "retained_features": {"5d": ["feat_a", "feat_b"]}
        }
        feature_path = tmp_path / "reports" / "analysis" / f"feature_pruning_{run_id}.json"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_path.write_text(json.dumps(feature_report))

        model_report = {
            "comparison": {
                "5d": {
                    "best_single_model": {"model": "classification"},
                    "redundant_models": [],
                    "ensemble": {"sharpe": 1.0, "max_drawdown": -0.2, "turnover": 0.3},
                }
            }
        }
        model_path = tmp_path / "reports" / "analysis" / f"model_comparison_{run_id}.json"
        model_path.write_text(json.dumps(model_report))

    report = compare_runs(run_a, run_b)
    assert report["config_delta"]["changed"]["portfolio"]["changed"]["top_k"]["to"] == 10
    assert report["portfolio_overlap"] == pytest.approx(1.0)
