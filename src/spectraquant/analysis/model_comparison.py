"""Model dominance and redundancy analysis."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_fold_metrics(metrics: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [m.get(key) for m in metrics if m.get(key) is not None]
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _extract_model_metrics(eval_payload: dict) -> dict[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, dict[str, float]]] = {}
    metrics = eval_payload.get("metrics") or {}
    for horizon_key, entry in metrics.items():
        walk_forward = entry.get("walk_forward") if isinstance(entry, dict) else None
        if not walk_forward:
            continue
        horizon_models: dict[str, dict[str, float]] = {}
        for model_label, fold_metrics in walk_forward.items():
            if not isinstance(fold_metrics, list):
                continue
            sharpe_stats = _summarize_fold_metrics(fold_metrics, "sharpe")
            hit_stats = _summarize_fold_metrics(fold_metrics, "hit_rate")
            stability = 1 / (1 + sharpe_stats["std"])
            horizon_models[model_label] = {
                "sharpe": sharpe_stats["mean"],
                "hit_rate": hit_stats["mean"],
                "stability": float(stability),
            }
        results[horizon_key] = horizon_models
    return results


def _extract_portfolio_metrics(portfolio_payload: dict) -> dict[str, dict[str, float]]:
    if "metrics" in portfolio_payload and isinstance(portfolio_payload["metrics"], dict):
        return {"ensemble": portfolio_payload["metrics"]}
    return {"ensemble": portfolio_payload}


def compare_models(
    eval_results_path: Path,
    portfolio_results_path: Path,
) -> dict:
    eval_results_path = Path(eval_results_path)
    portfolio_results_path = Path(portfolio_results_path)

    eval_payload = _load_json(eval_results_path)
    portfolio_payload = _load_json(portfolio_results_path)

    model_metrics = _extract_model_metrics(eval_payload)
    portfolio_metrics = _extract_portfolio_metrics(portfolio_payload)

    comparison: dict[str, Any] = {}
    for horizon_key, models in model_metrics.items():
        horizon_report: dict[str, Any] = {"models": models}
        if not models:
            comparison[horizon_key] = horizon_report
            continue

        best_model = max(models.items(), key=lambda item: item[1].get("sharpe", 0.0))
        best_name, best_stats = best_model
        horizon_report["best_single_model"] = {"model": best_name, **best_stats}

        ensemble_stats = portfolio_metrics.get("ensemble", {})
        ensemble_sharpe = ensemble_stats.get("sharpe_ratio", ensemble_stats.get("sharpe", 0.0))
        ensemble_drawdown = ensemble_stats.get("max_drawdown", 0.0)

        horizon_report["ensemble"] = {
            "sharpe": float(ensemble_sharpe) if ensemble_sharpe is not None else 0.0,
            "max_drawdown": float(ensemble_drawdown) if ensemble_drawdown is not None else 0.0,
            "turnover": float(ensemble_stats.get("turnover", 0.0)),
        }

        horizon_report["ensemble_outperforms_all"] = bool(
            horizon_report["ensemble"]["sharpe"] >= best_stats.get("sharpe", 0.0)
        )

        redundant_models: list[str] = []
        best_sharpe = best_stats.get("sharpe", 0.0)
        best_stability = best_stats.get("stability", 0.0)
        for name, stats in models.items():
            if name == best_name:
                continue
            if stats.get("sharpe", 0.0) < best_sharpe * 0.95 and stats.get("stability", 0.0) < best_stability:
                redundant_models.append(name)
        horizon_report["redundant_models"] = redundant_models
        comparison[horizon_key] = horizon_report

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("reports/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_comparison_{run_id}.json"
    payload = {
        "run_id": run_id,
        "eval_results": str(eval_results_path),
        "portfolio_results": str(portfolio_results_path),
        "comparison": comparison,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    payload["output_path"] = str(output_path)
    return payload


__all__ = ["compare_models"]
