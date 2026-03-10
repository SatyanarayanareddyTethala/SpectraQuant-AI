"""Run-to-run comparison utilities."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

RUN_REPORTS_DIR = Path("reports/run")


def _load_manifest(run_id: str) -> dict:
    manifest_path = RUN_REPORTS_DIR / run_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest missing for {run_id}: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _load_config(manifest: dict) -> dict:
    paths = manifest.get("paths", {}) if isinstance(manifest, dict) else {}
    config_path = paths.get("config") or manifest.get("config")
    if not config_path:
        raise FileNotFoundError("Config path missing in manifest")
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config snapshot missing: {config_file}")
    return yaml.safe_load(config_file.read_text())


def _diff_dicts(a: Any, b: Any) -> dict:
    if isinstance(a, dict) and isinstance(b, dict):
        added = {k: b[k] for k in b.keys() - a.keys()}
        removed = {k: a[k] for k in a.keys() - b.keys()}
        changed = {}
        for key in a.keys() & b.keys():
            diff = _diff_dicts(a[key], b[key])
            if diff:
                changed[key] = diff
        return {"added": added, "removed": removed, "changed": changed}
    if isinstance(a, list) and isinstance(b, list):
        return {
            "added": [item for item in b if item not in a],
            "removed": [item for item in a if item not in b],
        }
    if a != b:
        return {"from": a, "to": b}
    return {}


def _load_feature_report(run_id: str) -> dict:
    path = Path("reports/analysis") / f"feature_pruning_{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Feature pruning report missing for {run_id}: {path}")
    return json.loads(path.read_text())


def _load_model_report(run_id: str) -> dict:
    path = Path("reports/analysis") / f"model_comparison_{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Model comparison report missing for {run_id}: {path}")
    return json.loads(path.read_text())


def _load_portfolio_weights(manifest: dict) -> pd.DataFrame:
    paths = manifest.get("paths", {}) if isinstance(manifest, dict) else {}
    weights_path = paths.get("portfolio_weights") or paths.get("weights")
    if not weights_path:
        raise FileNotFoundError("Portfolio weights path missing in manifest")
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Portfolio weights missing: {path}")
    return pd.read_csv(path)


def _portfolio_overlap(weights_a: pd.DataFrame, weights_b: pd.DataFrame) -> float:
    if weights_a.empty or weights_b.empty:
        return 0.0
    weights_a = weights_a.set_index("date")
    weights_b = weights_b.set_index("date")
    latest_a = weights_a.tail(1)
    latest_b = weights_b.tail(1)
    common_cols = sorted(set(latest_a.columns) & set(latest_b.columns))
    if not common_cols:
        return 0.0
    a_vals = latest_a[common_cols].iloc[0].fillna(0.0)
    b_vals = latest_b[common_cols].iloc[0].fillna(0.0)
    overlap = np.minimum(a_vals, b_vals).sum()
    return float(overlap)


def compare_runs(run_id_a: str, run_id_b: str) -> dict:
    manifest_a = _load_manifest(run_id_a)
    manifest_b = _load_manifest(run_id_b)

    config_a = _load_config(manifest_a)
    config_b = _load_config(manifest_b)

    feature_a = _load_feature_report(run_id_a)
    feature_b = _load_feature_report(run_id_b)

    model_a = _load_model_report(run_id_a)
    model_b = _load_model_report(run_id_b)

    weights_a = _load_portfolio_weights(manifest_a)
    weights_b = _load_portfolio_weights(manifest_b)

    config_delta = _diff_dicts(config_a, config_b)

    feature_diff: dict[str, Any] = {}
    retained_a = feature_a.get("retained_features", {})
    retained_b = feature_b.get("retained_features", {})
    for horizon, features_a in retained_a.items():
        features_b = retained_b.get(horizon, [])
        feature_diff[horizon] = {
            "added": sorted(set(features_b) - set(features_a)),
            "removed": sorted(set(features_a) - set(features_b)),
            "overlap": sorted(set(features_a) & set(features_b)),
        }

    model_changes: dict[str, Any] = {}
    comparison_a = model_a.get("comparison", {})
    comparison_b = model_b.get("comparison", {})
    for horizon, entry_a in comparison_a.items():
        entry_b = comparison_b.get(horizon, {})
        model_changes[horizon] = {
            "best_single_model_a": entry_a.get("best_single_model", {}).get("model"),
            "best_single_model_b": entry_b.get("best_single_model", {}).get("model"),
            "redundant_models_a": entry_a.get("redundant_models", []),
            "redundant_models_b": entry_b.get("redundant_models", []),
        }

    performance_delta: dict[str, Any] = {}
    for horizon, entry_a in comparison_a.items():
        entry_b = comparison_b.get(horizon, {})
        perf_a = entry_a.get("ensemble", {})
        perf_b = entry_b.get("ensemble", {})
        performance_delta[horizon] = {
            "sharpe_delta": float(perf_b.get("sharpe", 0.0)) - float(perf_a.get("sharpe", 0.0)),
            "max_drawdown_delta": float(perf_b.get("max_drawdown", 0.0)) - float(perf_a.get("max_drawdown", 0.0)),
            "turnover_delta": float(perf_b.get("turnover", 0.0)) - float(perf_a.get("turnover", 0.0)),
        }

    overlap = _portfolio_overlap(weights_a, weights_b)

    run_id = f"{run_id_a}_{run_id_b}"
    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_a": run_id_a,
        "run_b": run_id_b,
        "config_delta": config_delta,
        "feature_set_diff": feature_diff,
        "model_changes": model_changes,
        "performance_delta": performance_delta,
        "portfolio_overlap": overlap,
    }

    output_dir = Path("reports/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_diff_{run_id_a}_{run_id_b}.json"
    output_path.write_text(json.dumps(report, indent=2, default=str))
    report["output_path"] = str(output_path)
    return report


__all__ = ["compare_runs"]
