"""Feature pruning analysis utilities."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from spectraquant.dataset.io import load_dataset

def _numeric_features(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = [c for c in df.columns if c.startswith("fwd_ret_") or c.startswith("up_")]
    excluded.append("regime")
    excluded.extend([c for c in df.columns if c in {"index", "level_0"}])
    return [c for c in numeric_cols if c not in excluded]


def _load_model_artifacts(model_artifacts: dict) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for name, artifact in model_artifacts.items():
        if isinstance(artifact, (str, Path)):
            data = joblib.load(artifact)
            data = {**data, "artifact_name": name, "path": str(artifact)}
            loaded.append(data)
        elif isinstance(artifact, dict):
            loaded.append({**artifact, "artifact_name": name})
        else:
            raise TypeError(f"Unsupported artifact type for {name}: {type(artifact)}")
    return loaded


def _importance_from_model(model: Any, feature_names: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=float)
        if values.ndim > 1:
            values = values[0]
        values = np.abs(values)
    else:
        values = np.zeros(len(feature_names))
    if values.shape[0] != len(feature_names):
        values = np.resize(values, len(feature_names))
    return {name: float(val) for name, val in zip(feature_names, values)}


def _permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: str,
) -> dict[str, dict[str, float]]:
    if X.empty:
        return {}
    if len(X) < 15:
        return {}
    if task == "classification":
        scoring = "roc_auc" if len(pd.Series(y).unique()) > 1 else "neg_log_loss"
    else:
        scoring = "neg_mean_squared_error"

    splitter = TimeSeriesSplit(n_splits=min(3, max(2, len(X) // 10)))
    fold_stats: dict[str, list[float]] = defaultdict(list)
    for train_idx, test_idx in splitter.split(X):
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        if X_test.empty:
            continue
        result = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=5, random_state=42)
        for name, mean_imp in zip(X_test.columns, result.importances_mean):
            fold_stats[name].append(float(mean_imp))

    return {
        name: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for name, vals in fold_stats.items()
    }


def _build_correlation_clusters(feature_corr: pd.DataFrame, threshold: float = 0.9) -> list[list[str]]:
    if feature_corr.empty:
        return []
    features = list(feature_corr.columns)
    visited = set()
    clusters: list[list[str]] = []
    for feature in features:
        if feature in visited:
            continue
        stack = [feature]
        cluster = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            correlated = feature_corr.index[feature_corr[current].abs() > threshold].tolist()
            for neighbor in correlated:
                if neighbor not in visited:
                    stack.append(neighbor)
        if len(cluster) > 1:
            clusters.append(sorted(cluster))
    return clusters


def analyze_feature_pruning(
    dataset_path: Path,
    model_artifacts: dict,
    horizons: list[int],
) -> dict:
    dataset_path = Path(dataset_path)
    df = load_dataset(dataset_path)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    feature_cols = _numeric_features(df)
    if not feature_cols:
        raise ValueError("No numeric features available for pruning analysis")

    artifacts = _load_model_artifacts(model_artifacts)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    retained_features: dict[str, list[str]] = {}
    dropped_features: dict[str, dict[str, str]] = {}
    importance_stats: dict[str, dict[str, Any]] = {}
    correlation_clusters: dict[str, list[list[str]]] = {}

    for horizon in horizons:
        horizon_key = f"{horizon}d"
        horizon_models = [a for a in artifacts if int(a.get("horizon", -1)) == horizon]
        if not horizon_models:
            continue

        horizon_imp: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        perm_imp: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

        for artifact in horizon_models:
            model = artifact.get("model")
            task = artifact.get("task") or "classification"
            target = artifact.get("target")
            if target not in df.columns:
                continue
            horizon_df = df.dropna(subset=feature_cols + [target])
            X = horizon_df[feature_cols]
            y = horizon_df[target]

            base_imp = _importance_from_model(model, feature_cols)
            for feature, value in base_imp.items():
                horizon_imp[feature][artifact["artifact_name"]].append(value)

            perm_stats = _permutation_importance(model, X, y, task=task)
            if perm_stats:
                perm_imp[artifact["artifact_name"]] = perm_stats

        importance_summary: dict[str, dict[str, float]] = {}
        for feature, model_vals in horizon_imp.items():
            flat_vals = [val for vals in model_vals.values() for val in vals]
            importance_summary[feature] = {
                "mean": float(np.mean(flat_vals)) if flat_vals else 0.0,
                "std": float(np.std(flat_vals)) if flat_vals else 0.0,
            }

        perm_summary: dict[str, dict[str, float]] = defaultdict(lambda: {"mean": 0.0, "std": 0.0})
        for model_name, stats in perm_imp.items():
            for feature, vals in stats.items():
                perm_summary[feature]["mean"] += vals.get("mean", 0.0)
                perm_summary[feature]["std"] = max(perm_summary[feature]["std"], vals.get("std", 0.0))
        if perm_imp:
            for feature in perm_summary:
                perm_summary[feature]["mean"] /= max(1, len(perm_imp))

        feature_corr = df[feature_cols].corr()
        clusters = _build_correlation_clusters(feature_corr)

        low_importance = set()
        means = [stats["mean"] for stats in importance_summary.values()]
        if means:
            threshold = np.quantile(means, 0.2)
            for feature, stats in importance_summary.items():
                if stats["mean"] <= threshold:
                    low_importance.add(feature)

        dropped: dict[str, str] = {}
        for feature in low_importance:
            dropped[feature] = "consistently_low_importance"

        aggregated_importance = {f: importance_summary.get(f, {}).get("mean", 0.0) for f in feature_cols}
        for cluster in clusters:
            sorted_cluster = sorted(cluster, key=lambda f: aggregated_importance.get(f, 0.0), reverse=True)
            if not sorted_cluster:
                continue
            leader = sorted_cluster[0]
            for peer in sorted_cluster[1:]:
                if aggregated_importance.get(peer, 0.0) <= aggregated_importance.get(leader, 0.0):
                    dropped.setdefault(peer, f"collinear_with_{leader}")

        retained = [f for f in feature_cols if f not in dropped]

        retained_features[horizon_key] = retained
        dropped_features[horizon_key] = dropped
        importance_stats[horizon_key] = {
            "model_importance": importance_summary,
            "permutation_importance": perm_summary,
        }
        correlation_clusters[horizon_key] = clusters

    report = {
        "run_id": run_id,
        "dataset": str(dataset_path),
        "retained_features": retained_features,
        "dropped_features": dropped_features,
        "importance_stats": importance_stats,
        "correlation_clusters": correlation_clusters,
    }

    output_dir = Path("reports/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"feature_pruning_{run_id}.json"
    output_path.write_text(json.dumps(report, indent=2))
    report["output_path"] = str(output_path)
    return report


__all__ = ["analyze_feature_pruning"]
