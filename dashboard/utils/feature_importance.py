from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _first_matching_line(lines: Iterable[str], keys: Iterable[str]) -> str | None:
    for line in lines:
        for key in keys:
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
            if line.startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
    return None


def _resolve_model_path(model_path: str) -> Path | None:
    candidate = Path(model_path)
    if candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    promoted = repo_root / "models" / "promoted" / "model.txt"
    if promoted.exists():
        return promoted
    latest = repo_root / "models" / "latest" / "model.txt"
    if latest.exists():
        return latest
    return None


def parse_feature_importance(model_path: str, method: str = "gain") -> pd.DataFrame:
    resolved = _resolve_model_path(model_path)
    if resolved is None or not resolved.exists():
        return pd.DataFrame(columns=["feature", "importance", "importance_pct", "group"])

    lines = [line.strip() for line in resolved.read_text().splitlines() if line.strip()]
    feature_names = _first_matching_line(lines, ("feature_names", "feature_name"))
    importance_line = _first_matching_line(lines, ("feature_importances", "feature_importance"))

    if not feature_names or not importance_line:
        return pd.DataFrame(columns=["feature", "importance", "importance_pct", "group"])

    features = feature_names.split()
    importance_values = [float(value) for value in importance_line.split()]
    size = min(len(features), len(importance_values))
    features = features[:size]
    importance_values = importance_values[:size]

    df = pd.DataFrame({"feature": features, "importance": importance_values})
    total = float(df["importance"].sum())
    if total > 0:
        df["importance_pct"] = df["importance"] / total * 100
    else:
        df["importance_pct"] = 0.0

    df["group"] = df["feature"].apply(lambda name: name.split("_", 1)[0] if "_" in name else "other")
    df.attrs["method"] = method
    df.attrs["model_path"] = str(resolved)
    return df
