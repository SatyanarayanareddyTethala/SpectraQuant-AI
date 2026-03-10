"""Filesystem-based model registry for lineage tracking."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


REGISTRY_DIR = Path("models/registry")
PROD_POINTER = Path("models/PROD_LATEST.json")
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def _hash_iter(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _hash_schema(columns: Iterable[str]) -> str:
    return _hash_iter(sorted(columns))


def register_model(
    version: int,
    dataset: pd.DataFrame,
    tickers: Iterable[str],
    factor_set_hash: str,
    metrics: Dict[str, float],
    config_hash: str,
    dataset_hash: str,
    git_commit: str | None,
) -> Path:
    if "date" not in dataset.columns:
        raise ValueError("Dataset missing date column for model registry")
    dates = pd.to_datetime(dataset["date"], utc=True, errors="coerce")
    if dates.isna().any():
        raise ValueError("Dataset contains invalid dates for model registry")
    training_window = {
        "start": dates.min().isoformat(),
        "end": dates.max().isoformat(),
    }
    metadata = {
        "model_version": int(version),
        "training_date": pd.Timestamp.utcnow().isoformat(),
        "universe_hash": _hash_iter(sorted({str(t) for t in tickers})),
        "factor_set_hash": factor_set_hash,
        "feature_schema_hash": _hash_schema([c for c in dataset.columns if c not in {"label"}]),
        "training_window": training_window,
        "metrics": metrics,
    }
    metadata["config_hash"] = config_hash
    metadata["dataset_hash"] = dataset_hash
    metadata["git_commit"] = git_commit
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    path = REGISTRY_DIR / f"model_v{version}.json"
    path.write_text(json.dumps(metadata, indent=2))
    latest_path = REGISTRY_DIR / "latest.json"
    latest_path.write_text(json.dumps(metadata, indent=2))
    return path


def load_latest_model_metadata() -> Dict:
    latest_path = REGISTRY_DIR / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError("No model registry metadata found.")
    return json.loads(latest_path.read_text())


def list_models() -> list[Dict]:
    models = []
    for path in sorted(REGISTRY_DIR.glob("model_v*.json")):
        models.append(json.loads(path.read_text()))
    return models


def promote_model(version: int) -> Path:
    path = REGISTRY_DIR / f"model_v{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"Model version {version} not found.")
    PROD_POINTER.write_text(path.read_text())
    return PROD_POINTER


def load_prod_model_metadata() -> Dict:
    if not PROD_POINTER.exists():
        raise FileNotFoundError("No promoted PROD model found.")
    return json.loads(PROD_POINTER.read_text())
