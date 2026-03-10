"""Dataset IO helpers for parquet/csv compatibility."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from spectraquant.core.time import normalize_time_index


DATASET_DIR = Path("reports/datasets")
RUN_REPORTS_DIR = Path("reports/run")


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    if {"date", "ticker"}.issubset(df.columns):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index(["date", "ticker"])
    df = normalize_time_index(df, context=f"load dataset {dataset_path}")
    return df


def _candidate_dataset_paths() -> Iterable[Path]:
    yield from sorted(DATASET_DIR.glob("dataset_*.parquet"))
    yield from sorted(DATASET_DIR.glob("dataset_*.csv"))


def latest_dataset_path() -> Path:
    manifest_path = latest_dataset_path_from_manifest()
    if manifest_path and manifest_path.exists():
        return manifest_path
    candidates = list(_candidate_dataset_paths())
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("No dataset files found in reports/datasets")


def dataset_path_from_manifest(manifest_path: str | Path) -> Path | None:
    path = Path(manifest_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    dataset_info = payload.get("dataset", {}) if isinstance(payload, dict) else {}
    dataset_path = dataset_info.get("path")
    if not dataset_path:
        return None
    return Path(dataset_path)


def latest_dataset_path_from_manifest() -> Path | None:
    if not RUN_REPORTS_DIR.exists():
        return None
    candidates = sorted(RUN_REPORTS_DIR.glob("*/manifest.json"))
    if not candidates:
        return None
    return dataset_path_from_manifest(candidates[-1])


__all__ = ["load_dataset", "latest_dataset_path", "latest_dataset_path_from_manifest"]
