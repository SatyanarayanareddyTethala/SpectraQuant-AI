from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]


def find_latest_file(glob_pattern: str) -> Path | None:
    try:
        pattern_path = Path(glob_pattern)
        if pattern_path.is_absolute():
            candidates = [p for p in pattern_path.parent.glob(pattern_path.name) if p.is_file()]
        else:
            candidates = [p for p in ROOT_DIR.glob(glob_pattern) if p.is_file()]
    except Exception:  # noqa: BLE001
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_latest_predictions() -> pd.DataFrame | None:
    latest = find_latest_file("reports/predictions/predictions_*.csv")
    if not latest:
        return None
    try:
        return pd.read_csv(latest)
    except Exception:  # noqa: BLE001
        return None


def load_training_metadata() -> dict | None:
    metadata_path = ROOT_DIR / "models" / "training_metadata.json"
    target = metadata_path if metadata_path.exists() else find_latest_file("models/training_metadata*.json")
    if not target:
        return None
    try:
        return json.loads(target.read_text())
    except Exception:  # noqa: BLE001
        return None


def load_latest_prices(ticker: str) -> pd.DataFrame | None:
    prices_dir = ROOT_DIR / "data" / "prices"
    if not ticker:
        return None
    parquet_path = prices_dir / f"{ticker}.parquet"
    csv_path = prices_dir / f"{ticker}.csv"
    try:
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        return None
    return None
