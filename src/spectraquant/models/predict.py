"""Prediction utilities for trained ML models."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from spectraquant.dataset.io import load_dataset, latest_dataset_path

MODELS_DIR = Path("models")
PREDICTIONS_DIR = Path("reports/predictions")


def _latest_dataset() -> Path:
    return latest_dataset_path()


def _load_model(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def _select_model(pattern: str) -> Path:
    candidates = sorted(MODELS_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No models found for pattern {pattern}")
    for candidate in candidates:
        if candidate.name.startswith("primary"):
            return candidate
    return candidates[0]


def predict(dataset_path: str | Path | None = None) -> Path:
    dataset_path = Path(dataset_path) if dataset_path else _latest_dataset()
    df = load_dataset(dataset_path)
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Dataset must be indexed by (date, ticker)")

    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])

    horizons = sorted({int(col.split("_")[-1][:-1]) for col in df.columns if col.startswith("fwd_ret_")})

    output = df[["date", "ticker"]].copy()
    for horizon in horizons:
        cls_path = _select_model(f"*_cls_{horizon}.pkl")
        reg_path = _select_model(f"*_reg_{horizon}.pkl")

        cls_artifact = _load_model(cls_path)
        reg_artifact = _load_model(reg_path)

        features = [col for col in cls_artifact["features"] if col in df.columns]
        X = df[features]
        cls_model = cls_artifact["model"]
        reg_model = reg_artifact["model"]

        prob = cls_model.predict_proba(X)[:, 1]
        pred = reg_model.predict(X)

        output[f"prob_up_{horizon}d"] = prob
        output[f"pred_ret_{horizon}d"] = pred

    if "regime" in df.columns:
        output["regime"] = df["regime"].values

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"predictions_{run_id}.csv"
    output.to_csv(path, index=False)
    return path


__all__ = ["predict"]
