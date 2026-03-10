"""Automated retraining utilities for SpectraQuant-AI."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
METADATA_PATH = MODELS_DIR / "training_metadata.json"
DEFAULT_INTERVAL_DAYS = 7
DEFAULT_MIN_IMPROVEMENT = 0.01
DEFAULT_DRIFT_THRESHOLD = 0.6
DEFAULT_DRIFT_RECENT_FRACTION = 0.1


def should_retrain(last_trained_date: str | None, retrain_interval_days: int) -> bool:
    """Return True when enough time has elapsed since the last training run."""

    if retrain_interval_days <= 0:
        logger.warning(
            "Retrain interval is non-positive (%s); skipping retraining decision.",
            retrain_interval_days,
        )
        return False

    if not last_trained_date:
        return True

    try:
        last_date = datetime.strptime(last_trained_date, "%Y-%m-%d").date()
    except ValueError:
        logger.warning("Invalid last_trained_date '%s'; retraining by default.", last_trained_date)
        return True

    today = datetime.now(timezone.utc).date()
    return (today - last_date) >= timedelta(days=retrain_interval_days)


def _load_training_metadata() -> dict:
    if METADATA_PATH.exists():
        try:
            with METADATA_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as exc:  # noqa: BLE001 - need to handle malformed metadata
            logger.warning("Failed to parse training metadata %s: %s", METADATA_PATH, exc)
    return {
        "last_trained": None,
        "best_metric": None,
        "model_version": 0,
    }


def _save_training_metadata(metadata: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Updated training metadata at %s", METADATA_PATH)


def _time_based_split(df: pd.DataFrame, validation_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df

    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        df = df.copy()
        df.index = pd.RangeIndex(start=0, stop=len(df))

    split_idx = max(int(len(df) * (1 - validation_fraction)), 1)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    if val_df.empty and not train_df.empty:
        val_df = train_df.iloc[-1:]
        train_df = train_df.iloc[:-1]
    return train_df, val_df


def _compute_validation_metric(train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    if val_df.empty or train_df.empty:
        logger.warning("Insufficient data for validation; returning 0.0 metric.")
        return 0.0

    label_col = None
    for candidate in ("label", "target", "y"):
        if candidate in val_df.columns:
            label_col = candidate
            break

    if label_col is None:
        logger.warning("No label column found for validation; returning 0.0 metric.")
        return 0.0

    try:
        train_mean = pd.to_numeric(train_df[label_col], errors="coerce").dropna().mean()
        val_labels = pd.to_numeric(val_df[label_col], errors="coerce").dropna()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to compute validation metric: %s", exc)
        return 0.0

    if val_labels.empty:
        logger.warning("Validation labels empty after coercion; returning 0.0 metric.")
        return 0.0

    mae = (val_labels - train_mean).abs().mean()
    metric = 1.0 / (1.0 + mae)
    return float(metric)


def _persist_model_artifact(version: int, metadata: dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODELS_DIR / f"model_v{version}.json"
    content = {"version": version, "trained_at": datetime.now(timezone.utc).isoformat(), "metadata": metadata}
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
    logger.info("Saved model artifact to %s", artifact_path)


def _compute_drift_score(dataset: pd.DataFrame, recent_fraction: float) -> float:
    if dataset.empty:
        return 0.0
    numeric = dataset.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
    if numeric.empty:
        return 0.0
    recent_rows = max(5, int(len(numeric) * recent_fraction))
    recent = numeric.tail(recent_rows)
    drift = (recent.mean() - numeric.mean()).abs() / (numeric.std() + 1e-9)
    max_drift = float(drift.max()) if not drift.empty else 0.0
    return max_drift if np.isfinite(max_drift) else 0.0


def _load_dataset_for_drift() -> pd.DataFrame:
    dataset_path = Path("dataset.csv")
    if not dataset_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(dataset_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load dataset for drift check: %s", exc)
        return pd.DataFrame()


def run_auto_retraining(config: dict) -> None:
    mlops_cfg = config.get("mlops", {}) if isinstance(config, dict) else {}
    retrain_interval_days = int(mlops_cfg.get("retrain_interval_days", DEFAULT_INTERVAL_DAYS))
    drift_threshold = float(mlops_cfg.get("drift_threshold", DEFAULT_DRIFT_THRESHOLD))
    recent_fraction = float(mlops_cfg.get("drift_recent_fraction", DEFAULT_DRIFT_RECENT_FRACTION))

    metadata = _load_training_metadata()
    last_trained = metadata.get("last_trained")

    interval_due = should_retrain(last_trained, retrain_interval_days)
    drift_due = False
    drift_score = None
    if drift_threshold > 0:
        dataset = _load_dataset_for_drift()
        drift_score = _compute_drift_score(dataset, recent_fraction)
        drift_due = drift_score >= drift_threshold
        logger.info(
            "Feature drift score %.4f (threshold %.4f). Drift retrain due: %s.",
            drift_score,
            drift_threshold,
            drift_due,
        )

    if not interval_due and not drift_due:
        logger.info(
            "Skipping retraining (not due yet). Last trained: %s. Drift score: %s",
            last_trained,
            drift_score if drift_score is not None else "n/a",
        )
        return

    from spectraquant.cli.main import cmd_train  # Imported lazily to avoid circular dependency

    logger.info("Auto-retraining due; invoking training pipeline.")
    cmd_train(config=config)
    updated = _load_training_metadata()
    logger.info(
        "Auto-retraining complete. Last trained: %s, model version: %s",
        updated.get("last_trained"),
        updated.get("model_version"),
    )
