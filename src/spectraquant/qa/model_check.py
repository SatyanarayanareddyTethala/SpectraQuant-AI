"""Model validation helpers."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


MODEL_PATTERN = re.compile(r"model_v(\d+)\.")


def check_model_artifacts(model_dir: str | Path) -> None:
    """Ensure trained model artifacts are present and non-empty."""

    model_path = Path(model_dir)
    assert model_path.exists(), f"Model directory {model_path} does not exist"
    artifacts = sorted(model_path.glob("model_v*"))
    assert artifacts, f"No model artifacts found in {model_path}"

    versions = []
    for artifact in artifacts:
        assert artifact.is_file(), f"Artifact {artifact} is not a file"
        assert artifact.stat().st_size > 0, f"Artifact {artifact} is empty"
        match = MODEL_PATTERN.search(artifact.name)
        assert match, f"Artifact {artifact.name} does not match version pattern"
        versions.append(int(match.group(1)))

    logger.info(
        "Model artifacts validated: %s (versions=%s)",
        [a.name for a in artifacts],
        sorted(set(versions)),
    )


def sanity_predict(model: Any, X_sample) -> None:
    """Run a lightweight prediction to verify outputs are valid probabilities."""

    assert hasattr(model, "predict") or hasattr(model, "predict_proba"), "Model lacks predict methods"
    X_arr = np.asarray(X_sample)
    assert X_arr.ndim == 2 and len(X_arr) > 0, "X_sample must be a non-empty 2D array"

    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(X_arr)
        probs = np.asarray(preds)
        if probs.ndim == 2 and probs.shape[1] > 1:
            probs = probs[:, 1]
    else:
        preds = model.predict(X_arr)
        probs = np.asarray(preds)

    assert probs.shape[0] == len(X_arr), "Prediction length mismatch"
    assert np.isfinite(probs).all(), "Predictions contain non-finite values"
    assert (probs >= 0).all() and (probs <= 1).all(), "Predictions are outside [0, 1]"

    logger.info("Sanity prediction completed: %s samples", len(probs))


def check_retrain_gating(metadata: dict, cfg: dict) -> None:
    """Ensure retraining respects configured intervals and versioning."""

    from spectraquant.mlops.auto_retrain import should_retrain  # local import to avoid cycles

    interval_days = int(cfg.get("mlops", {}).get("retrain_interval_days", 0))
    last_trained = metadata.get("last_trained")
    version = metadata.get("model_version")

    assert last_trained is not None, "training_metadata last_trained is missing"
    assert version is not None, "training_metadata model_version is missing"

    retrain_due = should_retrain(last_trained, interval_days)
    if retrain_due:
        logger.info(
            "Retrain due based on interval (last trained: %s, interval: %s days)",
            last_trained,
            interval_days,
        )
    else:
        logger.info(
            "Retrain not due (last trained: %s, interval: %s days, version: %s)",
            last_trained,
            interval_days,
            version,
        )
    assert not retrain_due, "Retraining should not proceed when interval has not elapsed during safety checks"
