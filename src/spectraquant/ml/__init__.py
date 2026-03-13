"""SpectraQuant ML – predictive analytics layer.

Provides Random Forest and XGBoost classification, walk-forward validation,
feature importance, optional SARIMAX forecasting, and an ensemble signal layer
that integrates with the existing SpectraQuant signal pipeline.

Public surface
--------------
features   – ML feature engineering (OHLCV + technical indicators + sentiment)
targets    – Supervised target creation
models     – RF / XGBoost model factories
walk_forward – Time-series walk-forward validation (no shuffle)
importance – Feature importance extraction
forecast   – Optional SARIMAX directional forecasting
ensemble   – Ensemble probability and signal generation
pipeline   – End-to-end ML pipeline entry-point
"""
from __future__ import annotations

from spectraquant.ml.features import add_features
from spectraquant.ml.targets import add_target
from spectraquant.ml.models import get_random_forest, get_xgboost, classification_metrics
from spectraquant.ml.walk_forward import walk_forward_validate, FoldResult
from spectraquant.ml.importance import get_feature_importance
from spectraquant.ml.ensemble import ensemble_probability, ensemble_to_signal
from spectraquant.ml.pipeline import run_ml_pipeline

__all__ = [
    "add_features",
    "add_target",
    "get_random_forest",
    "get_xgboost",
    "classification_metrics",
    "walk_forward_validate",
    "FoldResult",
    "get_feature_importance",
    "ensemble_probability",
    "ensemble_to_signal",
    "run_ml_pipeline",
]
