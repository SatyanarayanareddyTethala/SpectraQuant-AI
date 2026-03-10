"""Model training for multi-method ML pipeline."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from spectraquant.eval.walkforward import walk_forward_evaluate
from spectraquant.dataset.io import load_dataset

import importlib.util

LGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
if LGBM_AVAILABLE:
    from lightgbm import LGBMClassifier, LGBMRegressor
else:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


MODELS_DIR = Path("models")
EVAL_REPORTS_DIR = Path("reports/eval")


def _numeric_features(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = [c for c in df.columns if c.startswith("fwd_ret_") or c.startswith("up_")]
    excluded.append("regime")
    return [c for c in numeric_cols if c not in excluded]


def _split_xy(df: pd.DataFrame, feature_cols: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y


def _train_classifiers(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    models: dict[str, Any] = {}
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X, y)
    models["logistic"] = logreg

    if LGBM_AVAILABLE:
        lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    else:
        lgbm = GradientBoostingClassifier(random_state=42)
    lgbm.fit(X, y)
    models["primary"] = lgbm
    return models


def _train_regressors(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    models: dict[str, Any] = {}
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    models["ridge"] = ridge

    if LGBM_AVAILABLE:
        lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    else:
        lgbm = GradientBoostingRegressor(random_state=42)
    lgbm.fit(X, y)
    models["primary"] = lgbm
    return models


def train_models(dataset_path: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    df = load_dataset(dataset_path)

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Dataset must be indexed by (date, ticker)")

    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])

    feature_cols = _numeric_features(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for training")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"run_id": run_id, "models": {}, "features": feature_cols}

    horizons = sorted({int(col.split("_")[-1][:-1]) for col in df.columns if col.startswith("fwd_ret_")})
    for horizon in horizons:
        cls_target = f"up_{horizon}d"
        reg_target = f"fwd_ret_{horizon}d"

        if cls_target not in df.columns or reg_target not in df.columns:
            continue

        df_h = df.dropna(subset=[cls_target, reg_target] + feature_cols)
        X, y_cls = _split_xy(df_h, feature_cols, cls_target)
        _, y_reg = _split_xy(df_h, feature_cols, reg_target)

        X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2, shuffle=False)
        cls_models = _train_classifiers(X_train, y_train)
        cls_pred = cls_models["primary"].predict_proba(X_test)[:, 1]
        cls_auc = roc_auc_score(y_test, cls_pred) if len(y_test.unique()) > 1 else np.nan
        cls_logloss = log_loss(y_test, cls_pred, labels=[0, 1]) if len(y_test.unique()) > 1 else np.nan

        reg_models = _train_regressors(X, y_reg)

        eval_metrics = walk_forward_evaluate(
            df_h,
            feature_cols=feature_cols,
            cls_target=cls_target,
            reg_target=reg_target,
            horizon=horizon,
        )

        for model_name, model in cls_models.items():
            artifact = {
                "model": model,
                "features": feature_cols,
                "target": cls_target,
                "task": "classification",
                "horizon": horizon,
            }
            path = MODELS_DIR / f"{model_name}_cls_{horizon}.pkl"
            joblib.dump(artifact, path)
            results["models"][f"{model_name}_cls_{horizon}"] = str(path)

        for model_name, model in reg_models.items():
            artifact = {
                "model": model,
                "features": feature_cols,
                "target": reg_target,
                "task": "regression",
                "horizon": horizon,
            }
            path = MODELS_DIR / f"{model_name}_reg_{horizon}.pkl"
            joblib.dump(artifact, path)
            results["models"][f"{model_name}_reg_{horizon}"] = str(path)

        results.setdefault("metrics", {})[f"{horizon}d"] = {
            "auc": float(cls_auc) if cls_auc == cls_auc else None,
            "logloss": float(cls_logloss) if cls_logloss == cls_logloss else None,
            "walk_forward": eval_metrics,
        }

    eval_path = EVAL_REPORTS_DIR / f"model_eval_{run_id}.json"
    eval_path.write_text(json.dumps(results, indent=2, default=str))
    results["eval_path"] = str(eval_path)
    return results


__all__ = ["train_models"]
