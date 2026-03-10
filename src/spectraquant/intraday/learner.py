"""Intraday online learner utilities."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier, early_stopping

from spectraquant.core.time import ensure_datetime_column


MODEL_DIR = Path("models/intraday")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelState:
    model: Booster | None
    metadata: dict


class IntradayLearner:
    """Incremental intraday learner with rolling calibration."""

    def __init__(self) -> None:
        self.state = self._load_or_init()

    def _load_or_init(self) -> ModelState:
        model_path = MODEL_DIR / "lgbm_model.txt"
        meta_path = MODEL_DIR / "metadata.json"
        if model_path.exists() and meta_path.exists():
            model = Booster(model_file=str(model_path))
            metadata = json.loads(meta_path.read_text())
            return ModelState(model=model, metadata=metadata)
        metadata = {
            "last_trained": None,
            "calibration": [],
            "model_version": "intraday_lgbm_v1",
            "factor_set_version": "intraday_default",
        }
        return ModelState(model=None, metadata=metadata)

    def _save_state(self) -> None:
        model_path = MODEL_DIR / "lgbm_model.txt"
        meta_path = MODEL_DIR / "metadata.json"
        if self.state.model is not None:
            self.state.model.save_model(str(model_path))
        meta_path.write_text(json.dumps(self.state.metadata, indent=2))

    def _build_labels(self, df: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
        df = ensure_datetime_column(df, "date")
        if "close" not in df.columns:
            raise ValueError("Intraday features must include close for label creation")
        df = df.sort_values("date")
        future = df["close"].shift(-horizon_bars)
        df["label"] = (future > df["close"]).astype(int)
        return df.dropna(subset=["label"])

    def update_from_features(self, feature_dir: Path, horizon_bars: int = 1) -> None:
        frames: List[pd.DataFrame] = []
        for path in feature_dir.glob("*.parquet"):
            if path.name == "latest_snapshot.csv":
                continue
            df = pd.read_parquet(path)
            if df.empty:
                continue
            frames.append(df)
        if not frames:
            return
        features = pd.concat(frames, ignore_index=True)
        features = self._build_labels(features, horizon_bars=horizon_bars)
        feature_cols = [
            c for c in features.columns if c not in {"date", "ticker", "label"}
        ]
        X = features[feature_cols].select_dtypes(include=[np.number])
        y = features["label"].astype(int)
        if X.empty or y.empty:
            return
        model = LGBMClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
        )
        model.fit(
            X,
            y,
            eval_set=[(X, y)],
            eval_metric="auc",
            callbacks=[early_stopping(10, verbose=False)],
        )
        self.state.model = model.booster_
        self.state.metadata["feature_cols"] = feature_cols
        self.state.metadata["factor_set_version"] = hashlib.sha256(
            ",".join(sorted(feature_cols)).encode("utf-8")
        ).hexdigest()
        self.state.metadata["last_trained"] = pd.Timestamp.utcnow().isoformat()
        self._save_state()

    def predict(self, snapshot: pd.DataFrame) -> pd.Series:
        snapshot = ensure_datetime_column(snapshot, "date")
        feature_cols = self.state.metadata.get("feature_cols")
        if not feature_cols:
            return pd.Series(50.0, index=snapshot.index)
        X = snapshot[feature_cols].select_dtypes(include=[np.number])
        if X.empty:
            return pd.Series(50.0, index=snapshot.index)
        model = self.state.model
        if model is None:
            return pd.Series(50.0, index=snapshot.index)
        raw = model.predict(X.to_numpy())
        return pd.Series(raw, index=snapshot.index)
