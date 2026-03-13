"""End-to-end ML pipeline for SpectraQuant.

This module is the single entry-point for ML training and prediction.  It
wires together features → targets → model training → walk-forward evaluation
→ feature importance → optional SARIMAX → ensemble → signal output.

The pipeline writes its outputs to the repo's existing ``reports/`` hierarchy
and returns a structured result dict so the CLI and tests can inspect it.

Integration with the existing signal flow
------------------------------------------
``run_ml_pipeline`` returns an ``MLPipelineResult`` whose ``signals``
DataFrame has a ``signal_score`` column in ``[-1, 0, 1]``.  This aligns with
the existing ``AgentOutput.signal_score`` convention so ML signals can be
merged with rule-based signals in ``spectraquant.models.ensemble``.

Datetime / index safety
-----------------------
* The input OHLCV ``df`` must have a :class:`pandas.DatetimeIndex`.
* UTC normalisation is applied defensively; the pipeline raises on malformed
  indices rather than silently proceeding with wrong dates.
* ``future_return`` / ``target`` creation uses ``shift(-horizon)`` which
  preserves the existing row order without any shuffle.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spectraquant.ml.features import ML_FEATURE_COLS, add_features
from spectraquant.ml.targets import add_target
from spectraquant.ml.models import HAS_XGB, get_random_forest, get_xgboost
from spectraquant.ml.walk_forward import FoldResult, walk_forward_validate
from spectraquant.ml.importance import get_feature_importance, save_feature_importance
from spectraquant.ml.ensemble import ensemble_probability, ensemble_to_signal

logger = logging.getLogger(__name__)

_SIGNALS_DIR = Path("reports/ml/signals")
_EVAL_DIR = Path("reports/ml/eval")


@dataclass
class MLPipelineResult:
    """Structured output from :func:`run_ml_pipeline`."""

    signals: pd.DataFrame
    """DataFrame with columns ``signal_score`` (int: -1/0/1) and ``ensemble_prob``
    indexed by the same DatetimeIndex as the input ``df``."""

    rf_fold_results: list[FoldResult] = field(default_factory=list)
    """Walk-forward fold outcomes for the Random Forest model."""

    xgb_fold_results: list[FoldResult] = field(default_factory=list)
    """Walk-forward fold outcomes for the XGBoost model (empty if XGB unavailable)."""

    rf_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Feature importances from the final Random Forest model."""

    xgb_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Feature importances from the final XGBoost model (empty if XGB unavailable)."""

    metadata: dict = field(default_factory=dict)
    """Run metadata (timestamps, config used, row counts, etc.)."""


def _validate_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a UTC-normalised DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "run_ml_pipeline: the input DataFrame must have a DatetimeIndex.  "
            "Use pd.to_datetime() + set_index() before calling this function."
        )
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    else:
        df = df.copy()
        df.index = df.index.tz_convert("UTC")
    # Guard against the 1970-epoch regression that occurs when timestamps
    # come in as Unix-millisecond integers incorrectly parsed as nanoseconds.
    if df.index.min().year < 2000:
        raise ValueError(
            "run_ml_pipeline: DatetimeIndex contains dates before 2000-01-01.  "
            "Check that timestamps are not accidentally in milliseconds."
        )
    return df


def _load_ml_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the ``ml`` config section with safe defaults."""
    ml_cfg: dict[str, Any] = config.get("ml", {})
    return {
        "horizon": int(ml_cfg.get("horizon", 1)),
        "train_size": int(ml_cfg.get("train_size", 252)),
        "test_size": int(ml_cfg.get("test_size", 21)),
        "step_size": int(ml_cfg.get("step_size", 21)),
        "threshold": float(ml_cfg.get("threshold", 0.55)),
        "use_xgboost": bool(ml_cfg.get("use_xgboost", True)),
        "use_sarimax": bool(ml_cfg.get("use_sarimax", False)),
        "w_rf": float(ml_cfg.get("weights", {}).get("rf", 0.4)),
        "w_xgb": float(ml_cfg.get("weights", {}).get("xgb", 0.4)),
        "w_ts": float(ml_cfg.get("weights", {}).get("ts", 0.2)),
        "feature_columns": ml_cfg.get("feature_columns", ML_FEATURE_COLS),
    }


def run_ml_pipeline(
    df: pd.DataFrame,
    config: "dict[str, Any] | None" = None,
) -> MLPipelineResult:
    """Run the full ML pipeline on prepared OHLCV (+ optional sentiment) data.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a DatetimeIndex.  Must contain at minimum
        ``close`` and ``volume`` columns.  An optional ``sentiment_score``
        column is forwarded to the feature engineering step.
    config:
        Full SpectraQuant config dict.  The pipeline reads the ``ml``
        sub-section; if ``None`` sensible defaults are used.

    Returns
    -------
    MLPipelineResult
        Structured output including signals, fold results, and importances.

    Raises
    ------
    ValueError
        On malformed index, missing columns, or insufficient rows.
    """
    if config is None:
        config = {}
    cfg = _load_ml_config(config)

    # ------------------------------------------------------------------ #
    # 1. Validate and normalise the datetime index                         #
    # ------------------------------------------------------------------ #
    df = _validate_datetime_index(df)
    logger.info("ML pipeline: %d rows, index [%s → %s]", len(df), df.index[0], df.index[-1])

    # ------------------------------------------------------------------ #
    # 2. Feature engineering                                               #
    # ------------------------------------------------------------------ #
    feature_cols: list[str] = cfg["feature_columns"]
    df_feat = add_features(df)

    # ------------------------------------------------------------------ #
    # 3. Target creation                                                   #
    # ------------------------------------------------------------------ #
    horizon: int = cfg["horizon"]
    df_full = add_target(df_feat, horizon=horizon)

    # Keep only rows where both features and target are clean
    df_clean = df_full.dropna(subset=feature_cols + ["target"]).copy()
    df_clean["target"] = df_clean["target"].astype(int)

    if len(df_clean) < cfg["train_size"] + cfg["test_size"]:
        raise ValueError(
            f"run_ml_pipeline: only {len(df_clean)} clean rows available after "
            f"feature/target computation; need at least "
            f"{cfg['train_size'] + cfg['test_size']} rows."
        )

    # ------------------------------------------------------------------ #
    # 4. Walk-forward validation – Random Forest                          #
    # ------------------------------------------------------------------ #
    logger.info("ML pipeline: running Random Forest walk-forward validation …")
    rf_folds = walk_forward_validate(
        df_clean,
        feature_cols=feature_cols,
        model_factory=get_random_forest,
        train_size=cfg["train_size"],
        test_size=cfg["test_size"],
        step_size=cfg["step_size"],
    )
    logger.info("ML pipeline: Random Forest – %d folds completed.", len(rf_folds))

    # ------------------------------------------------------------------ #
    # 5. Walk-forward validation – XGBoost (optional)                     #
    # ------------------------------------------------------------------ #
    xgb_folds: list[FoldResult] = []
    if cfg["use_xgboost"] and HAS_XGB:
        logger.info("ML pipeline: running XGBoost walk-forward validation …")
        xgb_folds = walk_forward_validate(
            df_clean,
            feature_cols=feature_cols,
            model_factory=get_xgboost,
            train_size=cfg["train_size"],
            test_size=cfg["test_size"],
            step_size=cfg["step_size"],
        )
        logger.info("ML pipeline: XGBoost – %d folds completed.", len(xgb_folds))
    elif cfg["use_xgboost"] and not HAS_XGB:
        logger.warning("ML pipeline: use_xgboost=true but xgboost is not installed; skipping.")

    # ------------------------------------------------------------------ #
    # 6. Fit final models on full available data for production signals    #
    # ------------------------------------------------------------------ #
    X_all = df_clean[feature_cols]
    y_all = df_clean["target"]

    rf_final = get_random_forest()
    rf_final.fit(X_all, y_all)
    rf_probs: np.ndarray = rf_final.predict_proba(X_all)[:, 1]

    xgb_final = None
    xgb_probs: np.ndarray = np.full(len(df_clean), 0.5)
    xgb_importance_df = pd.DataFrame()

    if cfg["use_xgboost"] and HAS_XGB:
        xgb_final = get_xgboost()
        xgb_final.fit(X_all, y_all)
        xgb_probs = xgb_final.predict_proba(X_all)[:, 1]

    # ------------------------------------------------------------------ #
    # 7. Feature importances                                               #
    # ------------------------------------------------------------------ #
    rf_importance_df = get_feature_importance(rf_final, feature_cols)
    save_feature_importance(rf_importance_df, label="rf")
    logger.info("ML pipeline: Random Forest feature importances saved.")

    if xgb_final is not None:
        xgb_importance_df = get_feature_importance(xgb_final, feature_cols)
        save_feature_importance(xgb_importance_df, label="xgb")
        logger.info("ML pipeline: XGBoost feature importances saved.")

    # ------------------------------------------------------------------ #
    # 8. Optional SARIMAX directional signal                               #
    # ------------------------------------------------------------------ #
    ts_signal: "np.ndarray | None" = None
    if cfg["use_sarimax"]:
        try:
            from spectraquant.ml.forecast import sarimax_direction_signal, HAS_STATSMODELS

            if HAS_STATSMODELS:
                close_series = df_clean["close"] if "close" in df_clean.columns else None
                if close_series is not None:
                    ts_signal = sarimax_direction_signal(
                        close_series.pct_change().dropna(),
                        steps=len(df_clean),
                    )
                    # Ensure same length as clean df
                    if len(ts_signal) != len(df_clean):
                        ts_signal = None
            else:
                logger.warning("ML pipeline: use_sarimax=true but statsmodels is not installed; skipping.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ML pipeline: SARIMAX failed (%s); continuing without it.", exc)
            ts_signal = None

    # ------------------------------------------------------------------ #
    # 9. Ensemble                                                          #
    # ------------------------------------------------------------------ #
    ensemble_prob = ensemble_probability(
        rf_prob=rf_probs,
        xgb_prob=xgb_probs,
        ts_signal=ts_signal,
        w_rf=cfg["w_rf"],
        w_xgb=cfg["w_xgb"],
        w_ts=cfg["w_ts"],
    )

    signal_series = ensemble_to_signal(
        ensemble_prob,
        threshold=cfg["threshold"],
        index=df_clean.index,
    )

    signals_df = pd.DataFrame(
        {
            "ensemble_prob": ensemble_prob,
            "signal_score": signal_series,
        },
        index=df_clean.index,
    )

    # ------------------------------------------------------------------ #
    # 10. Persist signals                                                  #
    # ------------------------------------------------------------------ #
    _SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    signals_path = _SIGNALS_DIR / f"ml_signals_{ts_tag}.csv"
    signals_df.to_csv(signals_path)
    logger.info("ML pipeline: signals written to %s", signals_path)

    # ------------------------------------------------------------------ #
    # 11. Persist fold evaluation summary                                  #
    # ------------------------------------------------------------------ #
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if rf_folds:
        fold_rows = [
            {
                "fold": f.fold,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                **f.metrics,
            }
            for f in rf_folds
        ]
        eval_path = _EVAL_DIR / f"rf_folds_{ts_tag}.csv"
        pd.DataFrame(fold_rows).to_csv(eval_path, index=False)

    if xgb_folds:
        fold_rows = [
            {
                "fold": f.fold,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                **f.metrics,
            }
            for f in xgb_folds
        ]
        eval_path = _EVAL_DIR / f"xgb_folds_{ts_tag}.csv"
        pd.DataFrame(fold_rows).to_csv(eval_path, index=False)

    metadata = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(df),
        "clean_rows": len(df_clean),
        "horizon": horizon,
        "feature_columns": feature_cols,
        "use_xgboost": cfg["use_xgboost"] and HAS_XGB,
        "use_sarimax": ts_signal is not None,
        "rf_folds": len(rf_folds),
        "xgb_folds": len(xgb_folds),
        "signals_path": str(signals_path),
    }

    return MLPipelineResult(
        signals=signals_df,
        rf_fold_results=rf_folds,
        xgb_fold_results=xgb_folds,
        rf_importance=rf_importance_df,
        xgb_importance=xgb_importance_df,
        metadata=metadata,
    )


__all__ = ["MLPipelineResult", "run_ml_pipeline"]
