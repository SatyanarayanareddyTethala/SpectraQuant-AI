"""Ensemble scoring utilities."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from spectraquant.core.ranking import normalize_scores


SCORES_DIR = Path("reports/scores")


def _normalize_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True).mul(100)


def compute_ensemble_scores(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ensemble scores from prediction outputs."""
    if pred_df.empty:
        raise ValueError("pred_df is empty; cannot compute ensemble scores")

    df = pred_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])

    prob_cols = [c for c in df.columns if c.startswith("prob_up_")]
    ret_cols = [c for c in df.columns if c.startswith("pred_ret_")]

    ml_prob = df[prob_cols].mean(axis=1) if prob_cols else pd.Series(0.5, index=df.index)
    if ret_cols:
        ret_signal = df[ret_cols[-1]]
        ret_signal_name = ret_cols[-1]
    else:
        ret_signal_name = "__ret_signal__"
        ret_signal = pd.Series(0.0, index=df.index, name=ret_signal_name)

    signal_score = df["signal_score"] if "signal_score" in df.columns else pd.Series(50.0, index=df.index)
    signal_score_norm = normalize_scores(signal_score)
    ml_prob_norm = normalize_scores(ml_prob.mul(100))

    if "date" in df.columns:
        if ret_signal_name not in df.columns:
            df[ret_signal_name] = ret_signal
        ret_rank_norm = df.groupby("date")[ret_signal_name].transform(_normalize_rank)
    else:
        ret_rank_norm = _normalize_rank(ret_signal)

    weights = {
        "signal": 0.35,
        "ml": 0.45,
        "ret": 0.20,
    }

    if "regime" in df.columns:
        high_vol_chop = df["regime"] == "HIGH_VOL_CHOP"
        ret_weight = pd.Series(weights["ret"], index=df.index)
        ret_weight.loc[high_vol_chop] = weights["ret"] * 0.5
        total_weight = weights["signal"] + weights["ml"] + ret_weight
        signal_w = weights["signal"] / total_weight
        ml_w = weights["ml"] / total_weight
        ret_w = ret_weight / total_weight
        ensemble = (
            signal_score_norm * signal_w + ml_prob_norm * ml_w + ret_rank_norm * ret_w
        )
    else:
        ensemble = (
            weights["signal"] * signal_score_norm
            + weights["ml"] * ml_prob_norm
            + weights["ret"] * ret_rank_norm
        )

    df["signal_score_norm"] = signal_score_norm
    df["ml_prob_norm"] = ml_prob_norm
    df["ret_rank_norm"] = ret_rank_norm
    df["ensemble_score"] = ensemble
    return df


def write_ensemble_scores(pred_df: pd.DataFrame) -> Path:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    scored = compute_ensemble_scores(pred_df)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = SCORES_DIR / f"ensemble_scores_{run_id}.csv"
    scored.to_csv(path, index=False)
    return path


__all__ = ["compute_ensemble_scores", "write_ensemble_scores"]
