"""Schema validators for pipeline artifacts."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from spectraquant.core.ranking import normalize_scores

SCHEMA_VERSIONS = {
    "predictions": 5,
    "signals": 1,
    "portfolio_returns": 1,
    "portfolio_weights": 1,
    "execution_trades": 1,
    "execution_fills": 1,
    "execution_costs": 1,
    "execution_pnl": 1,
    "explainability": 1,
}

SCHEMA_COLUMNS = {
    "predictions": [
        "ticker",
        "date",
        "horizon",
        "score",
        "probability",
        "expected_return_annual",
        "expected_return_horizon",
        "expected_return",
        "predicted_return",
        "target_price",
        "predicted_return_1d",
        "target_price_1d",
        "model_version",
        "factor_set_version",
        "regime",
        "schema_version",
        # New explainability columns (added in schema v5)
        "reason",
        "event_type",
        "analysis_model",
        "expected_move_pct",
        "stop_price",
        "confidence",
        "risk_score",
        "news_refs",
    ],
    "signals": ["ticker", "date", "horizon", "score", "signal", "rank", "regime", "schema_version"],
    "portfolio_returns": ["date", "return", "schema_version"],
    "portfolio_weights": ["date", "schema_version"],
    "execution_trades": ["date", "ticker", "side", "weight_change", "price", "schema_version"],
    "execution_fills": ["date", "ticker", "side", "weight_change", "fill_price", "schema_version"],
    "execution_costs": [
        "date",
        "ticker",
        "weight_change",
        "slippage_cost",
        "transaction_cost",
        "total_cost",
        "schema_version",
    ],
    "execution_pnl": ["date", "gross_pnl", "costs", "net_pnl", "schema_version"],
    "explainability": [
        "date",
        "ticker",
        "horizon",
        "factor_group",
        "contribution",
        "model_version",
        "factor_set_version",
        "regime",
        "schema_version",
    ],
}


def order_columns(df: pd.DataFrame, artifact: str) -> pd.DataFrame:
    if artifact not in SCHEMA_COLUMNS:
        raise ValueError(f"Unknown schema artifact: {artifact}")
    if artifact == "portfolio_weights":
        tickers = [c for c in df.columns if c not in {"date", "schema_version"}]
        ordered = ["date", *sorted(tickers), "schema_version"]
        return df.loc[:, ordered]
    ordered = SCHEMA_COLUMNS[artifact]
    return df.loc[:, ordered]


def schema_version_for(artifact: str) -> int:
    if artifact not in SCHEMA_VERSIONS:
        raise ValueError(f"Unknown schema artifact: {artifact}")
    return SCHEMA_VERSIONS[artifact]
from spectraquant.core.time import ensure_datetime_column

import logging as _logging

_schema_logger = _logging.getLogger(__name__)


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ticker",
        "date",
        "horizon",
        "score",
        "probability",
        "expected_return_annual",
        "expected_return_horizon",
        "expected_return",
        "predicted_return",
        "target_price",
        "predicted_return_1d",
        "target_price_1d",
        "model_version",
        "factor_set_version",
        "regime",
        "schema_version",
    }
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    df = ensure_datetime_column(df, "date")
    df["score"] = normalize_scores(df["score"])
    if df["score"].isna().any():
        raise ValueError("Predictions contain NaN scores")
    if ((df["score"] < 0) | (df["score"] > 100)).any():
        raise ValueError("Prediction scores out of bounds")
    if df["ticker"].isna().any():
        raise ValueError("Predictions contain missing tickers")
    if df["model_version"].isna().any() or df["factor_set_version"].isna().any():
        raise ValueError("Predictions missing model or factor set versions")
    _assert_return_columns_not_degenerate(df)
    return df


def _assert_return_columns_not_degenerate(df: pd.DataFrame) -> None:
    """Raise if expected_return_annual is constant while horizon/predicted returns vary."""
    if len(df) < 3:
        return
    annual = pd.to_numeric(df["expected_return_annual"], errors="coerce").dropna()
    if annual.empty or annual.nunique() > 1:
        return
    # expected_return_annual is constant – check if returns actually vary
    horizon_col = pd.to_numeric(df["expected_return_horizon"], errors="coerce").dropna()
    predicted_col = pd.to_numeric(df["predicted_return"], errors="coerce").dropna()
    returns_vary = (
        (len(horizon_col) >= 2 and horizon_col.nunique() > 1)
        or (len(predicted_col) >= 2 and predicted_col.nunique() > 1)
    )
    if returns_vary:
        constant_val = float(annual.iloc[0])
        tickers = df["ticker"].unique()[:3].tolist() if "ticker" in df.columns else []
        horizons = df["horizon"].unique()[:3].tolist() if "horizon" in df.columns else []
        raise ValueError(
            f"expected_return_annual is constant ({constant_val:.4f}) across all rows "
            f"but predicted/horizon returns vary – likely clipping saturation or wrong "
            f"annualization. Sample tickers={tickers}, horizons={horizons}. "
            "Check annualization formula and model prediction scale."
        )
    # If we reach here, annual is constant AND returns are also identical (degenerate model)
    _schema_logger.warning(
        "expected_return_annual is constant (%.4f) and returns are also identical – "
        "model may be producing degenerate predictions.",
        float(annual.iloc[0]),
    )


def validate_signals(df: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "score", "signal", "rank", "horizon", "regime", "schema_version"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Signals missing columns: {sorted(missing)}")
    df = ensure_datetime_column(df, "date")
    if df["signal"].isna().any():
        raise ValueError("Signals contain NaNs")
    allowed = {"BUY", "HOLD", "SELL"}
    if not df["signal"].astype(str).str.upper().isin(allowed).all():
        raise ValueError("Signals contain invalid values")
    df["score"] = normalize_scores(df["score"])
    if df["rank"].isna().any():
        raise ValueError("Signals missing ranks")
    return df


def validate_portfolio_results(results: Dict) -> None:
    returns = results.get("returns")
    metrics = results.get("metrics", {})
    if not isinstance(returns, (pd.Series, np.ndarray)):
        raise ValueError("Portfolio returns missing")
    series = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
    if series.isna().any():
        raise ValueError("Portfolio returns contain NaNs")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Portfolio returns not sorted by date")
    risk_score = metrics.get("risk_score")
    if risk_score is not None and not (0 <= float(risk_score) <= 100):
        raise ValueError("Risk score out of bounds")
