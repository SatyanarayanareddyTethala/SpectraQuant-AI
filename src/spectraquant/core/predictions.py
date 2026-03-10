"""Prediction utilities for ticker-specific expected returns.

Extended to support news-aware blending and explainable output fields:
    reason, event_type, analysis_model, expected_move_pct, target_price,
    stop_price, confidence, risk_score, news_refs.

When ``news_context_by_ticker`` is supplied the expected move is blended
between the technical estimate and a news-derived move.  The blend weight
is derived from the news magnitude, source rank, and recency.  When no
news context is present the system falls back to the pure technical signal
(``w_news ≈ 0``).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from spectraquant.core.ranking import normalize_scores

TRADING_DAYS = 252
ANNUAL_RETURN_MIN = -0.40
ANNUAL_RETURN_MAX = 0.40
ANNUAL_RETURN_TARGET = 0.30

# ---------------------------------------------------------------------------
# News-blend helpers
# ---------------------------------------------------------------------------

def _news_blend_weight(news_ctx: Dict[str, Any]) -> float:
    """Compute the news blend weight from a news context dict.

    w_news = clamp(magnitude * source_rank * recency, 0, 1)

    If any value is missing the corresponding factor defaults to 0 (no news)
    or 1.0 (no penalty).
    """
    magnitude = float(news_ctx.get("magnitude", 0.0) or 0.0)
    source_rank = float(news_ctx.get("source_rank", 1.0) or 1.0)
    recency = float(news_ctx.get("recency", 1.0) or 1.0)
    return float(np.clip(magnitude * source_rank * recency, 0.0, 1.0))


def _news_expected_move(news_ctx: Dict[str, Any]) -> float:
    """Derive the news-implied expected move from a news context dict.

    Uses ``magnitude`` and ``sentiment`` from the context.  Positive
    sentiment contributes a positive move; negative a negative move.
    """
    magnitude = float(news_ctx.get("magnitude", 0.0) or 0.0)
    sentiment = str(news_ctx.get("sentiment", "neutral") or "neutral").lower()
    direction = 1.0 if sentiment == "positive" else (-1.0 if sentiment == "negative" else 0.0)
    return direction * magnitude * 0.05  # 5% max move at magnitude=1.0


import logging as _log

def _derive_analysis_model(news_ctx: Optional[Dict[str, Any]], regime: str) -> str:
    """Select analysis model label deterministically from context."""
    try:
        from spectraquant.intelligence.model_selector import ModelSelector
        selector = ModelSelector()
        return selector.select(news_ctx, regime=regime).value
    except (ImportError, ValueError, TypeError) as exc:
        _log.getLogger(__name__).warning(
            "_derive_analysis_model: falling back to MOMENTUM due to %s: %s",
            type(exc).__name__, exc,
        )
        return "MOMENTUM"


def _build_reason(
    ticker: str,
    news_ctx: Optional[Dict[str, Any]],
    analysis_model: str,
    expected_move: float,
) -> str:
    """Build a human-readable reason string for a prediction."""
    if not news_ctx:
        return f"Technical signal: {analysis_model.lower()} model"

    event_type = str(news_ctx.get("event_type", "") or "").replace("_", " ")
    sentiment = str(news_ctx.get("sentiment", "") or "")
    headlines = news_ctx.get("top_headlines", "")

    parts = [f"Analysis model: {analysis_model}"]
    if event_type:
        parts.append(f"Event: {event_type} ({sentiment})")
    if headlines:
        parts.append(f"News: {headlines[:120]}")
    direction = "upside" if expected_move > 0 else "downside"
    parts.append(f"Expected {direction}: {expected_move * 100:.1f}%")
    return "; ".join(parts)


def _safe_float(value: float | int | None, default: float = 0.0) -> float:
    try:
        if value is None or not np.isfinite(value):
            return default
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _scale_factor_score(score: float | int | None) -> float:
    score_value = _safe_float(score, 0.0)
    return float(np.tanh(score_value) * 0.02)


def compute_expected_return(
    metrics: Mapping[str, float],
    factor_score: float | int | None,
    horizon_days: float,
) -> tuple[float, float, float, float]:
    mean_return = _safe_float(metrics.get("mean_return"))
    volatility = _safe_float(metrics.get("volatility"))
    momentum_daily = _safe_float(metrics.get("momentum_daily"))
    rsi = _safe_float(metrics.get("rsi"), 50.0)
    factor_signal = _scale_factor_score(factor_score)
    rsi_adjust = (rsi - 50.0) / 1000.0

    expected_daily = (
        (0.45 * mean_return)
        + (0.35 * momentum_daily)
        + (0.2 * factor_signal)
        - (0.1 * volatility)
        + rsi_adjust
    )
    # Derive daily cap consistent with annual bounds
    daily_cap = float(np.expm1(np.log1p(ANNUAL_RETURN_MAX) / TRADING_DAYS))
    # Smooth tanh scaling: 75% of cap as target keeps linear behaviour for small signals
    # while gently compressing large ones without hard saturation
    daily_target = 0.75 * daily_cap
    expected_daily_scaled = float(np.tanh(expected_daily / daily_target) * daily_target)
    # Compute horizon return via compounding (horizon-first)
    h_days = max(float(horizon_days), 1e-9)
    expected_horizon = (1 + expected_daily_scaled) ** h_days - 1
    # Derive annual return from horizon return (horizon-aware annualization)
    daily_from_h = (1 + expected_horizon) ** (1 / h_days) - 1
    expected_annual = (1 + daily_from_h) ** TRADING_DAYS - 1
    # Safety-net clip (should rarely activate after tanh scaling)
    expected_annual = float(np.clip(expected_annual, ANNUAL_RETURN_MIN, ANNUAL_RETURN_MAX))

    score_raw = (
        (momentum_daily * 1000.0)
        + (mean_return * 800.0)
        - (volatility * 500.0)
        + (factor_signal * 5000.0)
        + (rsi - 50.0)
    )
    return float(expected_horizon), float(expected_daily_scaled), float(score_raw), float(expected_annual)


def build_prediction_frame(
    tickers: Iterable[str],
    metrics_by_ticker: Dict[str, Mapping[str, float]],
    factor_scores: Dict[str, float],
    horizon: str,
    horizon_days: float,
    model_version: str,
    factor_set_version: str,
    regime: str,
    prediction_dates: Dict[str, pd.Timestamp],
    news_context_by_ticker: Optional[Dict[str, Dict[str, Any]]] = None,
    price_by_ticker: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Build prediction frame with optional news-aware blending.

    Parameters
    ----------
    tickers, metrics_by_ticker, factor_scores, horizon, horizon_days,
    model_version, factor_set_version, regime, prediction_dates :
        Same as before (backward-compatible).
    news_context_by_ticker : dict, optional
        Per-ticker news context dicts.  Each dict should contain:
        ``magnitude``, ``sentiment``, ``source_rank``, ``recency``,
        ``event_type``, ``top_headlines``, ``competitor_shock``,
        ``news_refs``.  Missing keys are handled gracefully.
    price_by_ticker : dict, optional
        Last closing price per ticker, used to compute ``target_price``
        and ``stop_price``.
    """
    frame = pd.DataFrame({"ticker": list(tickers)})
    if frame.empty:
        return frame

    expected_horizon_returns: list[float] = []
    expected_annual_returns: list[float] = []
    expected_daily_returns: list[float] = []
    raw_scores: list[float] = []

    # New explainability columns
    reasons: list[str] = []
    event_types: list[str] = []
    analysis_models: list[str] = []
    expected_move_pcts: list[float] = []
    target_prices: list[float] = []
    stop_prices: list[float] = []
    confidences: list[float] = []
    risk_scores_col: list[float] = []
    news_refs_col: list[Any] = []

    news_ctx_map = news_context_by_ticker or {}
    price_map = price_by_ticker or {}

    for ticker in frame["ticker"]:
        metrics = metrics_by_ticker.get(ticker) or {
            "mean_return": 0.0,
            "volatility": 0.0,
            "momentum_daily": 0.0,
            "rsi": 50.0,
        }
        factor_score = factor_scores.get(ticker, 0.0)
        expected_horizon, expected_daily, score_raw, expected_annual = compute_expected_return(
            metrics, factor_score, horizon_days
        )

        # --- News-aware blending ---
        news_ctx = news_ctx_map.get(ticker)
        if news_ctx:
            w_news = _news_blend_weight(news_ctx)
            news_move = _news_expected_move(news_ctx)
            # Blend: technical horizon return + news move
            blended_horizon = (1.0 - w_news) * expected_horizon + w_news * news_move
        else:
            w_news = 0.0
            blended_horizon = expected_horizon

        expected_horizon_returns.append(float(blended_horizon))
        expected_annual_returns.append(float(expected_annual))
        expected_daily_returns.append(float(expected_daily))
        raw_scores.append(float(score_raw))

        # --- Explainability fields ---
        analysis_model = _derive_analysis_model(news_ctx, regime)
        reason = _build_reason(ticker, news_ctx, analysis_model, blended_horizon)
        event_type = str((news_ctx or {}).get("event_type", "") or "")
        news_refs: List[str] = list((news_ctx or {}).get("news_refs", []) or [])

        # Target and stop prices
        last_price = price_map.get(ticker, 0.0) or 0.0
        atr_proxy = _safe_float((metrics or {}).get("volatility")) * last_price if last_price else 0.0
        if last_price > 0:
            target_price = last_price * (1.0 + blended_horizon)
            atr = atr_proxy if atr_proxy > 0 else last_price * 0.02
            stop_price = last_price - 1.5 * atr
        else:
            target_price = 0.0
            stop_price = 0.0

        # Confidence: blend of probability and news signal quality
        news_uncertainty = float((news_ctx or {}).get("uncertainty", 1.0) or 1.0)
        # Lower uncertainty means higher confidence
        base_confidence = 0.5 + (blended_horizon / 0.1) * 0.05 if abs(blended_horizon) > 0 else 0.5
        base_confidence = float(np.clip(base_confidence, 0.05, 0.95))
        if news_ctx:
            news_confidence = 1.0 - news_uncertainty * 0.5
            confidence = 0.5 * base_confidence + 0.5 * news_confidence
        else:
            confidence = base_confidence
        confidence = float(np.clip(confidence, 0.05, 0.95))

        # Risk score
        volatility = _safe_float((metrics or {}).get("volatility"))
        risk_score = float(np.clip(volatility * 20.0, 0.0, 1.0))

        reasons.append(reason)
        event_types.append(event_type)
        analysis_models.append(analysis_model)
        expected_move_pcts.append(round(blended_horizon * 100, 4))
        target_prices.append(round(target_price, 4))
        stop_prices.append(round(stop_price, 4))
        confidences.append(round(confidence, 4))
        risk_scores_col.append(round(risk_score, 4))
        news_refs_col.append(news_refs)

    frame["expected_return_annual"] = expected_annual_returns
    frame["expected_return_horizon"] = expected_horizon_returns
    frame["expected_return"] = expected_daily_returns
    frame["predicted_return"] = expected_horizon_returns
    frame["predicted_return_1d"] = expected_daily_returns
    frame["score"] = normalize_scores(pd.Series(raw_scores, index=frame.index))
    logits = (frame["score"] - 50) / 12
    probs = 1 / (1 + np.exp(-logits))
    probs = np.clip(probs, 0.05, 0.95)
    if float(np.std(probs)) < 1e-6:
        jitter = np.linspace(-0.002, 0.002, len(probs))
        probs = np.clip(probs + jitter, 0.05, 0.95)
    frame["probability"] = probs
    frame["model_version"] = model_version
    frame["factor_set_version"] = factor_set_version
    frame["horizon"] = horizon
    frame["regime"] = regime
    frame["date"] = frame["ticker"].map(prediction_dates)

    # Explainability columns
    frame["reason"] = reasons
    frame["event_type"] = event_types
    frame["analysis_model"] = analysis_models
    frame["expected_move_pct"] = expected_move_pcts
    frame["target_price"] = target_prices
    frame["stop_price"] = stop_prices
    frame["confidence"] = confidences
    frame["risk_score"] = risk_scores_col
    frame["news_refs"] = news_refs_col

    return frame
