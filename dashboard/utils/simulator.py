from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from dashboard.utils.diagnostics import (
    HORIZON_OUT_OF_RANGE,
    MISSING_REQUIRED_ARTIFACT,
    NO_SIGNAL_AFTER_FILTER,
    SIGNAL_RETURN_MISALIGNMENT,
    Diagnostic,
    DiagnosticError,
    make_diagnostic,
)
from dashboard.utils.time_index import align_on_time_index, normalize_time_index


P10_Z = -1.2816
P50_Z = 0.0
P90_Z = 1.2816
TRADING_DAYS = 252


@dataclass
class SimulationResult:
    curve: pd.DataFrame
    expected_final_value: float
    expected_profit: float
    cagr: float
    max_drawdown_p90: float
    risk_label: str
    risk_note: str
    expected_return: float
    volatility_annualized: float
    volatility_level: str
    confidence_score: float
    confidence_label: str
    probability: float
    trend_direction: str
    transaction_cost_pct: float
    management_fee_pct: float
    holding_days: int
    total_contributed: float


def extract_expected_return(pred_row: pd.Series) -> Optional[float]:
    for col in ("expected_return_horizon", "expected_return", "predicted_return", "predicted_return_1d"):
        if col in pred_row.index:
            value = pd.to_numeric(pred_row[col], errors="coerce")
            if pd.notna(value):
                return float(value)
    return None


def extract_probability(pred_row: pd.Series) -> Optional[float]:
    for col in ("probability", "prob", "confidence"):
        if col in pred_row.index:
            value = pd.to_numeric(pred_row[col], errors="coerce")
            if pd.notna(value):
                return float(value)
    return None


def realized_volatility(close_series: pd.Series, window: int = 60) -> Optional[float]:
    clean = pd.to_numeric(close_series, errors="coerce").dropna()
    if clean.empty or len(clean) < max(window, 5):
        return None
    returns = clean.pct_change().dropna()
    if returns.empty:
        return None
    return float(returns.tail(window).std())


def _risk_label(vol_annualized: float) -> str:
    if vol_annualized < 0.15:
        return "Low"
    if vol_annualized < 0.30:
        return "Med"
    return "High"


def _volatility_level(vol_annualized: float) -> str:
    if vol_annualized < 0.15:
        return "Low"
    if vol_annualized < 0.30:
        return "Moderate"
    return "High"


def _trend_direction(close_series: pd.Series, window: int = 20) -> str:
    clean = pd.to_numeric(close_series, errors="coerce").dropna()
    if clean.empty or len(clean) < max(window, 2):
        return "Sideways"
    recent = clean.tail(window)
    if recent.iloc[-1] > recent.iloc[0]:
        return "Upward"
    if recent.iloc[-1] < recent.iloc[0]:
        return "Downward"
    return "Sideways"


def _confidence_score(
    expected_return: float,
    probability: Optional[float],
    vol_annualized: float,
) -> tuple[float, str]:
    signal_strength = min(abs(expected_return) / 0.15, 1.0)
    prob_term = 0.0
    if probability is not None and np.isfinite(probability):
        prob_term = min(abs(probability - 0.5) / 0.5, 1.0)
    vol_penalty = min(vol_annualized / 0.60, 1.0)
    score = (0.45 * signal_strength) + (0.35 * prob_term) + (0.20 * (1 - vol_penalty))
    score = float(np.clip(score, 0.0, 1.0) * 100)
    if score >= 70:
        label = "High"
    elif score >= 40:
        label = "Medium"
    else:
        label = "Low"
    return score, label


def _max_drawdown(series: Iterable[float]) -> float:
    values = np.array(series, dtype=float)
    if values.size == 0:
        return 0.0
    cummax = np.maximum.accumulate(values)
    drawdowns = (values / np.where(cummax == 0, 1, cummax)) - 1
    return float(drawdowns.min())


def simulate_equity_curve(
    pred_row: pd.Series,
    close_series: pd.Series,
    holding_days: int,
    investment_amount: float,
    horizon_days: float,
    transaction_cost_pct: float,
    management_fee_pct: float,
    sip_mode: str = "None",
    sip_amount: float = 0.0,
    volatility_window: int = 60,
) -> Optional[SimulationResult]:
    expected_return = extract_expected_return(pred_row)
    if expected_return is None or np.isnan(expected_return):
        return None
    if expected_return <= -0.999:
        return None

    daily_sigma = realized_volatility(close_series, volatility_window)
    risk_note = ""
    if daily_sigma is None or daily_sigma <= 0 or np.isnan(daily_sigma):
        daily_sigma = 0.15 / np.sqrt(TRADING_DAYS)
        risk_note = "Estimated risk (no history)"
    vol_annualized = float(daily_sigma * np.sqrt(TRADING_DAYS))
    volatility_level = _volatility_level(vol_annualized)
    probability = extract_probability(pred_row)
    trend_direction = _trend_direction(close_series)
    confidence_score, confidence_label = _confidence_score(expected_return, probability, vol_annualized)

    horizon_days = max(horizon_days, 1.0)
    holding_days = max(int(holding_days), 1)
    daily_mu = np.log1p(expected_return) / horizon_days

    time_index = np.arange(0, holding_days + 1)
    sqrt_time = np.sqrt(time_index)
    fee_drag = np.exp(-(management_fee_pct / TRADING_DAYS) * time_index)

    def build_curve(z_score: float) -> np.ndarray:
        growth = np.exp(daily_mu * time_index + z_score * daily_sigma * sqrt_time)
        curve = investment_amount * (1 - transaction_cost_pct) * growth * fee_drag
        if sip_mode == "Monthly SIP" and sip_amount > 0:
            for day in range(30, holding_days + 1, 30):
                lag = time_index - day
                lag = np.where(lag < 0, 0, lag)
                lag_sqrt = np.sqrt(lag)
                lag_growth = np.exp(daily_mu * lag + z_score * daily_sigma * lag_sqrt)
                lag_fee = np.exp(-(management_fee_pct / TRADING_DAYS) * lag)
                contrib = sip_amount * (1 - transaction_cost_pct) * lag_growth * lag_fee
                contrib = np.where(time_index >= day, contrib, 0.0)
                curve += contrib
        return curve

    p10 = build_curve(P10_Z)
    p50 = build_curve(P50_Z)
    p90 = build_curve(P90_Z)

    total_contributed = float(investment_amount)
    if sip_mode == "Monthly SIP" and sip_amount > 0:
        total_contributed += sip_amount * (holding_days // 30)

    expected_final_value = float(p50[-1])
    expected_profit = float(expected_final_value - total_contributed)
    years = holding_days / TRADING_DAYS
    if years > 0 and total_contributed > 0:
        cagr = float((expected_final_value / total_contributed) ** (1 / years) - 1)
    else:
        cagr = 0.0

    curve_df = pd.DataFrame(
        {"Day": time_index, "P10": p10, "P50": p50, "P90": p90}
    ).set_index("Day")

    return SimulationResult(
        curve=curve_df,
        expected_final_value=expected_final_value,
        expected_profit=expected_profit,
        cagr=cagr,
        max_drawdown_p90=_max_drawdown(p90),
        risk_label=_risk_label(vol_annualized),
        risk_note=risk_note,
        expected_return=float(expected_return),
        volatility_annualized=vol_annualized,
        volatility_level=volatility_level,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        probability=float(probability) if probability is not None else float("nan"),
        trend_direction=trend_direction,
        transaction_cost_pct=float(transaction_cost_pct),
        management_fee_pct=float(management_fee_pct),
        holding_days=holding_days,
        total_contributed=total_contributed,
    )


@dataclass
class PortfolioSimulationResult:
    curve: pd.DataFrame
    expected_return: float
    cagr: float
    max_drawdown: float
    volatility_annualized: float
    contributions: pd.Series
    confidence_score: float
    confidence_label: str


@dataclass
class PortfolioSimulationOutcome:
    weights: pd.Series
    diagnostics: list[Diagnostic]
    aligned_returns: pd.DataFrame
    aligned_prices: pd.DataFrame | None


def combine_portfolio_results(
    results: dict[str, SimulationResult],
    weights: dict[str, float],
) -> PortfolioSimulationResult:
    curve = None
    contributions = {}
    for ticker, result in results.items():
        weight = weights.get(ticker, 0.0)
        contrib = result.curve[["P10", "P50", "P90"]] * weight
        contributions[ticker] = float(weight)
        curve = contrib if curve is None else curve.add(contrib, fill_value=0.0)
    if curve is None:
        empty = pd.DataFrame(columns=["P10", "P50", "P90"])
        return PortfolioSimulationResult(
            curve=empty,
            expected_return=0.0,
            cagr=0.0,
            max_drawdown=0.0,
            volatility_annualized=0.0,
            contributions=pd.Series(dtype=float),
            confidence_score=0.0,
            confidence_label="Low",
        )
    final_value = float(curve["P50"].iloc[-1])
    initial_value = float(curve["P50"].iloc[0]) if not curve["P50"].empty else 0.0
    expected_return = (final_value / initial_value - 1) if initial_value else 0.0
    holding_days = len(curve.index) - 1
    years = holding_days / TRADING_DAYS if holding_days > 0 else 0.0
    cagr = (final_value / initial_value) ** (1 / years) - 1 if years and initial_value else 0.0
    max_drawdown = _max_drawdown(curve["P90"])
    vol_annualized = float(
        sum(result.volatility_annualized * weights.get(ticker, 0.0) for ticker, result in results.items())
    )
    confidence_score = float(
        sum(result.confidence_score * weights.get(ticker, 0.0) for ticker, result in results.items())
    )
    confidence_label = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 40 else "Low"
    return PortfolioSimulationResult(
        curve=curve,
        expected_return=expected_return,
        cagr=float(cagr),
        max_drawdown=float(max_drawdown),
        volatility_annualized=vol_annualized,
        contributions=pd.Series(contributions).sort_values(ascending=False),
        confidence_score=confidence_score,
        confidence_label=confidence_label,
    )


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [col for col in required if col not in df.columns]


def _prepare_time_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if date_col in df.columns:
        df = df.copy()
        df.index = df[date_col]
        df = df.drop(columns=[date_col])
    return normalize_time_index(df)


def simulate_portfolio_from_signals(
    signals_df: pd.DataFrame | None,
    returns_df: pd.DataFrame | None,
    prices_df: pd.DataFrame | None,
    *,
    horizon: str | None = None,
    alpha_threshold: float = 0.0,
    min_overlap: int = 5,
) -> PortfolioSimulationOutcome:
    diagnostics: list[Diagnostic] = []
    empty_series = pd.Series(dtype=float)
    empty_frame = pd.DataFrame()

    if signals_df is None or signals_df.empty:
        diagnostics.append(
            make_diagnostic(
                MISSING_REQUIRED_ARTIFACT,
                detected={"artifact": "signals", "rows": 0},
                suggestion="Generate signals via the pipeline before simulating.",
                message="Signals are missing.",
            )
        )
        return PortfolioSimulationOutcome(empty_series, diagnostics, empty_frame, None)

    if returns_df is None or returns_df.empty:
        diagnostics.append(
            make_diagnostic(
                MISSING_REQUIRED_ARTIFACT,
                detected={"artifact": "returns", "rows": 0},
                suggestion="Generate portfolio returns via the pipeline.",
                message="Portfolio returns are missing.",
            )
        )
        return PortfolioSimulationOutcome(empty_series, diagnostics, empty_frame, None)

    required_cols = _require_columns(signals_df, ["ticker", "signal"])
    if required_cols:
        diagnostics.append(
            make_diagnostic(
                MISSING_REQUIRED_ARTIFACT,
                detected={"missing_columns": required_cols},
                suggestion="Ensure signals include ticker, signal, and date columns.",
                message="Signals are missing required columns.",
            )
        )
        return PortfolioSimulationOutcome(empty_series, diagnostics, empty_frame, None)

    filtered = signals_df.copy()
    if horizon and "horizon" in filtered.columns:
        horizon_mask = filtered["horizon"].astype(str) == str(horizon)
        if not horizon_mask.any():
            diagnostics.append(
                make_diagnostic(
                    HORIZON_OUT_OF_RANGE,
                    detected={
                        "requested": horizon,
                        "available": sorted(filtered["horizon"].dropna().astype(str).unique().tolist()),
                    },
                    suggestion="Select a horizon that exists in the signal file.",
                    message="Requested horizon not found in signals.",
                )
            )
            return PortfolioSimulationOutcome(empty_series, diagnostics, empty_frame, None)
        filtered = filtered[horizon_mask]

    filtered = filtered[filtered["signal"].astype(str).str.upper() == "BUY"]
    if alpha_threshold > 0 and "score" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["score"], errors="coerce") >= alpha_threshold]

    if filtered.empty:
        diagnostics.append(
            make_diagnostic(
                NO_SIGNAL_AFTER_FILTER,
                detected={
                    "alpha_threshold": alpha_threshold,
                    "signals_rows": len(signals_df.index),
                    "buy_signals": 0,
                },
                suggestion="Lower the alpha threshold or verify BUY signals were generated.",
                message="No BUY signals remain after filtering.",
            )
        )
        return PortfolioSimulationOutcome(empty_series, diagnostics, empty_frame, None)

    try:
        returns_indexed = _prepare_time_index(returns_df, date_col="date")
    except DiagnosticError as exc:
        diagnostics.append(exc.diagnostic)
        return PortfolioSimulationOutcome(empty_series, diagnostics, empty_frame, None)

    try:
        signal_dates = filtered[["date"]].drop_duplicates() if "date" in filtered.columns else filtered.copy()
        signal_dates = _prepare_time_index(signal_dates, date_col="date")
    except DiagnosticError as exc:
        diagnostics.append(exc.diagnostic)
        return PortfolioSimulationOutcome(empty_series, diagnostics, empty_frame, None)

    aligned_signals, aligned_returns, align_diags = align_on_time_index(
        signal_dates, returns_indexed, how="inner", min_overlap=min_overlap
    )
    diagnostics.extend(align_diags)
    if aligned_signals.empty or aligned_returns.empty:
        diagnostics.append(
            make_diagnostic(
                SIGNAL_RETURN_MISALIGNMENT,
                detected={"signal_rows": len(signal_dates.index), "return_rows": len(returns_indexed.index)},
                suggestion="Ensure signals and returns overlap in time.",
                message="Signals and returns have no overlapping dates.",
            )
        )
        return PortfolioSimulationOutcome(empty_series, diagnostics, aligned_returns, None)

    overlap_dates = set(aligned_signals.index)
    filtered = filtered[filtered["date"].isin(overlap_dates)] if "date" in filtered.columns else filtered
    if filtered.empty:
        diagnostics.append(
            make_diagnostic(
                SIGNAL_RETURN_MISALIGNMENT,
                detected={"overlap_dates": len(overlap_dates)},
                suggestion="Align signals to the same date range as returns.",
                message="Date alignment removed all signals.",
            )
        )
        return PortfolioSimulationOutcome(empty_series, diagnostics, aligned_returns, None)

    aligned_prices = None
    if prices_df is not None and not prices_df.empty:
        try:
            prices_indexed = _prepare_time_index(prices_df, date_col="date")
            _, aligned_prices, price_diags = align_on_time_index(
                returns_indexed, prices_indexed, how="inner", min_overlap=min_overlap
            )
            diagnostics.extend(price_diags)
        except DiagnosticError as exc:
            diagnostics.append(exc.diagnostic)

    tickers = filtered["ticker"].astype(str)
    unique_tickers = sorted(set(tickers))
    if not unique_tickers:
        diagnostics.append(
            make_diagnostic(
                NO_SIGNAL_AFTER_FILTER,
                detected={"unique_tickers": 0},
                suggestion="Verify BUY signals include tickers.",
                message="No tickers available for weighting.",
            )
        )
        return PortfolioSimulationOutcome(empty_series, diagnostics, aligned_returns, aligned_prices)

    weight_value = 1.0 / len(unique_tickers)
    weights = pd.Series({ticker: weight_value for ticker in unique_tickers})
    if (weights == 0).all():
        diagnostics.append(
            make_diagnostic(
                NO_SIGNAL_AFTER_FILTER,
                detected={"weights_nonzero": int((weights != 0).sum())},
                suggestion="Ensure BUY signals pass filters and overlap with returns.",
                message="All portfolio weights resolved to zero.",
            )
        )

    return PortfolioSimulationOutcome(weights, diagnostics, aligned_returns, aligned_prices)
