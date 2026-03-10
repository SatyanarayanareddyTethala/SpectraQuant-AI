"""Portfolio simulation utilities for SpectraQuant."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from spectraquant.core.policy import enforce_turnover_policy, enforce_weight_policy
from spectraquant.core.portfolio import apply_risk_constraints
from spectraquant.core.time import ensure_datetime_column
from spectraquant.data.normalize import normalize_price_columns, normalize_price_frame

logger = logging.getLogger(__name__)
DIAGNOSTICS_DIR = Path("reports/diagnostics")


def _index_summary(index: pd.Index) -> dict:
    tz = getattr(index, "tz", None)
    tz_name = str(tz) if tz is not None else None
    if index.empty:
        return {"count": 0, "start": None, "end": None, "tz": tz_name}
    return {
        "count": int(len(index)),
        "start": index.min().isoformat(),
        "end": index.max().isoformat(),
        "tz": tz_name,
    }


def _write_diagnostics(payload: dict, prefix: str) -> Path:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = DIAGNOSTICS_DIR / f"{prefix}_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _select_price_series(price_df: pd.DataFrame) -> pd.Series:
    normalized = normalize_price_columns(price_df)
    normalized = normalize_price_frame(normalized)
    for col in ("close", "adj_close", "price"):
        if col in normalized.columns:
            return normalized[col]

    for col in normalized.columns:
        if isinstance(col, str) and "close" in col:
            return normalized[col]

    numeric_cols = normalized.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        return normalized[numeric_cols[0]]
    raise ValueError("Price dataframe does not contain a numeric price column.")


def _infer_signal(value) -> bool:
    if isinstance(value, str):
        return value.strip().upper() == "BUY"
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, bool):
        return value
    return False


def _rebalance_dates(index: pd.Index, frequency: str) -> pd.DatetimeIndex:
    freq_map = {"weekly": "W-FRI", "monthly": "ME"}
    resample_freq = freq_map.get(frequency.lower(), "ME")
    return index.to_series().resample(resample_freq).last().dropna().index


def _compute_weights(
    tickers: list[str],
    price_data: Mapping[str, pd.DataFrame],
    as_of: pd.Timestamp,
    method: str,
    lookback: int = 20,
) -> pd.Series:
    if method.lower() == "volatility":
        vols = {}
        for ticker in tickers:
            data = price_data.get(ticker)
            if data is None:
                continue
            price = _select_price_series(data)
            history = price.loc[:as_of].tail(lookback + 1)
            returns = history.pct_change().dropna()
            if returns.empty:
                continue
            vol = returns.std()
            if vol and not np.isnan(vol) and vol > 0:
                vols[ticker] = vol
        if vols:
            inv_vol = {k: 1 / v for k, v in vols.items()}
            total = sum(inv_vol.values())
            return pd.Series({k: v / total for k, v in inv_vol.items()})
        logger.info("Volatility weighting selected but no valid vol estimates; using equal weights.")

    if tickers:
        weight = 1 / len(tickers)
        return pd.Series({t: weight for t in tickers})
    return pd.Series(dtype=float)


def _compute_score_weights(
    scores: pd.Series,
    price_data: Mapping[str, pd.DataFrame],
    as_of: pd.Timestamp,
    *,
    lookback: int = 20,
    min_volume: float | None = None,
) -> Tuple[pd.Series, list[dict]]:
    diagnostics: list[dict] = []
    if scores.empty:
        diagnostics.append({"date": as_of.isoformat(), "reason": "empty_scores"})
        return pd.Series(dtype=float), diagnostics

    filtered = scores.copy()
    if min_volume is not None:
        liquid = {}
        for ticker in scores.index:
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue
            normalized = normalize_price_columns(df)
            normalized = normalize_price_frame(normalized)
            if "volume" not in normalized.columns:
                continue
            vol_series = pd.to_numeric(normalized["volume"], errors="coerce").dropna()
            if vol_series.empty:
                continue
            latest_volume = vol_series.loc[:as_of].tail(1)
            if not latest_volume.empty and latest_volume.iloc[0] >= min_volume:
                liquid[ticker] = scores.loc[ticker]
        filtered = pd.Series(liquid)
        if filtered.empty:
            diagnostics.append({"date": as_of.isoformat(), "reason": "liquidity_filter"} )
            return pd.Series(dtype=float), diagnostics

    vols = {}
    for ticker in filtered.index:
        data = price_data.get(ticker)
        if data is None:
            continue
        price = _select_price_series(data)
        history = price.loc[:as_of].tail(lookback + 1)
        returns = history.pct_change().dropna()
        if returns.empty:
            continue
        vol = returns.std()
        vols[ticker] = vol if vol and not np.isnan(vol) else np.nan

    weights = {}
    for ticker, score in filtered.items():
        vol = vols.get(ticker, np.nan)
        denom = vol if vol and not np.isnan(vol) and vol > 0 else 1.0
        weights[ticker] = float(score) / denom

    weight_series = pd.Series(weights)
    weight_series = weight_series.replace([np.inf, -np.inf], np.nan).dropna()
    if weight_series.empty:
        diagnostics.append({"date": as_of.isoformat(), "reason": "invalid_weights"})
        return pd.Series(dtype=float), diagnostics

    weight_series = weight_series / weight_series.sum()
    return weight_series, diagnostics


def simulate_portfolio(
    signals_df: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    config: Dict,
) -> Dict:
    """Simulate a simple long-only portfolio.

    Parameters
    ----------
    signals_df : pd.DataFrame
        BUY signals indexed by date with tickers as columns or a column named
        ``ticker`` with signals.
    price_data : dict[str, pd.DataFrame]
        Mapping of ticker to price history dataframes.
    config : dict
        Configuration dictionary controlling weighting and rebalance frequency.
    """

    if signals_df.empty:
        raise ValueError("signals_df is empty; cannot simulate portfolio")

    portfolio_cfg = config.get("portfolio", {}) if isinstance(config, dict) else {}
    weighting = (portfolio_cfg.get("weighting") or "equal").lower()
    rebalance = (portfolio_cfg.get("rebalance") or "monthly").lower()
    top_k = int(portfolio_cfg.get("top_k", 20))
    min_volume = portfolio_cfg.get("liquidity_min_volume")
    research_mode = bool(config.get("research_mode", False))
    signals_point_in_time = bool(config.get("signals_point_in_time", False))

    if "date" in signals_df.columns:
        signals_df = ensure_datetime_column(signals_df, "date").set_index("date", drop=False)

    ensemble_mode = "ensemble_score" in signals_df.columns

    if ensemble_mode:
        if "ticker" not in signals_df.columns or "date" not in signals_df.columns:
            raise ValueError("Ensemble scores require ticker and date columns")
        signals = signals_df.set_index("date")
    elif "ticker" in signals_df.columns and "signal" in signals_df.columns and "date" in signals_df.columns:
        pivot = signals_df.pivot_table(index="date", columns="ticker", values="signal")
        signals = pivot
    else:
        if "date" not in signals_df.columns:
            raise ValueError("signals_df must include an explicit date column")
        signals = signals_df.set_index("date")

    signals = signals.sort_index()
    if signals.index.tz is None:
        signals.index = signals.index.tz_localize("UTC")
    else:
        signals.index = signals.index.tz_convert("UTC")

    all_returns = {}
    for ticker, df in price_data.items():
        price_series = _select_price_series(df)
        all_returns[ticker] = price_series.pct_change()
    returns_df = pd.DataFrame(all_returns).sort_index()
    returns_df.index = pd.to_datetime(returns_df.index, utc=True, errors="coerce")
    returns_df = returns_df.loc[~returns_df.index.isna()].sort_index()
    if returns_df.empty:
        raise ValueError("Returns dataframe is empty after normalization; check price history alignment.")

    signal_dates = signals.index
    common_dates = signal_dates.intersection(returns_df.index)
    if common_dates.empty and isinstance(signal_dates, pd.DatetimeIndex) and isinstance(returns_df.index, pd.DatetimeIndex):
        signals_norm = signal_dates.normalize()
        returns_norm = returns_df.index.normalize()
        if not signals_norm.intersection(returns_norm).empty:
            logger.warning(
                "Signals and returns overlap by calendar date but not exact timestamp; normalizing signals to daily close timestamps."
            )
            signals = signals.copy()
            signals.index = signals_norm
            signals = signals.groupby(level=0).agg(
                lambda col: col.dropna().iloc[-1] if col.notna().any() else pd.NA
            )
            signals = signals.ffill()
            signal_dates = signals.index
    common_dates = signal_dates.intersection(returns_df.index)
    alignment_issue = False
    if common_dates.empty:
        alignment_issue = True
        signals_summary = _index_summary(signals.index)
        returns_summary = _index_summary(returns_df.index)
        
        # Try to infer frequency, but handle cases with < 3 dates
        try:
            signals_freq = pd.infer_freq(signals.index)
        except (ValueError, TypeError):
            signals_freq = None
        try:
            returns_freq = pd.infer_freq(returns_df.index)
        except (ValueError, TypeError):
            returns_freq = None
        
        logger.warning(
            "Signals/returns index mismatch: signals=%s->%s tz=%s freq=%s returns=%s->%s tz=%s freq=%s",
            signals_summary["start"],
            signals_summary["end"],
            signals_summary["tz"],
            signals_freq,
            returns_summary["start"],
            returns_summary["end"],
            returns_summary["tz"],
            returns_freq,
        )
        diagnostics_payload = {
            "issue": "signals_returns_no_overlap",
            "signals": signals_summary,
            "returns": returns_summary,
            "research_mode": research_mode,
            "action": "ffill_reindex" if research_mode else "raise",
        }
        diagnostics_path = _write_diagnostics(diagnostics_payload, "portfolio_alignment")
        logger.warning("Alignment diagnostics written to %s", diagnostics_path)
        if research_mode:
            if ensemble_mode:
                signal_dates = pd.DatetimeIndex(sorted(signal_dates.unique()))
            else:
                signals = signals.reindex(returns_df.index, method="ffill")
                signal_dates = signals.index
        else:
            raise RuntimeError("Signals and returns have no overlapping dates; see diagnostics for details.")
    else:
        if ensemble_mode:
            signal_dates = pd.DatetimeIndex(sorted(signal_dates.unique()))
        else:
            signals = signals.reindex(returns_df.index, method="ffill")
            signal_dates = signals.index
    if signals.empty or returns_df.empty:
        raise ValueError("Signals and returns are misaligned; unable to compute portfolio weights.")

    if rebalance == "single" and len(signal_dates) > 0:
        rebal_dates = pd.DatetimeIndex([signal_dates.max()])
    else:
        rebal_dates = _rebalance_dates(signal_dates, rebalance)
        rebal_dates = pd.DatetimeIndex([d for d in rebal_dates if d in signal_dates])
        if rebal_dates.empty and len(signal_dates) > 0:
            rebal_dates = pd.DatetimeIndex([signal_dates.max()])

    portfolio_returns = pd.Series(0.0, index=returns_df.index)
    weight_history = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)

    last_weights = pd.Series(0.0, index=returns_df.columns)
    policy_repairs: list[dict] = []
    diagnostics: list[dict] = []
    had_buy_signal = False
    had_scores = False
    had_valid_weights = False
    for date in rebal_dates:
        if date not in signals.index:
            continue
        if ensemble_mode:
            score_rows = signals.loc[date]
            if isinstance(score_rows, pd.Series):
                score_rows = score_rows.to_frame().T
            score_rows = score_rows.dropna(subset=["ensemble_score"]) if not score_rows.empty else score_rows
            if score_rows.empty:
                diagnostics.append({"date": date.isoformat(), "reason": "no_scores"})
                last_weights = pd.Series(0.0, index=returns_df.columns)
            else:
                had_scores = True
                scores = score_rows.set_index("ticker")["ensemble_score"].sort_values(ascending=False)
                scores = scores.head(top_k)
                weights, weight_diags = _compute_score_weights(
                    scores,
                    price_data,
                    date,
                    lookback=20,
                    min_volume=float(min_volume) if min_volume is not None else None,
                )
                diagnostics.extend(weight_diags)
                if weights.empty or weights.sum() == 0:
                    diagnostics.append({"date": date.isoformat(), "reason": "zero_weights"})
                    last_weights = pd.Series(0.0, index=returns_df.columns)
                else:
                    had_valid_weights = True
                    weights = apply_risk_constraints(weights, price_data, date, config)
                    weights, repairs = enforce_weight_policy(weights, config)
                    policy_repairs.extend({**repair, "date": date.isoformat()} for repair in repairs)
                    full_weights = pd.Series(0.0, index=returns_df.columns)
                    full_weights.update(weights)
                    full_weights, repairs = enforce_turnover_policy(full_weights, last_weights, config)
                    policy_repairs.extend({**repair, "date": date.isoformat()} for repair in repairs)
                    last_weights = full_weights
        else:
            signal_row = signals.loc[date]
            selected = [ticker for ticker, val in signal_row.items() if _infer_signal(val)]
            if not selected:
                logger.info("No BUY signals at %s; holding cash.", date.date())
                diagnostics.append({"date": date.isoformat(), "reason": "no_signals"})
                last_weights = pd.Series(0.0, index=returns_df.columns)
            else:
                had_buy_signal = True
                weights = _compute_weights(selected, price_data, date, weighting)
                weights = apply_risk_constraints(weights, price_data, date, config)
                weights, repairs = enforce_weight_policy(weights, config)
                policy_repairs.extend(
                    {**repair, "date": date.isoformat()} for repair in repairs
                )
                full_weights = pd.Series(0.0, index=returns_df.columns)
                full_weights.update(weights)
                full_weights, repairs = enforce_turnover_policy(full_weights, last_weights, config)
                policy_repairs.extend(
                    {**repair, "date": date.isoformat()} for repair in repairs
                )
                last_weights = full_weights

        weight_history.loc[date:] = last_weights.values

    weight_history = weight_history.ffill().fillna(0)
    if weight_history.to_numpy().sum() == 0:
        if alignment_issue:
            zero_reason = "misalignment"
        elif ensemble_mode and signals["ensemble_score"].isna().all():
            zero_reason = "nan_scores"
        elif not ensemble_mode and signals.isna().all().all():
            zero_reason = "nan_signals"
        elif ensemble_mode and not had_scores:
            zero_reason = "no_scores"
        elif not ensemble_mode and not had_buy_signal:
            zero_reason = "signals_point_in_time_no_historical_coverage" if signals_point_in_time else "no_buy_signals"
        elif not had_valid_weights and ensemble_mode:
            zero_reason = "invalid_weights"
        else:
            zero_reason = "unknown"
        diagnostics.append({"date": None, "reason": f"all_zero_weights:{zero_reason}"})
        diagnostics_payload = {
            "issue": "all_zero_weights",
            "reason": zero_reason,
            "signals": _index_summary(signals.index),
            "returns": _index_summary(returns_df.index),
            "diagnostics": diagnostics[-10:],
        }
        diagnostics_path = _write_diagnostics(diagnostics_payload, "portfolio_weights")
        logger.warning("Zero-weight diagnostics written to %s", diagnostics_path)

    portfolio_returns = (returns_df * weight_history).sum(axis=1)
    portfolio_returns = portfolio_returns.fillna(0)

    cumulative = (1 + portfolio_returns).cumprod()
    cumulative_returns = cumulative - 1
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()

    vol = portfolio_returns.std() * np.sqrt(252)
    mean_return = portfolio_returns.mean() * 252
    sharpe = mean_return / vol if vol not in (0, np.nan) else np.nan

    downside = portfolio_returns[portfolio_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if not downside.empty else np.nan
    sortino = mean_return / downside_std if downside_std not in (0, np.nan) else np.nan

    return_stability = 1 / (1 + portfolio_returns.rolling(window=20, min_periods=5).std().mean())

    metrics = {
        "cumulative_return": float(cumulative_returns.iloc[-1]) if not cumulative_returns.empty else 0.0,
        "volatility": float(vol) if not np.isnan(vol) else 0.0,
        "max_drawdown": float(max_drawdown) if not np.isnan(max_drawdown) else 0.0,
        "sharpe_ratio": float(sharpe) if not np.isnan(sharpe) else 0.0,
        "sortino_ratio": float(sortino) if not np.isnan(sortino) else 0.0,
        "return_stability": float(return_stability) if not np.isnan(return_stability) else 0.0,
    }

    return {
        "returns": portfolio_returns,
        "metrics": metrics,
        "weights": weight_history,
        "policy_repairs": policy_repairs,
        "diagnostics": diagnostics,
    }
