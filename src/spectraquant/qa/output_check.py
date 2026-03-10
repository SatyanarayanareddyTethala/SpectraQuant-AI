"""Validation utilities for signals and portfolio outputs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from spectraquant.core.time import (
    ensure_datetime_column,
    is_intraday_horizon,
    resolve_prediction_date_for_horizon,
)

logger = logging.getLogger(__name__)


REQUIRED_SIGNAL_COLUMNS = {"ticker", "signal"}


def check_signals(signals_df: pd.DataFrame) -> None:
    """Validate that signals are well-formed and finite."""

    assert isinstance(signals_df, pd.DataFrame), "signals_df must be DataFrame"
    assert not signals_df.empty, "signals_df is empty"
    assert "date" in signals_df.columns, "signals_df missing date column"
    signals_df = signals_df.copy()
    signals_df["date"] = pd.to_datetime(signals_df["date"], utc=True, errors="coerce")
    assert signals_df["date"].notna().all(), "signals_df date column contains invalid values"

    if REQUIRED_SIGNAL_COLUMNS.issubset(signals_df.columns):
        values = signals_df["signal"]
    else:
        values = signals_df.stack(dropna=False)

    assert values.notna().all(), "Signals contain NaNs"
    normalized = values.astype(str).str.upper().str.strip()
    allowed = {"BUY", "HOLD", "SELL"}
    assert normalized.isin(allowed).all(), "Signals contain invalid values"

    logger.info("Signals integrity OK: %s rows", len(signals_df))


def write_date_alignment_report(
    pred_df: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    intraday_price_data: Dict[str, pd.DataFrame] | None = None,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Write horizon-aware date alignment diagnostics for predictions."""

    pred_df = ensure_datetime_column(pred_df.copy(), "date")
    rows = []
    for (ticker, horizon), group in pred_df.groupby(["ticker", "horizon"], dropna=False):
        horizon = str(horizon)
        source = intraday_price_data if is_intraday_horizon(horizon) else price_data
        price_df = source.get(ticker) if source else None
        if price_df is None:
            rows.append(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "last_price_date": None,
                    "prediction_date": group["date"].max(),
                    "delta_seconds": None,
                    "status": "FAIL",
                    "message": "Missing price data",
                }
            )
            continue
        try:
            last_price_date = resolve_prediction_date_for_horizon(
                ticker, horizon, price_data, intraday_price_data
            )
            pred_date = group["date"].max()
            delta = (pred_date - last_price_date).total_seconds()
            status = "PASS" if delta == 0 else "FAIL"
            rows.append(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "last_price_date": last_price_date,
                    "prediction_date": pred_date,
                    "delta_seconds": int(delta),
                    "status": status,
                    "message": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "last_price_date": None,
                    "prediction_date": group["date"].max(),
                    "delta_seconds": None,
                    "status": "FAIL",
                    "message": str(exc),
                }
            )

    report_df = pd.DataFrame(rows)
    report_df["date"] = report_df["prediction_date"]
    report_df = ensure_datetime_column(report_df, "date")
    output_dir = output_dir or Path("reports/qa")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"date_alignment_{timestamp}.csv"
    report_df.to_csv(path, index=False)
    return path


def check_prediction_dates(
    pred_df: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    intraday_price_data: Dict[str, pd.DataFrame] | None = None,
) -> None:
    """Validate prediction dates align with latest available price timestamps."""

    assert isinstance(pred_df, pd.DataFrame), "pred_df must be DataFrame"
    assert not pred_df.empty, "pred_df is empty"
    assert {"ticker", "date"}.issubset(pred_df.columns), "pred_df missing ticker/date columns"
    pred_df = ensure_datetime_column(pred_df, "date")

    for (ticker, horizon), pred_row in pred_df.groupby(["ticker", "horizon"], dropna=False):
        horizon = str(horizon)
        latest_date = resolve_prediction_date_for_horizon(
            ticker, horizon, price_data, intraday_price_data
        )
        assert pd.notna(latest_date), f"No valid price dates for {ticker}"
        assert latest_date.year != 1970, f"Epoch date detected for {ticker}"
        pred_date = pred_row["date"].max()
        assert pred_date == latest_date, (
            f"Prediction date mismatch for {ticker} ({horizon}): pred={pred_date} price={latest_date}"
        )

    logger.info("Prediction dates align with latest price timestamps for %s tickers", pred_df["ticker"].nunique())


def check_portfolio_outputs(results: Dict) -> None:
    """Validate simulated portfolio outputs."""

    returns = results.get("returns")
    metrics = results.get("metrics", {})

    assert isinstance(returns, (pd.Series, np.ndarray)), "Portfolio returns missing"
    if isinstance(returns, np.ndarray):
        returns_series = pd.Series(returns)
    else:
        returns_series = returns

    assert pd.api.types.is_numeric_dtype(returns_series), "Portfolio returns are not numeric"
    assert returns_series.notna().all(), "Portfolio returns contain NaNs"
    assert returns_series.index.is_monotonic_increasing, "Portfolio returns index not sorted by date"

    cumulative = (1 + returns_series).cumprod()
    assert cumulative.index.is_monotonic_increasing, "Cumulative returns not aligned by date"

    risk_score = metrics.get("risk_score")
    if risk_score is not None:
        assert 0 <= float(risk_score) <= 100, "Risk score out of bounds"

    logger.info(
        "Portfolio outputs validated: %s return points, metrics keys=%s",
        len(returns_series),
        sorted(metrics.keys()),
    )


def check_execution_accounting(
    trades_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    pnl_df: pd.DataFrame,
) -> None:
    """Validate execution accounting identities."""

    trades_df = trades_df.copy()
    fills_df = fills_df.copy()
    costs_df = costs_df.copy()
    pnl_df = ensure_datetime_column(pnl_df.copy(), "date")

    if not trades_df.empty:
        trades_df = ensure_datetime_column(trades_df, "date")
    if not fills_df.empty:
        fills_df = ensure_datetime_column(fills_df, "date")
    if not costs_df.empty:
        costs_df = ensure_datetime_column(costs_df, "date")

    if trades_df.empty and fills_df.empty and costs_df.empty:
        gross = float(pnl_df["gross_pnl"].sum()) if not pnl_df.empty else 0.0
        costs = float(pnl_df["costs"].sum()) if not pnl_df.empty else 0.0
        net = float(pnl_df["net_pnl"].sum()) if not pnl_df.empty else 0.0
        if not np.isclose(gross - costs, net, atol=1e-8):
            raise AssertionError("Execution P&L identity failed: gross - costs != net")
        return

    if trades_df.empty or fills_df.empty:
        raise AssertionError("Execution trades/fills are empty")

    if not trades_df["ticker"].isin(fills_df["ticker"]).all():
        raise AssertionError("Trades contain tickers missing from fills")

    if costs_df.empty:
        raise AssertionError("Execution costs are empty")

    if pnl_df.empty:
        raise AssertionError("Execution P&L is empty")

    gross = float(pnl_df["gross_pnl"].sum())
    costs = float(pnl_df["costs"].sum())
    net = float(pnl_df["net_pnl"].sum())
    if not np.isclose(gross - costs, net, atol=1e-8):
        raise AssertionError("Execution P&L identity failed: gross - costs != net")

    fill_changes = fills_df.groupby("ticker")["weight_change"].sum()
    trade_changes = trades_df.groupby("ticker")["weight_change"].sum()
    if not fill_changes.equals(trade_changes):
        raise AssertionError("Execution fills do not match trade weight changes")
