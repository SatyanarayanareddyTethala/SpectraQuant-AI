"""Snapshot-based paper execution engine."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from spectraquant.core.schema import order_columns, schema_version_for
from spectraquant.core.time import ensure_datetime_column
from spectraquant.data.normalize import normalize_price_columns


def run_paper_execution(
    weights_df: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    config: Dict,
    output_dir: Path,
) -> Dict[str, Path]:
    """Simulate snapshot-based fills using latest portfolio weights."""

    weights_df = ensure_datetime_column(weights_df, "date")
    latest_date = weights_df["date"].max()
    snapshot = weights_df[weights_df["date"] == latest_date]
    if snapshot.empty:
        raise ValueError("No portfolio snapshot available for execution.")

    current_weights = snapshot.drop(columns=["date"]).iloc[0].fillna(0.0)
    previous_snapshots = weights_df[weights_df["date"] < latest_date]
    if previous_snapshots.empty:
        previous_weights = pd.Series(0.0, index=current_weights.index)
    else:
        previous_weights = (
            previous_snapshots[previous_snapshots["date"] == previous_snapshots["date"].max()]
            .drop(columns=["date"])
            .iloc[0]
            .reindex(current_weights.index)
            .fillna(0.0)
        )

    deltas = current_weights - previous_weights
    if deltas.abs().sum() == 0:
        raise ValueError("No portfolio weight changes to execute.")

    slippage_bps = float(config.get("execution", {}).get("slippage_bps", 5))
    cost_bps = float(config.get("execution", {}).get("transaction_cost_bps", 1))
    slippage_rate = slippage_bps / 10000
    cost_rate = cost_bps / 10000

    trades = []
    fills = []
    costs = []
    for ticker, delta in deltas.items():
        if delta == 0 or ticker not in price_data:
            continue
        price_df = price_data.get(ticker)
        if price_df is None or price_df.empty:
            continue
        norm = normalize_price_columns(price_df, ticker)
        if "date" not in norm.columns and isinstance(norm.index, pd.DatetimeIndex):
            norm = norm.reset_index()
        norm = ensure_datetime_column(norm, "date")
        price_series = norm.loc[norm["date"] <= latest_date, "close"]
        if price_series.empty:
            continue
        last_price = float(price_series.iloc[-1])
        side = "BUY" if delta > 0 else "SELL"
        fill_price = last_price * (1 + slippage_rate if delta > 0 else 1 - slippage_rate)
        notional = abs(delta)
        slippage_cost = abs(last_price * notional * slippage_rate)
        transaction_cost = abs(last_price * notional * cost_rate)
        trades.append(
            {
                "date": latest_date,
                "ticker": ticker,
                "side": side,
                "weight_change": float(delta),
                "price": last_price,
            }
        )
        fills.append(
            {
                "date": latest_date,
                "ticker": ticker,
                "side": side,
                "weight_change": float(delta),
                "fill_price": fill_price,
            }
        )
        costs.append(
            {
                "date": latest_date,
                "ticker": ticker,
                "weight_change": float(delta),
                "slippage_cost": slippage_cost,
                "transaction_cost": transaction_cost,
                "total_cost": slippage_cost + transaction_cost,
            }
        )

    if trades:
        trades_df = ensure_datetime_column(pd.DataFrame(trades), "date")
    else:
        trades_df = pd.DataFrame(columns=["date", "ticker", "side", "weight_change", "price"])

    if fills:
        fills_df = ensure_datetime_column(pd.DataFrame(fills), "date")
    else:
        fills_df = pd.DataFrame(columns=["date", "ticker", "side", "weight_change", "fill_price"])

    if costs:
        costs_df = ensure_datetime_column(pd.DataFrame(costs), "date")
    else:
        costs_df = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "weight_change",
                "slippage_cost",
                "transaction_cost",
                "total_cost",
            ]
        )
    total_costs = float(costs_df["total_cost"].sum()) if not costs_df.empty else 0.0
    pnl_df = pd.DataFrame(
        [
            {
                "date": latest_date,
                "gross_pnl": 0.0,
                "costs": total_costs,
                "net_pnl": -total_costs,
            }
        ]
    )
    pnl_df = ensure_datetime_column(pnl_df, "date")

    trades_df["schema_version"] = schema_version_for("execution_trades")
    fills_df["schema_version"] = schema_version_for("execution_fills")
    costs_df["schema_version"] = schema_version_for("execution_costs")
    pnl_df["schema_version"] = schema_version_for("execution_pnl")

    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "trades.csv"
    fills_path = output_dir / "fills.csv"
    costs_path = output_dir / "costs.csv"
    pnl_path = output_dir / "daily_pnl.csv"
    trades_df = order_columns(trades_df, "execution_trades")
    fills_df = order_columns(fills_df, "execution_fills")
    costs_df = order_columns(costs_df, "execution_costs")
    pnl_df = order_columns(pnl_df, "execution_pnl")

    trades_df.to_csv(trades_path, index=False)
    fills_df.to_csv(fills_path, index=False)
    costs_df.to_csv(costs_path, index=False)
    pnl_df.to_csv(pnl_path, index=False)

    return {
        "trades": trades_path,
        "fills": fills_path,
        "costs": costs_path,
        "daily_pnl": pnl_path,
    }
