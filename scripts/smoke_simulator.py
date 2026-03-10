#!/usr/bin/env python3
from __future__ import annotations

import sys

import pandas as pd

from dashboard.utils.artifacts import load_latest_predictions, load_latest_prices
from dashboard.utils.simulator import simulate_equity_curve


def main() -> int:
    preds = load_latest_predictions()
    if preds is None or preds.empty:
        print("No predictions found. Run: python -m src.spectraquant.cli.main predict")
        return 0

    if "ticker" not in preds.columns or "horizon" not in preds.columns:
        print("Predictions file missing required columns.")
        return 0

    preds = preds.copy()
    if "date" in preds.columns:
        preds["date"] = pd.to_datetime(preds["date"], errors="coerce")

    horizon = preds["horizon"].dropna().iloc[0]
    ticker = preds["ticker"].dropna().iloc[0]
    pred_row = preds[(preds["ticker"] == ticker) & (preds["horizon"] == horizon)].head(1)
    if pred_row.empty:
        print("No prediction row found for simulator.")
        return 0

    prices = load_latest_prices(ticker)
    close_series = pd.Series(dtype=float)
    if prices is not None:
        for candidate in ("close", "adj_close", "price"):
            if candidate in prices.columns:
                close_series = pd.to_numeric(prices[candidate], errors="coerce")
                break

    if close_series.empty:
        print("Price history missing; simulator will use estimated risk.")

    result = simulate_equity_curve(
        pred_row.iloc[0],
        close_series,
        holding_days=30,
        investment_amount=10000,
        horizon_days=1.0,
        transaction_cost_pct=0.001,
        management_fee_pct=0.005,
        sip_mode="None",
        sip_amount=0.0,
    )
    if result is None:
        print("Simulator could not run due to missing/invalid predictions.")
        return 0

    print("Simulator outputs:")
    print(f"  Expected final value: {result.expected_final_value:,.2f}")
    print(f"  Expected profit: {result.expected_profit:,.2f}")
    print(f"  CAGR: {result.cagr:+.2%}")
    print(f"  Risk label: {result.risk_label} {result.risk_note}".strip())
    print(f"  Max drawdown (P90): {result.max_drawdown_p90:+.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
