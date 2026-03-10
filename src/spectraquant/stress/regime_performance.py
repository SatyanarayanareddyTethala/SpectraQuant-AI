"""Regime segmented performance analysis."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIMES = [
    "LOW_VOL_TREND",
    "LOW_VOL_CHOP",
    "HIGH_VOL_TREND",
    "HIGH_VOL_CHOP",
]


def _metrics(returns: pd.Series) -> dict[str, float]:
    if returns.empty:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0}
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()
    vol = returns.std() * np.sqrt(252)
    mean_return = returns.mean() * 252
    sharpe = mean_return / vol if vol not in (0, np.nan) else 0.0
    hit_rate = float((returns > 0).mean())
    return {
        "sharpe": float(sharpe) if sharpe == sharpe else 0.0,
        "max_drawdown": float(max_drawdown) if max_drawdown == max_drawdown else 0.0,
        "hit_rate": hit_rate,
    }


def analyze_regime_performance(portfolio_timeseries: Any, regime_series: Any) -> dict:
    returns = pd.Series(portfolio_timeseries).copy()
    regimes = pd.Series(regime_series).copy()
    returns.index = pd.to_datetime(returns.index, utc=True, errors="coerce")
    regimes.index = pd.to_datetime(regimes.index, utc=True, errors="coerce")

    combined = pd.concat([returns, regimes], axis=1).dropna()
    combined.columns = ["return", "regime"]

    report: dict[str, Any] = {"per_regime": {}}
    for regime in REGIMES:
        subset = combined.loc[combined["regime"] == regime, "return"]
        report["per_regime"][regime] = _metrics(subset)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("reports/stress")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"regime_perf_{run_id}.json"
    report.update({"run_id": run_id, "output_path": str(output_path)})
    output_path.write_text(json.dumps(report, indent=2))
    return report


__all__ = ["analyze_regime_performance"]
