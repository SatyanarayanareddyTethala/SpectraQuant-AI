"""Market regime detection utilities."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from spectraquant.data.normalize import normalize_price_frame


def compute_regime(price_data: Dict[str, pd.DataFrame]) -> str:
    """Compute a simple market regime from aggregate returns."""

    if not price_data:
        raise ValueError("Price data missing for regime computation")

    returns = []
    for df in price_data.values():
        df = normalize_price_frame(df)
        close = pd.to_numeric(df.get("close"), errors="coerce")
        series = close.pct_change().dropna()
        if not series.empty:
            returns.append(series)
    if not returns:
        raise ValueError("Unable to compute regime; no returns available")
    combined = pd.concat(returns, axis=1).mean(axis=1)
    vol = combined.tail(20).std()
    drawdown = (1 + combined).cumprod()
    rolling_max = drawdown.cummax()
    dd = (drawdown / rolling_max - 1).min()
    trend = combined.tail(20).mean()

    if dd < -0.05 or vol > 0.03:
        return "RISK_OFF"
    if trend > 0.002:
        return "RISK_ON"
    return "NEUTRAL"
