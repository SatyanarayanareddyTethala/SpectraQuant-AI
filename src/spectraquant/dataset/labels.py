"""Label generation utilities."""
from __future__ import annotations

import pandas as pd


def compute_forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Compute forward returns for the given horizon."""
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    close = pd.to_numeric(close, errors="coerce").astype(float)
    fwd = close.shift(-horizon) / close - 1
    return fwd


__all__ = ["compute_forward_returns"]
