"""Portfolio allocation — risk parity, volatility targeting, and signal blending.

Produces a weight vector that respects constraints and targets a given
portfolio volatility level.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def allocate_risk_parity(
    cov: pd.DataFrame | np.ndarray,
    symbols: list[str] | None = None,
) -> pd.Series:
    """Inverse-volatility (risk parity lite) allocation.

    Each asset receives weight proportional to ``1 / sigma_i``.

    Parameters
    ----------
    cov : pd.DataFrame or np.ndarray
        Covariance matrix (n × n).
    symbols : list of str, optional
        Asset names; inferred from *cov* if it is a DataFrame.

    Returns
    -------
    pd.Series
        Weights summing to 1.0, indexed by symbol.
    """
    if isinstance(cov, pd.DataFrame):
        symbols = symbols or list(cov.columns)
        cov_arr = cov.values
    else:
        cov_arr = np.asarray(cov)
        if symbols is None:
            symbols = [f"asset_{i}" for i in range(cov_arr.shape[0])]

    vols = np.sqrt(np.diag(cov_arr))
    vols = np.where(vols < 1e-12, 1e-12, vols)  # avoid division by zero
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return pd.Series(weights, index=symbols, name="weight")


def allocate_vol_target(
    scores: pd.Series,
    cov: pd.DataFrame | np.ndarray,
    target_vol: float = 0.15,
    symbols: list[str] | None = None,
) -> pd.Series:
    """Score-weighted allocation scaled to a target portfolio volatility.

    Steps:
    1. Convert scores to raw weights (proportional to abs score, sign preserved).
    2. Compute raw portfolio vol from *cov*.
    3. Scale weights so annualized portfolio vol ≈ *target_vol*.

    Parameters
    ----------
    scores : pd.Series
        Per-asset scores (e.g. blended agent output).  Index = symbols.
    cov : pd.DataFrame or np.ndarray
        Covariance matrix.
    target_vol : float
        Annualized volatility target (e.g. 0.15 = 15 %).
    symbols : list of str, optional
        Overrides score index for labelling.

    Returns
    -------
    pd.Series
        Weights (may sum to != 1 due to vol scaling), indexed by symbol.
    """
    if isinstance(cov, pd.DataFrame):
        cov_arr = cov.values
    else:
        cov_arr = np.asarray(cov)

    syms = symbols or list(scores.index)
    s = scores.reindex(syms).fillna(0.0).values.astype(float)

    total = np.abs(s).sum()
    if total < 1e-12:
        logger.warning("All scores near zero — returning equal weights")
        w = np.ones(len(syms)) / len(syms)
    else:
        w = s / total

    port_var = float(w @ cov_arr @ w)
    port_vol = np.sqrt(max(port_var, 1e-18))

    if port_vol > 1e-12:
        scale = target_vol / port_vol
    else:
        scale = 1.0

    scaled = w * scale
    return pd.Series(scaled, index=syms, name="weight")


def allocate(
    scores: pd.Series,
    returns: pd.DataFrame,
    method: str = "vol_target",
    target_vol: float = 0.15,
    lookback: int = 60,
    **kwargs: Any,
) -> pd.Series:
    """High-level allocation dispatcher.

    Parameters
    ----------
    scores : pd.Series
        Per-asset scores.
    returns : pd.DataFrame
        Historical returns, columns = symbols.
    method : str
        ``"risk_parity"`` or ``"vol_target"``.
    target_vol : float
        Annualized volatility target.
    lookback : int
        Rolling window (trading days) for covariance estimation.

    Returns
    -------
    pd.Series
        Portfolio weights.
    """
    common = sorted(set(scores.index) & set(returns.columns))
    if not common:
        logger.warning("No overlapping symbols between scores and returns")
        return pd.Series(dtype=float, name="weight")

    ret = returns[common].tail(lookback).dropna(axis=1, how="all")
    if ret.empty or ret.shape[0] < 2:
        logger.warning("Insufficient return data for covariance; equal weights")
        w = pd.Series(1.0 / len(common), index=common, name="weight")
        return w

    # Annualize daily covariance (assume 365 days for crypto)
    cov = ret.cov() * 365
    symbols = list(ret.columns)
    scores_aligned = scores.reindex(symbols).fillna(0.0)

    if method == "risk_parity":
        return allocate_risk_parity(cov, symbols)
    else:
        return allocate_vol_target(scores_aligned, cov, target_vol, symbols)
