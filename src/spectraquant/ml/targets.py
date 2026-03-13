"""Supervised target generation for SpectraQuant ML.

Creates binary classification targets from future price returns.
``future_return`` and ``target`` are appended to the dataframe; the last
*horizon* rows will have NaN targets because the future price is not yet
available—callers must dropna() before training.
"""
from __future__ import annotations

import pandas as pd


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Append ``future_return`` and ``target`` columns to *df*.

    Parameters
    ----------
    df:
        DataFrame containing a ``close`` column.
    horizon:
        Number of periods ahead used to compute the forward return.
        Must be a positive integer.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with two new columns:

        ``future_return``
            ``close.shift(-horizon) / close - 1``  (float, NaN for the last
            *horizon* rows)
        ``target``
            1 if ``future_return > 0``, else 0  (int, NaN propagated as NaN)

    Raises
    ------
    ValueError
        If *horizon* is not a positive integer or ``close`` is missing.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"add_target: horizon must be a positive integer, got {horizon!r}")

    cols_lower = [str(c).lower() for c in df.columns]
    if "close" not in cols_lower:
        raise ValueError("add_target: DataFrame must contain a 'close' column")

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    out["future_return"] = close.shift(-horizon) / close - 1
    out["target"] = (out["future_return"] > 0).astype("Int64").where(out["future_return"].notna())

    return out


__all__ = ["add_target"]
