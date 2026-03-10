"""Simple REST-based collectors for free on-chain data sources.

Each ``fetch_*`` helper returns a plain *dict* on success or an empty
*dict* on any network / parse failure so that downstream code never
has to handle exceptions from the collection layer.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_TIMEOUT_S = 10


# ---------------------------------------------------------------------------
# Individual collectors
# ---------------------------------------------------------------------------

def fetch_mempool_fees(*, timeout: int = _TIMEOUT_S) -> dict[str, Any]:
    """Fetch Bitcoin mempool / fee estimates from *mempool.space*."""
    url = "https://mempool.space/api/v1/fees/recommended"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SpectraQuant/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode())
        logger.debug("mempool fees: %s", data)
        return data
    except Exception:
        logger.warning("Failed to fetch mempool fees", exc_info=True)
        return {}


def fetch_fear_greed_index(*, timeout: int = _TIMEOUT_S) -> dict[str, Any]:
    """Fetch the latest Crypto Fear & Greed Index value."""
    url = "https://api.alternative.me/fng/?limit=1&format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SpectraQuant/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode())
        entry = payload.get("data", [{}])[0]
        logger.debug("fear & greed: %s", entry)
        return entry
    except Exception:
        logger.warning("Failed to fetch fear & greed index", exc_info=True)
        return {}


def fetch_gas_prices(*, timeout: int = _TIMEOUT_S) -> dict[str, Any]:
    """Fetch Ethereum gas prices from the Etherscan public gas oracle."""
    url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SpectraQuant/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode())
        result = payload.get("result", {})
        logger.debug("gas prices: %s", result)
        return result if isinstance(result, dict) else {}
    except Exception:
        logger.warning("Failed to fetch gas prices", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def collect_all(symbols: list[str] | None = None) -> pd.DataFrame:
    """Aggregate all free on-chain sources into a single DataFrame.

    Parameters
    ----------
    symbols:
        Optional list of symbols to tag each row with.  When *None* the
        collectors still run but the ``symbol`` column is set to ``"MARKET"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a UTC ``asof_utc`` DatetimeIndex and a ``symbol``
        column alongside every metric collected.
    """
    now_utc = datetime.now(timezone.utc)
    symbols = symbols or ["MARKET"]

    mempool = fetch_mempool_fees()
    fng = fetch_fear_greed_index()
    gas = fetch_gas_prices()

    rows: list[dict[str, Any]] = []
    for sym in symbols:
        row: dict[str, Any] = {"asof_utc": now_utc, "symbol": sym}
        if mempool:
            row["mempool_fastest_fee"] = mempool.get("fastestFee")
            row["mempool_half_hour_fee"] = mempool.get("halfHourFee")
            row["mempool_hour_fee"] = mempool.get("hourFee")
        if fng:
            row["fear_greed_value"] = _safe_float(fng.get("value"))
        if gas:
            row["gas_safe_low"] = _safe_float(gas.get("SafeGasPrice"))
            row["gas_propose"] = _safe_float(gas.get("ProposeGasPrice"))
            row["gas_fast"] = _safe_float(gas.get("FastGasPrice"))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    df = df.set_index("asof_utc")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """Coerce *value* to float, returning *None* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
