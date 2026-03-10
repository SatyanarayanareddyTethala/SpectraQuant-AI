"""Capital Intelligence — portfolio risk and exposure control.

Enforces position-level and portfolio-level constraints before allowing
new trades, and generates a risk report for the intelligence pipeline.

Checks performed
----------------
- Daily loss limit (stop trading for the day if hit)
- Gross exposure cap (total notional / equity)
- Single-name exposure cap
- Sector exposure cap
- Max simultaneous open positions
- Per-trade minimum confidence gate
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default limits (overridden by config)
# ---------------------------------------------------------------------------

_DEFAULT_LIMITS: Dict[str, Any] = {
    "equity_base": 1_000_000.0,
    "max_positions": 10,
    "max_gross_exposure": 1.5,     # 150 % of equity
    "max_name_exposure": 0.10,     # 10 % per name
    "max_sector_exposure": 0.30,   # 30 % per sector
    "daily_loss_limit": 5_000.0,   # absolute INR / base currency
    "min_confidence": 0.50,
    "cooldown_minutes": 60,
}


# ---------------------------------------------------------------------------
# Exposure calculations
# ---------------------------------------------------------------------------

def compute_exposures(
    positions: List[Dict[str, Any]],
    equity: float,
) -> Dict[str, Any]:
    """Compute current exposure metrics from open positions.

    Parameters
    ----------
    positions : list[dict]
        Each entry: ``{ticker, sector, notional, side}``.
        ``notional`` = shares × current_price (absolute value).
    equity : float
        Current portfolio equity.

    Returns
    -------
    dict
        Keys: ``gross_exposure``, ``by_name``, ``by_sector``,
        ``long_notional``, ``short_notional``.
    """
    gross = 0.0
    long_notional = 0.0
    short_notional = 0.0
    by_name: Dict[str, float] = {}
    by_sector: Dict[str, float] = {}

    for pos in positions:
        notional = abs(float(pos.get("notional", 0.0)))
        side = str(pos.get("side", "long")).lower()
        ticker = str(pos.get("ticker", ""))
        sector = str(pos.get("sector", "")) or "UNKNOWN"

        gross += notional
        if side == "long":
            long_notional += notional
        else:
            short_notional += notional

        by_name[ticker] = by_name.get(ticker, 0.0) + notional
        by_sector[sector] = by_sector.get(sector, 0.0) + notional

    safe_equity = max(equity, 1.0)
    return {
        "gross_exposure": gross / safe_equity,
        "long_notional": long_notional,
        "short_notional": short_notional,
        "by_name": {k: v / safe_equity for k, v in by_name.items()},
        "by_sector": {k: v / safe_equity for k, v in by_sector.items()},
        "position_count": len(positions),
    }


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

def check_trade_allowed(
    trade: Dict[str, Any],
    current_exposures: Dict[str, Any],
    daily_pnl: float,
    limits: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Determine whether a proposed trade is allowed.

    Parameters
    ----------
    trade : dict
        Proposed trade with keys: ``ticker``, ``side``, ``notional``,
        ``sector``, ``confidence``.
    current_exposures : dict
        Output of :func:`compute_exposures`.
    daily_pnl : float
        Realised + unrealised PnL today (negative = loss).
    limits : dict, optional
        Override default limits.

    Returns
    -------
    tuple[bool, list[str]]
        ``(allowed, rejection_reasons)``
    """
    lim = dict(_DEFAULT_LIMITS)
    if limits:
        lim.update(limits)

    reasons: List[str] = []
    equity = float(lim.get("equity_base", 1_000_000.0))

    # Daily loss limit (limit is stored as a positive threshold value)
    if daily_pnl <= -float(lim["daily_loss_limit"]):
        reasons.append(
            f"daily_loss_limit breached: pnl={daily_pnl:.2f} limit={lim['daily_loss_limit']:.2f}"
        )

    # Max positions
    if current_exposures.get("position_count", 0) >= int(lim["max_positions"]):
        reasons.append(
            f"max_positions reached: {current_exposures['position_count']} "
            f">= {lim['max_positions']}"
        )

    # Gross exposure
    trade_notional = abs(float(trade.get("notional", 0.0)))
    new_gross = current_exposures.get("gross_exposure", 0.0) + trade_notional / max(equity, 1.0)
    if new_gross > float(lim["max_gross_exposure"]):
        reasons.append(
            f"gross_exposure would be {new_gross:.3f} > {lim['max_gross_exposure']:.3f}"
        )

    # Name exposure
    ticker = str(trade.get("ticker", ""))
    current_name_exp = current_exposures.get("by_name", {}).get(ticker, 0.0)
    new_name_exp = current_name_exp + trade_notional / max(equity, 1.0)
    if new_name_exp > float(lim["max_name_exposure"]):
        reasons.append(
            f"name_exposure for {ticker} would be {new_name_exp:.3f} "
            f"> {lim['max_name_exposure']:.3f}"
        )

    # Sector exposure
    sector = str(trade.get("sector", "")) or "UNKNOWN"
    current_sec_exp = current_exposures.get("by_sector", {}).get(sector, 0.0)
    new_sec_exp = current_sec_exp + trade_notional / max(equity, 1.0)
    if new_sec_exp > float(lim["max_sector_exposure"]):
        reasons.append(
            f"sector_exposure for {sector} would be {new_sec_exp:.3f} "
            f"> {lim['max_sector_exposure']:.3f}"
        )

    # Minimum confidence
    confidence = float(trade.get("confidence", 1.0))
    min_conf = float(lim.get("min_confidence", 0.50))
    if confidence < min_conf:
        reasons.append(
            f"confidence {confidence:.3f} < min_confidence {min_conf:.3f}"
        )

    allowed = len(reasons) == 0
    if not allowed:
        logger.info("Trade %s rejected: %s", ticker, reasons[0])
    return allowed, reasons


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------

def generate_risk_report(
    positions: List[Dict[str, Any]],
    daily_pnl: float,
    equity: float,
    limits: Optional[Dict[str, Any]] = None,
    output_dir: str = "reports/intelligence",
) -> Dict[str, Any]:
    """Build a risk snapshot and optionally persist to disk.

    Parameters
    ----------
    positions : list[dict]
        Open positions.
    daily_pnl : float
        Realised + unrealised PnL for the day.
    equity : float
        Current portfolio equity.
    limits : dict, optional
        Override default limits.
    output_dir : str
        Where to write the report JSON.

    Returns
    -------
    dict
        Risk report dict.
    """
    lim = dict(_DEFAULT_LIMITS)
    if limits:
        lim.update(limits)

    exposures = compute_exposures(positions, equity)

    # Determine limit breaches
    breaches: List[str] = []
    if daily_pnl <= -float(lim["daily_loss_limit"]):
        breaches.append("daily_loss_limit")
    if exposures["gross_exposure"] > float(lim["max_gross_exposure"]):
        breaches.append("gross_exposure")
    if exposures["position_count"] >= int(lim["max_positions"]):
        breaches.append("max_positions")
    for sec, exp in exposures["by_sector"].items():
        if exp > float(lim["max_sector_exposure"]):
            breaches.append(f"sector_exposure:{sec}")
    for name, exp in exposures["by_name"].items():
        if exp > float(lim["max_name_exposure"]):
            breaches.append(f"name_exposure:{name}")

    report: Dict[str, Any] = {
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
        "equity": equity,
        "daily_pnl": daily_pnl,
        "exposures": exposures,
        "limits": lim,
        "breaches": breaches,
        "status": "warning" if breaches else "ok",
    }

    # Persist
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"risk_report_{ts}.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    report["output_path"] = str(out_path)

    logger.info(
        "Risk report: equity=%.0f pnl=%.2f breaches=%s",
        equity,
        daily_pnl,
        breaches or "none",
    )
    return report
