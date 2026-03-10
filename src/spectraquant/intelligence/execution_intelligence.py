"""Execution Intelligence — wait, watch, and execute with trigger logic.

Manages the lifecycle of planned trades through three states:
  WAIT     → conditions not yet met; monitoring price.
  WATCH    → entry condition approaching; increase polling frequency.
  EXECUTE  → all conditions met; ready to submit order.

Also respects cooldowns and do-not-trade rules from the policy engine.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade state machine
# ---------------------------------------------------------------------------

class TradeState(str, Enum):
    WAIT = "WAIT"
    WATCH = "WATCH"
    EXECUTE = "EXECUTE"
    BLOCKED = "BLOCKED"
    DONE = "DONE"


# ---------------------------------------------------------------------------
# Trigger evaluation
# ---------------------------------------------------------------------------

def _price_trigger_met(
    current_price: float,
    side: str,
    entry: Optional[float],
    watch_pct: float = 0.005,
) -> TradeState:
    """Determine WAIT / WATCH / EXECUTE based on price proximity to entry."""
    if entry is None:
        return TradeState.WAIT

    diff_pct = (current_price - entry) / max(abs(entry), 1e-9)

    if side == "long":
        if current_price >= entry:
            return TradeState.EXECUTE
        elif diff_pct >= -watch_pct:
            return TradeState.WATCH
        return TradeState.WAIT
    else:  # short
        if current_price <= entry:
            return TradeState.EXECUTE
        elif diff_pct <= watch_pct:
            return TradeState.WATCH
        return TradeState.WAIT


def evaluate_trigger(
    trade: Dict[str, Any],
    market_snapshot: Dict[str, Any],
    policy_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a single plan trade against current market data.

    Parameters
    ----------
    trade : dict
        Plan trade entry (from :func:`trade_planner.generate_premarket_plan`).
    market_snapshot : dict
        Current market data keyed by ticker:
        ``{ticker: {"price": float, "volume": float, "spread_bps": float}}``.
    policy_config : dict, optional
        Risk / policy parameters (used for do-not-trade checks).

    Returns
    -------
    dict
        Updated trade dict with added keys: ``state``, ``trigger_reason``,
        ``evaluated_at``.
    """
    result = dict(trade)
    ticker = str(trade.get("ticker", ""))
    side = str(trade.get("side", "long"))
    entry = trade.get("entry")
    stop = trade.get("stop")
    cfg = policy_config or {}

    snap = market_snapshot.get(ticker, {})
    price = snap.get("price")

    evaluated_at = datetime.now(tz=timezone.utc).isoformat()

    if price is None:
        result.update(
            state=TradeState.WAIT,
            trigger_reason="no_market_data",
            evaluated_at=evaluated_at,
        )
        return result

    price = float(price)

    # ---- Do-not-trade checks -------------------------------------------
    dnt_reasons: List[str] = []

    # Stop-loss breach: price already past stop
    if stop is not None:
        if side == "long" and price <= float(stop):
            dnt_reasons.append(f"price {price} already at/below stop {stop}")
        elif side == "short" and price >= float(stop):
            dnt_reasons.append(f"price {price} already at/above stop {stop}")

    # Spread check
    spread_bps = snap.get("spread_bps", 0.0)
    max_spread = cfg.get("max_spread_bps", 50.0)
    if spread_bps and spread_bps > max_spread:
        dnt_reasons.append(f"spread {spread_bps:.1f} bps > {max_spread:.1f} bps")

    # News risk
    news_risk = float(trade.get("news_context", {}).get("risk_score", 0.0))
    if news_risk > cfg.get("news_risk_threshold", 0.7):
        dnt_reasons.append(f"elevated news risk {news_risk:.2f}")

    if dnt_reasons:
        result.update(
            state=TradeState.BLOCKED,
            trigger_reason="; ".join(dnt_reasons),
            evaluated_at=evaluated_at,
        )
        logger.info("Trade %s BLOCKED: %s", ticker, dnt_reasons[0])
        return result

    # ---- State machine -------------------------------------------------
    watch_pct = cfg.get("watch_entry_pct", 0.005)
    state = _price_trigger_met(price, side, entry, watch_pct=watch_pct)

    result.update(
        state=state,
        trigger_reason=f"price={price:.4f} vs entry={entry}",
        evaluated_at=evaluated_at,
    )
    logger.debug("Trade %s: %s (price=%.4f)", ticker, state, price)
    return result


# ---------------------------------------------------------------------------
# Cooldown manager
# ---------------------------------------------------------------------------

class CooldownManager:
    """Track per-ticker execution cooldowns in memory.

    Parameters
    ----------
    cooldown_minutes : float
        Cooldown period in minutes after an execution.
    """

    def __init__(self, cooldown_minutes: float = 60.0) -> None:
        self.cooldown_seconds = cooldown_minutes * 60.0
        self._last_exec: Dict[str, float] = {}

    def is_cooling(self, ticker: str) -> bool:
        """Return *True* if ticker is still in cooldown."""
        import time

        last = self._last_exec.get(ticker, 0.0)
        return (time.time() - last) < self.cooldown_seconds

    def record_execution(self, ticker: str) -> None:
        """Mark a ticker as just executed (starts cooldown)."""
        import time

        self._last_exec[ticker] = time.time()
        logger.debug("Cooldown started for %s (%.0f s)", ticker, self.cooldown_seconds)


# ---------------------------------------------------------------------------
# Batch monitor
# ---------------------------------------------------------------------------

def monitor_plan(
    plan: Dict[str, Any],
    market_snapshot: Dict[str, Any],
    policy_config: Optional[Dict[str, Any]] = None,
    cooldown_mgr: Optional[CooldownManager] = None,
) -> Dict[str, Any]:
    """Evaluate all trades in a plan against current market data.

    Parameters
    ----------
    plan : dict
        Pre-market plan (output of :func:`trade_planner.generate_premarket_plan`).
    market_snapshot : dict
        ``{ticker: {"price": float, ...}}``
    policy_config : dict, optional
        Policy/risk configuration.
    cooldown_mgr : CooldownManager, optional
        Shared cooldown manager (created internally if *None*).

    Returns
    -------
    dict
        Summary: ``{"ready": [...], "watching": [...], "waiting": [...],
        "blocked": [...], "as_of": str}``.
    """
    if cooldown_mgr is None:
        cooldown_mgr = CooldownManager(
            cooldown_minutes=float((policy_config or {}).get("cooldown_minutes", 60.0))
        )

    ready: List[Dict[str, Any]] = []
    watching: List[Dict[str, Any]] = []
    waiting: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []

    for trade in plan.get("trades", []):
        ticker = str(trade.get("ticker", ""))
        if cooldown_mgr.is_cooling(ticker):
            blocked.append({**trade, "state": TradeState.BLOCKED, "trigger_reason": "cooldown_active"})
            continue

        evaluated = evaluate_trigger(trade, market_snapshot, policy_config)
        state = evaluated.get("state", TradeState.WAIT)

        if state == TradeState.EXECUTE:
            ready.append(evaluated)
        elif state == TradeState.WATCH:
            watching.append(evaluated)
        elif state == TradeState.BLOCKED:
            blocked.append(evaluated)
        else:
            waiting.append(evaluated)

    return {
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
        "ready": ready,
        "watching": watching,
        "waiting": waiting,
        "blocked": blocked,
    }


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

def save_execution_snapshot(
    snapshot: Dict[str, Any],
    output_dir: str = "reports/intelligence",
) -> str:
    """Write execution snapshot to disk; return output path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"execution_snapshot_{ts}.json"
    with open(out_path, "w") as fh:
        json.dump(snapshot, fh, indent=2, default=str)
    logger.debug("Execution snapshot saved to %s", out_path)
    return str(out_path)
