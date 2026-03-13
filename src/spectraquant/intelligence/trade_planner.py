"""Trade Planner — pre-market ranked trade plan for NSE/India.

Generates a ranked, risk-sized trade plan one hour before market open by
combining:
- News-first candidate list (from news_universe when enabled)
- Signal / prediction scores
- Market regime from regime_engine
- Analog memory confidence calibration
- Capital intelligence risk constraints

Output is written to ``reports/plans/premarket_plan_YYYYMMDD_HHMMSS.json``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "thresholds": {"buy": 0.55, "sell": 0.45},
    "max_positions": 10,
    "max_sector_exposure": 0.30,
    "daily_loss_limit": 5000.0,
    "equity_base": 1_000_000.0,
    "risk_per_trade_pct": 0.01,
    "output_dir": "reports/plans",
    "planner_enabled": True,
}


# ---------------------------------------------------------------------------
# Risk sizing
# ---------------------------------------------------------------------------

def _size_position(
    equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
) -> int:
    """Compute share quantity via fixed-fractional risk sizing.

    Returns 0 if entry == stop (undefined risk).
    """
    risk_amount = equity * risk_pct
    risk_per_share = abs(entry - stop)
    if risk_per_share < 1e-9:
        return 0
    return max(0, int(risk_amount / risk_per_share))


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def _score_candidate(
    candidate: Dict[str, Any],
    regime: str,
    buy_threshold: float,
    use_analog: bool = False,
    analog_memory: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Score a single candidate and return a plan entry or None (filtered).

    Parameters
    ----------
    candidate : dict
        Must contain ``ticker``, ``score`` (signal strength 0-1),
        ``side`` (long/short), ``entry``, ``stop``.
        Optional: ``sector``, ``confidence``, ``news_context``.
    regime : str
        Current market regime label.
    buy_threshold : float
        Minimum signal score to include.
    use_analog : bool
        Whether to use analog memory for confidence calibration.
    analog_memory : AnalogMarketMemory, optional
        Memory instance for calibration.

    Returns
    -------
    dict or None
    """
    score = float(candidate.get("score", 0.0))
    side = str(candidate.get("side", "long")).lower()

    # Apply regime filter: suppress longs in RISK_OFF/PANIC
    if regime in ("RISK_OFF", "PANIC") and side == "long":
        score *= 0.5

    if score < buy_threshold:
        return None

    conf = float(candidate.get("confidence", score))

    # Optionally calibrate confidence with analog memory
    if use_analog and analog_memory is not None:
        try:
            state = {
                "rsi_14": candidate.get("rsi", 50.0),
                "atr_pct": candidate.get("atr_pct", 0.02),
                "regime": regime,
                "news_sentiment": candidate.get("news_sentiment", 0.0),
            }
            neighbors = analog_memory.query_similar(state, k=20)
            conf = analog_memory.calibrate_confidence(conf, neighbors)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:  # noqa: BLE001
            logger.debug("Analog calibration skipped: %s", exc)

    return {
        "ticker": candidate.get("ticker", ""),
        "side": side,
        "score": round(score, 4),
        "confidence": round(conf, 4),
        "entry": candidate.get("entry"),
        "stop": candidate.get("stop"),
        "target": candidate.get("target"),
        "sector": candidate.get("sector", ""),
        "news_context": candidate.get("news_context", {}),
        "regime_at_plan": regime,
    }


# ---------------------------------------------------------------------------
# Sector exposure check
# ---------------------------------------------------------------------------

def _apply_sector_cap(
    ranked: List[Dict[str, Any]],
    max_sector_pct: float,
    max_positions: int,
) -> List[Dict[str, Any]]:
    """Filter ranked list to respect sector concentration limits.

    Parameters
    ----------
    ranked : list[dict]
        Candidates sorted by score descending.
    max_sector_pct : float
        Maximum fraction of positions in any single sector (e.g. 0.30).
    max_positions : int
        Hard cap on total positions.

    Returns
    -------
    list[dict]
        Filtered list (at most max_positions entries).
    """
    selected: List[Dict[str, Any]] = []
    sector_counts: Dict[str, int] = {}

    for item in ranked:
        if len(selected) >= max_positions:
            break
        sector = str(item.get("sector", "")) or "UNKNOWN"
        count = sector_counts.get(sector, 0)
        # Cap: sector_count / max_positions <= max_sector_pct
        if count / max(max_positions, 1) >= max_sector_pct:
            continue
        selected.append(item)
        sector_counts[sector] = count + 1

    return selected


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def generate_premarket_plan(
    candidates: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    regime_dict: Optional[Dict[str, Any]] = None,
    analog_memory: Optional[Any] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a pre-market trade plan from a list of scored candidates.

    Parameters
    ----------
    candidates : list[dict]
        Each entry must have at minimum: ``ticker``, ``score``,
        ``side``, ``entry``, ``stop``.
    config : dict, optional
        Planning config with keys from :data:`_DEFAULT_CONFIG`.
    regime_dict : dict, optional
        Output of :func:`regime_engine.get_current_regime`.
    analog_memory : AnalogMarketMemory, optional
        For confidence calibration.
    seed : int
        Random seed (used for deterministic tie-breaking).

    Returns
    -------
    dict
        Plan with keys: ``plan_date``, ``as_of``, ``regime``, ``trades``,
        ``status``, ``output_path``.
    """
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    as_of = datetime.now(tz=timezone.utc)
    plan_date = as_of.strftime("%Y-%m-%d")

    regime = "CHOPPY"
    regime_conf = 0.5
    if regime_dict:
        regime = regime_dict.get("label", "CHOPPY")
        regime_conf = regime_dict.get("confidence", 0.5)

    buy_threshold = float(cfg.get("thresholds", {}).get("buy", 0.55))
    max_positions = int(cfg.get("max_positions", 10))
    max_sector_exp = float(cfg.get("max_sector_exposure", 0.30))
    equity = float(cfg.get("equity_base", 1_000_000.0))
    risk_pct = float(cfg.get("risk_per_trade_pct", 0.01))

    use_analog = analog_memory is not None

    # Score and filter
    scored: List[Dict[str, Any]] = []
    for cand in candidates:
        result = _score_candidate(
            cand,
            regime=regime,
            buy_threshold=buy_threshold,
            use_analog=use_analog,
            analog_memory=analog_memory,
        )
        if result is not None:
            scored.append(result)

    # Sort descending by confidence then score
    scored.sort(
        key=lambda x: (x["confidence"], x["score"]),
        reverse=True,
    )

    # Apply sector caps and position limit
    selected = _apply_sector_cap(scored, max_sector_exp, max_positions)

    # Size positions
    trades: List[Dict[str, Any]] = []
    for rank, item in enumerate(selected, start=1):
        entry = item.get("entry")
        stop = item.get("stop")
        shares = 0
        if entry is not None and stop is not None:
            shares = _size_position(equity, risk_pct, float(entry), float(stop))

        trades.append(
            {
                "rank": rank,
                "ticker": item["ticker"],
                "side": item["side"],
                "score": item["score"],
                "confidence": item["confidence"],
                "entry": entry,
                "stop": stop,
                "target": item.get("target"),
                "shares": shares,
                "sector": item.get("sector", ""),
                "regime_at_plan": item.get("regime_at_plan", regime),
                "news_context": item.get("news_context", {}),
            }
        )

    plan: Dict[str, Any] = {
        "plan_date": plan_date,
        "as_of": as_of.isoformat(),
        "regime": {"label": regime, "confidence": regime_conf},
        "equity_base": equity,
        "candidates_evaluated": len(candidates),
        "candidates_passed_filter": len(scored),
        "trades": trades,
        "status": "generated",
    }

    # Write to disk
    output_dir = str(cfg.get("output_dir", "reports/plans"))
    output_path = _write_plan(plan, output_dir, as_of)
    plan["output_path"] = output_path

    logger.info(
        "Pre-market plan generated: %d trades, regime=%s, path=%s",
        len(trades),
        regime,
        output_path,
    )
    return plan


def _write_plan(
    plan: Dict[str, Any],
    output_dir: str,
    as_of: datetime,
) -> str:
    """Persist plan JSON; return output path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"premarket_plan_{as_of.strftime('%Y%m%d_%H%M%S')}.json"
    out_path = out_dir / fname
    with open(out_path, "w") as fh:
        json.dump(plan, fh, indent=2, default=str)
    return str(out_path)
