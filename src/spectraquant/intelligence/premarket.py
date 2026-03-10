"""Pre-market trading plan generation."""
from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def premarket_plan(
    config: Optional[Any] = None,
    simulation: bool = True,
) -> Dict[str, Any]:
    """Generate the daily pre-market trading plan.

    Parameters
    ----------
    config : IntelligenceConfig, optional
        If *None* the default configuration is loaded.
    simulation : bool
        When *True* (default), use synthetic / historical data instead of
        live market feeds.

    Returns
    -------
    dict
        Plan metadata with keys: ``plan_date``, ``as_of``, ``simulation``,
        ``trades``, ``status``.
    """
    if config is None:
        from spectraquant.intelligence.config import load_config
        config = load_config()

    # AS-OF timestamp — strictly "now", never a future date
    as_of = datetime.now(tz=timezone.utc)
    plan_date = as_of.strftime("%Y-%m-%d")

    logger.info("Generating premarket plan for %s (simulation=%s)", plan_date, simulation)

    risk_cfg = config.risk
    sim_cfg = config.simulation

    # In simulation mode generate synthetic trade ideas
    use_sim = simulation or sim_cfg.enabled
    trades: List[Dict[str, Any]] = []

    if use_sim:
        rng = random.Random(sim_cfg.seed)
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        for sym in symbols:
            entry = round(rng.uniform(100, 500), 2)
            risk_per_share = round(entry * rng.uniform(0.01, 0.03), 2)
            stop_loss = round(entry - risk_per_share, 2)
            target = round(entry + risk_per_share * rng.uniform(1.5, 3.0), 2)
            trades.append(
                {
                    "symbol": sym,
                    "direction": "long",
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "target": target,
                    "risk_per_share": risk_per_share,
                    "trigger_json": {"price": {"target": entry}},
                }
            )
    else:
        logger.info("Live data path — no trades generated (stub)")

    plan: Dict[str, Any] = {
        "plan_date": plan_date,
        "as_of": as_of.isoformat(),
        "simulation": use_sim,
        "equity_base": risk_cfg.equity_base,
        "risk_fraction": risk_cfg.alpha_risk_fraction,
        "trades": trades,
        "status": "generated",
    }

    logger.info("Premarket plan ready: %d trades", len(trades))
    return plan
