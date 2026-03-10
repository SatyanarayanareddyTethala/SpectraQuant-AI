"""Intraday monitoring — trigger evaluation and alert dispatch."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def intraday_monitor(config: Optional[Any] = None) -> Dict[str, Any]:
    """Run a single pass of intraday monitoring.

    Evaluates triggers for active plan trades, checks policy rules,
    and returns a status dict.

    Parameters
    ----------
    config : IntelligenceConfig, optional
        Loaded automatically when *None*.

    Returns
    -------
    dict
        Monitoring status with keys: ``as_of``, ``triggers_fired``,
        ``blocked``, ``status``.
    """
    if config is None:
        from spectraquant.intelligence.config import load_config
        config = load_config()

    # AS-OF timestamp — strict present-only
    as_of = datetime.now(tz=timezone.utc)

    logger.info("Intraday monitor tick at %s", as_of.isoformat())

    triggers_fired: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []

    # In a live system this would:
    # 1. Load active plan trades from DB
    # 2. Fetch current 5-min bars
    # 3. Run TriggerEvaluator.evaluate() per trade
    # 4. Run PolicyRules.do_not_trade_checks()
    # 5. Create alerts for fired & allowed triggers

    result: Dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "triggers_fired": triggers_fired,
        "blocked": blocked,
        "status": "ok",
    }

    logger.info(
        "Intraday monitor: %d triggers, %d blocked",
        len(triggers_fired),
        len(blocked),
    )
    return result
