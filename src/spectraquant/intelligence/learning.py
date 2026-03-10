"""Nightly learning update — outcome computation and model refresh."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def nightly_update(config: Optional[Any] = None) -> Dict[str, Any]:
    """Run the nightly learning cycle.

    Steps (production path):
    1. Compute trade outcomes (PnL, MAE, MFE, holding time).
    2. Label failures for the feedback loop.
    3. Update feature importances.
    4. Optionally trigger model retraining.

    Parameters
    ----------
    config : IntelligenceConfig, optional
        Loaded automatically when *None*.

    Returns
    -------
    dict
        Summary with keys: ``as_of``, ``outcomes_computed``,
        ``failures_labelled``, ``retrain_triggered``, ``status``.
    """
    if config is None:
        from spectraquant.intelligence.config import load_config
        config = load_config()

    # AS-OF timestamp — never peek at tomorrow
    as_of = datetime.now(tz=timezone.utc)

    logger.info("Nightly update starting at %s", as_of.isoformat())

    outcomes_computed = 0
    failures_labelled = 0
    retrain_triggered = False

    # In production this would:
    # 1. Query today's fills → compute PnL, MAE, MFE
    # 2. Label failures (stop-hit, time-decay, missed-entry)
    # 3. Check if retrain criteria are met
    # 4. Kick off retrain if needed

    result: Dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "outcomes_computed": outcomes_computed,
        "failures_labelled": failures_labelled,
        "retrain_triggered": retrain_triggered,
        "status": "ok",
    }

    logger.info(
        "Nightly update done: %d outcomes, %d failures, retrain=%s",
        outcomes_computed,
        failures_labelled,
        retrain_triggered,
    )
    return result
