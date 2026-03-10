"""Bootstrap helper — initialises the database and runs first-time setup."""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def run_bootstrap() -> Dict[str, Any]:
    """Programmatically execute the bootstrap sequence.

    Steps:
    1. Load configuration.
    2. Initialise database tables (create-all).
    3. Generate the first pre-market plan (simulation mode).

    Returns
    -------
    dict
        Bootstrap results with ``config_loaded``, ``db_initialised``,
        ``first_plan`` keys.
    """
    from spectraquant.intelligence.config import load_config
    from spectraquant.intelligence.db.session import init_db, get_engine
    from spectraquant.intelligence.db.models import Base
    from spectraquant.intelligence.premarket import premarket_plan

    logger.info("Bootstrap: loading configuration")
    config = load_config()

    logger.info("Bootstrap: initialising database")
    engine = get_engine(config.database.url)
    Base.metadata.create_all(bind=engine)

    logger.info("Bootstrap: generating first plan (simulation)")
    plan = premarket_plan(config=config, simulation=True)

    result: Dict[str, Any] = {
        "config_loaded": True,
        "db_initialised": True,
        "first_plan": plan,
    }
    logger.info("Bootstrap complete")
    return result
