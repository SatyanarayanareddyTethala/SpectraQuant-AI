"""APScheduler-based task scheduler with FastAPI health endpoints.

Run directly::

    python -m spectraquant.intelligence.scheduler
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import apscheduler.schedulers.background as aps_bg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduler class
# ---------------------------------------------------------------------------

class IntelligenceScheduler:
    """Manages recurring intelligence jobs via APScheduler."""

    def __init__(self, config: Any = None) -> None:
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        if config is None:
            from spectraquant.intelligence.config import load_config
            config = load_config()

        self.config = config
        self.scheduler = aps_bg.BackgroundScheduler(timezone=config.market.timezone)

        sched_cfg = config.scheduler

        # Pre-market plan (default: Mon-Fri 08:28)
        self.scheduler.add_job(
            func=self._run_premarket,
            trigger=CronTrigger.from_crontab(
                sched_cfg.premarket_cron, timezone=config.market.timezone
            ),
            id="premarket_plan",
            name="Pre-market plan",
            replace_existing=True,
        )

        # Hourly news (default: Mon-Fri at :05 past)
        self.scheduler.add_job(
            func=self._run_hourly_news,
            trigger=CronTrigger.from_crontab(
                sched_cfg.hourly_news_cron, timezone=config.market.timezone
            ),
            id="hourly_news",
            name="Hourly news",
            replace_existing=True,
        )

        # Intraday monitor (default: every 60 s)
        self.scheduler.add_job(
            func=self._run_intraday,
            trigger=IntervalTrigger(seconds=sched_cfg.intraday_interval_seconds),
            id="intraday_monitor",
            name="Intraday monitor",
            replace_existing=True,
        )

        # Nightly update
        self.scheduler.add_job(
            func=self._run_nightly,
            trigger=CronTrigger.from_crontab(
                sched_cfg.nightly_cron, timezone=config.market.timezone
            ),
            id="nightly_update",
            name="Nightly update",
            replace_existing=True,
        )

        # Weekly retrain (Sunday)
        learn_cfg = config.learning
        self.scheduler.add_job(
            func=self._run_retrain,
            trigger=CronTrigger(
                day_of_week="sun",
                hour=learn_cfg.retrain_hour,
                timezone=config.market.timezone,
            ),
            id="weekly_retrain",
            name="Weekly retrain",
            replace_existing=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("Starting intelligence scheduler")
        self.scheduler.start()

    def stop(self) -> None:
        logger.info("Stopping intelligence scheduler")
        self.scheduler.shutdown(wait=False)

    def list_jobs(self) -> List[Dict[str, Any]]:
        jobs: List[Dict[str, Any]] = []
        for job in self.scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": (
                        job.next_run_time.isoformat() if job.next_run_time else None
                    ),
                }
            )
        return jobs

    # ------------------------------------------------------------------
    # Job wrappers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_premarket() -> None:
        from spectraquant.intelligence.premarket import premarket_plan
        premarket_plan()

    @staticmethod
    def _run_hourly_news() -> None:
        from spectraquant.intelligence.hourly_news import hourly_news
        hourly_news()

    @staticmethod
    def _run_intraday() -> None:
        from spectraquant.intelligence.intraday import intraday_monitor
        intraday_monitor()

    @staticmethod
    def _run_nightly() -> None:
        from spectraquant.intelligence.learning import nightly_update
        nightly_update()

    @staticmethod
    def _run_retrain() -> None:
        from spectraquant.intelligence.learning import nightly_update
        logger.info("Weekly retrain triggered")
        nightly_update()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app() -> "FastAPI":
    """Create a lightweight FastAPI app with health & scheduler endpoints."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    _state: Dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: "FastAPI"):  # type: ignore[name-defined]
        sched = IntelligenceScheduler()
        sched.start()
        _state["scheduler"] = sched
        yield
        sched.stop()

    app = FastAPI(title="SpectraQuant Intelligence Scheduler", lifespan=lifespan)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "utc": datetime.now(tz=timezone.utc).isoformat(),
        }

    @app.get("/scheduler/jobs")
    def scheduler_jobs() -> List[Dict[str, Any]]:
        sched = _state.get("scheduler")
        if sched is None:
            return []
        return sched.list_jobs()

    return app


# ---------------------------------------------------------------------------
# __main__ support
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
