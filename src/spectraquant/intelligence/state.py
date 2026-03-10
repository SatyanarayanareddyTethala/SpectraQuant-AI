"""Deduplication and cooldown management for alerts."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class DedupeManager:
    """Generates deterministic dedupe keys and checks for duplicates / cooldowns."""

    # ------------------------------------------------------------------
    # Key generators
    # ------------------------------------------------------------------

    @staticmethod
    def plan_key(date: datetime) -> str:
        """One plan per calendar day: ``PLAN:YYYYMMDD``."""
        return f"PLAN:{date.strftime('%Y%m%d')}"

    @staticmethod
    def news_key(plan_id: int, date: datetime) -> str:
        """One news digest per plan per hour: ``NEWS:{plan_id}:{YYYYMMDDHH}``."""
        return f"NEWS:{plan_id}:{date.strftime('%Y%m%d%H')}"

    @staticmethod
    def exec_key(plan_id: int, symbol: str, trigger_id: str) -> str:
        """One execution alert per plan / symbol / trigger:
        ``EXEC:{plan_id}:{symbol}:{trigger_id}``."""
        return f"EXEC:{plan_id}:{symbol}:{trigger_id}"

    # ------------------------------------------------------------------
    # Duplicate & cooldown checks
    # ------------------------------------------------------------------

    @staticmethod
    def is_duplicate(key: str, session: "Session") -> bool:  # noqa: F821
        """Return *True* if an alert with *key* already exists in the DB.

        Imports the Alert model lazily to avoid circular dependencies.
        """
        from spectraquant.intelligence.db.models import Alert

        existing = (
            session.query(Alert)
            .filter(Alert.dedupe_key == key)
            .first()
        )
        if existing is not None:
            logger.debug("Duplicate detected for key=%s", key)
            return True
        return False

    @staticmethod
    def cooldown_active(
        symbol: str,
        category: str,
        session: "Session",  # noqa: F821
        cooldown_seconds: int = 900,
    ) -> bool:
        """Return *True* if an alert for *symbol* / *category* was sent
        within the last *cooldown_seconds* (default 15 min)."""
        from spectraquant.intelligence.db.models import Alert

        now = datetime.now(tz=timezone.utc)
        cutoff = datetime.fromtimestamp(
            now.timestamp() - cooldown_seconds, tz=timezone.utc
        )

        recent = (
            session.query(Alert)
            .filter(
                Alert.symbol == symbol,
                Alert.category == category,
                Alert.created_at >= cutoff,
            )
            .first()
        )
        if recent is not None:
            logger.debug(
                "Cooldown active for %s/%s (last alert at %s)",
                symbol,
                category,
                recent.created_at,
            )
            return True
        return False
