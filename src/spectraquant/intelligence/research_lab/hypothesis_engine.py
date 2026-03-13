"""Hypothesis Engine — detect where current intelligence fails.

Observes prediction errors, failed trades, regime shifts, and news surprise
events to generate structured hypotheses for further investigation.

Output schema
-------------
Each :class:`Hypothesis` has:
  - hypothesis_id   : unique short hash-derived string
  - trigger_reason  : what triggered this hypothesis
  - affected_regime : market regime context
  - suggested_feature_change : human-readable remediation suggestion
  - timestamp       : ISO-8601 UTC
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A single structured hypothesis."""

    hypothesis_id: str
    trigger_reason: str
    affected_regime: str
    suggested_feature_change: str
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Hypothesis":
        return cls(
            hypothesis_id=d["hypothesis_id"],
            trigger_reason=d["trigger_reason"],
            affected_regime=d["affected_regime"],
            suggested_feature_change=d["suggested_feature_change"],
            timestamp=d.get("timestamp", ""),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Hypothesis generation rules
# ---------------------------------------------------------------------------

_RULES: List[Dict[str, Any]] = [
    {
        "condition": lambda m: m.get("failure_rate", 0.0) > 0.50,
        "trigger": "High failure rate (>{:.0%}): prediction quality degraded".format(0.50),
        "feature_change": "Review feature staleness; retrain with recent data or add volatility filter",
    },
    {
        "condition": lambda m: m.get("regime_failures", {}).get("TRENDING", 0) > 3,
        "trigger": "Momentum signals failing during TRENDING regime",
        "feature_change": "Add trend-strength filter; reduce momentum weight in strong-trend environment",
    },
    {
        "condition": lambda m: m.get("regime_failures", {}).get("PANIC", 0) > 2,
        "trigger": "Strategy failing during PANIC regime",
        "feature_change": "Disable momentum; increase volatility-scaling; add defensive regime gate",
    },
    {
        "condition": lambda m: m.get("news_shock_count", 0) > 3,
        "trigger": "Repeated NEWS_SHOCK failures",
        "feature_change": "Weight news sentiment higher; add pre-announcement hold-off rule",
    },
    {
        "condition": lambda m: m.get("overconfidence_count", 0) > 5,
        "trigger": "Overconfidence in predictions",
        "feature_change": "Apply confidence calibration; tighten buy threshold",
    },
    {
        "condition": lambda m: m.get("regime_shift_count", 0) > 3,
        "trigger": "Repeated REGIME_SHIFT failures",
        "feature_change": "Add regime-change detector; shorten holding horizon on regime uncertainty",
    },
    {
        "condition": lambda m: m.get("avg_slippage_bps", 0.0) > 30,
        "trigger": "Excessive slippage degrading returns",
        "feature_change": "Tighten liquidity filter; reduce position size in low-liquidity assets",
    },
    {
        "condition": lambda m: m.get("out_of_sample_sharpe", 0.0) < 0.5
                               and m.get("in_sample_sharpe", 0.0) > 1.5,
        "trigger": "Large in-sample vs out-of-sample Sharpe gap (overfit suspected)",
        "feature_change": "Reduce feature count; add regularisation; use shorter lookback windows",
    },
]


def _stable_id(trigger: str, regime: str) -> str:
    """Generate a deterministic, short ID from trigger + regime."""
    raw = f"{trigger}::{regime}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------

class HypothesisEngine:
    """Generate hypotheses from performance metrics and failure data.

    Parameters
    ----------
    memory_path : str
        Path to ``research_memory.json`` (used to deduplicate).
    """

    def __init__(self, memory_path: str = "data/intelligence/research_memory.json") -> None:
        self._memory_path = memory_path
        self._seen_ids: set[str] = set()
        self._load_seen_ids()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, metrics: Dict[str, Any]) -> List[Hypothesis]:
        """Generate new (non-duplicate) hypotheses from ``metrics``.

        Parameters
        ----------
        metrics : dict
            Keys include: ``failure_rate``, ``regime_failures``,
            ``news_shock_count``, ``overconfidence_count``,
            ``regime_shift_count``, ``avg_slippage_bps``,
            ``in_sample_sharpe``, ``out_of_sample_sharpe``.

        Returns
        -------
        list[Hypothesis]
            Only hypotheses not already in research memory.
        """
        new_hypotheses: List[Hypothesis] = []
        affected_regime = metrics.get("dominant_regime", "UNKNOWN")

        for rule in _RULES:
            try:
                if not rule["condition"](metrics):
                    continue
            except Exception:  # noqa: BLE001
                continue

            trigger = rule["trigger"]
            hid = _stable_id(trigger, affected_regime)
            if hid in self._seen_ids:
                logger.debug("Skipping duplicate hypothesis %s", hid)
                continue

            h = Hypothesis(
                hypothesis_id=hid,
                trigger_reason=trigger,
                affected_regime=affected_regime,
                suggested_feature_change=rule["feature_change"],
                metadata={"source_metrics": {k: v for k, v in metrics.items()
                                              if isinstance(v, (int, float, str))}},
            )
            new_hypotheses.append(h)
            self._seen_ids.add(hid)
            logger.info("New hypothesis generated: %s — %s", hid, trigger)

        return new_hypotheses

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_seen_ids(self) -> None:
        """Populate seen IDs from memory file (if it exists)."""
        path = self._memory_path
        if not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            for h in data.get("hypotheses", []):
                hid = h.get("hypothesis_id")
                if hid:
                    self._seen_ids.add(hid)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load hypothesis IDs from memory: %s", exc)
