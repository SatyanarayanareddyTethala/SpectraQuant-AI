"""News-first market selector for SpectraQuant-AI-V3.

Scores a list of :class:`~spectraquant_v3.core.news_schema.NewsIntelligenceRecord`
objects for equities and crypto separately and produces a deterministic
:class:`MarketSelectorDecision` that routes the system to run equities,
crypto, both, or neither.

Design constraints
------------------
* Pure function — no I/O, no network calls, no file-system access.
* Provider-agnostic — only uses :class:`~spectraquant_v3.core.news_schema.NewsIntelligenceRecord`
  fields; never calls any provider directly.
* Asset-class-safe — only touches ``record.asset``; no imports from
  ``spectraquant_v3.crypto`` or ``spectraquant_v3.equities``.
* V2-safe — zero imports from ``spectraquant``; lives entirely in V3.
* Deterministic — identical inputs always produce identical outputs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from spectraquant_v3.core.enums import MarketRoute
from spectraquant_v3.core.news_schema import NewsIntelligenceRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event-type → asset-class affinity table
# ---------------------------------------------------------------------------
# Each event type carries an implicit affinity toward equities or crypto.
# Adapted from the V2 event_ontology.py taxonomy (Phases 1–6 audit, §4 Idea 2).
# Values represent the weight multiplier applied to a record's contribution to
# the per-asset-class score.  All values are in [0.0, 1.0].
# ---------------------------------------------------------------------------

EVENT_ASSET_AFFINITY: dict[str, dict[str, float]] = {
    # Equity-dominant events
    "earnings":              {"equity": 1.00, "crypto": 0.05},
    "m_and_a":               {"equity": 0.90, "crypto": 0.10},
    "corporate_action":      {"equity": 0.80, "crypto": 0.05},
    "operations_disruption": {"equity": 0.70, "crypto": 0.20},
    # Cross-asset events (macro affects both; crypto reacts more to rates)
    "macro":                 {"equity": 0.60, "crypto": 0.80},
    "risk":                  {"equity": 0.60, "crypto": 0.50},
    # Crypto-dominant events
    "regulatory":            {"equity": 0.50, "crypto": 0.85},
    "listing":               {"equity": 0.10, "crypto": 0.90},
    "security_incident":     {"equity": 0.30, "crypto": 0.75},
    # Fallback for unrecognised event types
    "unknown":               {"equity": 0.50, "crypto": 0.50},
}

# Default affinity for event types not in the lookup table
_DEFAULT_AFFINITY: dict[str, float] = {"equity": 0.50, "crypto": 0.50}

# ---------------------------------------------------------------------------
# Regime multipliers
# ---------------------------------------------------------------------------
# Applied to both scores before routing thresholds are evaluated.
# PANIC is handled separately as an unconditional veto.

_REGIME_MULTIPLIERS: dict[str, float] = {
    "RISK_OFF": 0.5,
    "EVENT_DRIVEN": 1.2,
}

# Regimes that unconditionally veto all routing (force RUN_NONE).
_VETO_REGIMES: frozenset[str] = frozenset({"PANIC"})


# ---------------------------------------------------------------------------
# Dataclass: MarketSelectorDecision
# ---------------------------------------------------------------------------


@dataclass
class ScoredRecord:
    """Score contribution from a single :class:`NewsIntelligenceRecord`.

    Included in :attr:`MarketSelectorDecision.top_equity_records` and
    :attr:`MarketSelectorDecision.top_crypto_records` for explainability.
    """

    canonical_symbol: str
    event_type: str
    asset: str
    raw_weight: float
    """Unscaled per-record contribution before normalisation."""


@dataclass
class MarketSelectorDecision:
    """Output of :meth:`MarketSelector.score`.

    Attributes:
        route:               The routing decision.
        equity_score:        Aggregated equity news score in ``[0.0, 1.0]``.
        crypto_score:        Aggregated crypto news score in ``[0.0, 1.0]``.
        equity_record_count: Number of equity :class:`NewsIntelligenceRecord`
            inputs.
        crypto_record_count: Number of crypto :class:`NewsIntelligenceRecord`
            inputs.
        record_count:        Total :class:`NewsIntelligenceRecord` inputs.
        regime_label:        Regime string provided to the selector, or
            ``"UNKNOWN"`` if none was given.
        regime_vetoed:       ``True`` when the regime forced ``RUN_NONE``
            regardless of scores.
        threshold_used:      The effective opportunity-floor threshold that
            was actually compared to scores when deciding whether to run any
            market (after applying any overrides such as
            ``low_opportunity_floor``).
        both_threshold_used: The effective high-opportunity threshold that
            was actually compared to scores when deciding to run both
            markets (after applying any overrides such as
            ``high_opportunity_threshold``).
        thresholds:          Mapping of all named routing thresholds and
            their values that influenced this decision (including configured
            defaults and any per-call overrides).
        rationale:           Human-readable score breakdown.
        scored_at:           ISO-8601 UTC timestamp of the scoring call.
        top_equity_records:  Up to ``top_n`` highest-contributing equity
            records for explainability.
        top_crypto_records:  Up to ``top_n`` highest-contributing crypto
            records for explainability.
    """

    route: MarketRoute
    equity_score: float
    crypto_score: float
    equity_record_count: int
    crypto_record_count: int
    record_count: int
    regime_label: str
    regime_vetoed: bool
    threshold_used: float
    both_threshold_used: float
    rationale: str
    scored_at: str
    thresholds: dict[str, float] = field(default_factory=dict)
    top_equity_records: list[ScoredRecord] = field(default_factory=list)
    top_crypto_records: list[ScoredRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MarketSelector
# ---------------------------------------------------------------------------


class MarketSelector:
    """Deterministic news-first market selector.

    Parameters
    ----------
    config:
        Optional configuration overrides.  Recognised keys:

        ``min_score_to_run`` (float, default ``0.35``)
            Minimum per-asset-class score required for that asset class to be
            considered for routing.

        ``both_threshold`` (float, default ``0.60``)
            Legacy strong-signal threshold retained for backward compatibility.

        ``low_opportunity_floor`` (float, default value of
        ``min_score_to_run``)
            If both scores are below this floor, route to ``RUN_NONE``.

        ``high_opportunity_threshold`` (float, default value of
        ``both_threshold``)
            Score considered high-opportunity for each asset class.

        ``both_margin`` (float, default ``0.10``)
            Maximum absolute score difference for high-opportunity
            ``RUN_BOTH`` decisions.

        ``minimum_score_gap`` (float, default ``0.15``)
            Minimum score delta required to break ties and route to a single
            asset class.

        ``half_life_hours`` (float, default ``6.0``)
            Half-life in hours for the exponential recency decay:
            ``recency(t) = exp(-ln(2) / half_life_hours * age_hours)``.

        ``top_n`` (int, default ``5``)
            Maximum number of scored records to include in
            :attr:`MarketSelectorDecision.top_equity_records` and
            :attr:`MarketSelectorDecision.top_crypto_records`.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._min_score: float = float(cfg.get("min_score_to_run", 0.35))
        self._both_threshold: float = float(cfg.get("both_threshold", 0.60))
        self._low_opportunity_floor: float = float(
            cfg.get("low_opportunity_floor", self._min_score)
        )
        self._high_opportunity_threshold: float = float(
            cfg.get("high_opportunity_threshold", self._both_threshold)
        )
        self._both_margin: float = float(cfg.get("both_margin", 0.10))
        self._minimum_score_gap: float = float(cfg.get("minimum_score_gap", 0.15))
        self._half_life_hours: float = float(cfg.get("half_life_hours", 6.0))
        self._top_n: int = int(cfg.get("top_n", 5))

        if self._low_opportunity_floor < 0.0:
            raise ValueError(
                f"low_opportunity_floor must be >= 0, got {self._low_opportunity_floor!r}"
            )
        if self._high_opportunity_threshold < 0.0:
            raise ValueError(
                f"high_opportunity_threshold must be >= 0, got {self._high_opportunity_threshold!r}"
            )
        if self._low_opportunity_floor > self._high_opportunity_threshold:
            raise ValueError(
                f"low_opportunity_floor ({self._low_opportunity_floor!r}) must be "
                f"<= high_opportunity_threshold ({self._high_opportunity_threshold!r})"
            )
        if self._both_margin < 0.0:
            raise ValueError(
                f"both_margin must be >= 0, got {self._both_margin!r}"
            )
        if self._minimum_score_gap < 0.0:
            raise ValueError(
                f"minimum_score_gap must be >= 0, got {self._minimum_score_gap!r}"
            )
        if self._half_life_hours <= 0.0:
            raise ValueError(
                f"half_life_hours must be > 0, got {self._half_life_hours!r}"
            )
        if self._top_n < 0:
            raise ValueError(f"top_n must be >= 0, got {self._top_n!r}")

        # Pre-compute decay constant
        self._lambda: float = math.log(2.0) / self._half_life_hours

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        records: list[NewsIntelligenceRecord],
        *,
        regime_label: str = "UNKNOWN",
    ) -> MarketSelectorDecision:
        """Score *records* and return a routing decision.

        Parameters
        ----------
        records:
            Provider-agnostic news intelligence records.  May be empty.
        regime_label:
            Optional market regime string, e.g. ``"PANIC"``, ``"RISK_OFF"``,
            ``"EVENT_DRIVEN"``.  Case-sensitive (upper-case convention from
            V2 ``regime_engine.py``).  Unknown values are treated as neutral.

        Returns
        -------
        MarketSelectorDecision
            A fully populated decision with scores, route, rationale, and
            contributing records.
        """
        scored_at = datetime.now(tz=timezone.utc).isoformat()

        # 1. Partition by asset class
        equity_records = [r for r in records if r.asset == "equity"]
        crypto_records = [r for r in records if r.asset == "crypto"]

        # 2. Score each asset class
        equity_weights, equity_score_raw = self._score_records(equity_records, "equity")
        crypto_weights, crypto_score_raw = self._score_records(crypto_records, "crypto")

        # 3. Apply regime multipliers (before veto check)
        regime_upper = regime_label.upper()
        regime_vetoed = regime_upper in _VETO_REGIMES
        multiplier = _REGIME_MULTIPLIERS.get(regime_upper, 1.0)

        equity_score = min(equity_score_raw * multiplier, 1.0)
        crypto_score = min(crypto_score_raw * multiplier, 1.0)

        # 4. Regime veto → RUN_NONE regardless of scores
        if regime_vetoed:
            route = MarketRoute.RUN_NONE
        else:
            route = self._decide(equity_score, crypto_score)

        # 5. Build top-N contributing records for explainability
        top_equity = self._top_records(equity_records, equity_weights)
        top_crypto = self._top_records(crypto_records, crypto_weights)

        # 6. Build rationale string
        rationale = self._build_rationale(
            route=route,
            equity_score=equity_score,
            crypto_score=crypto_score,
            equity_records=equity_records,
            crypto_records=crypto_records,
            regime_label=regime_label,
            regime_vetoed=regime_vetoed,
            multiplier=multiplier,
        )

        return MarketSelectorDecision(
            route=route,
            equity_score=round(equity_score, 6),
            crypto_score=round(crypto_score, 6),
            equity_record_count=len(equity_records),
            crypto_record_count=len(crypto_records),
            record_count=len(records),
            regime_label=regime_label,
            regime_vetoed=regime_vetoed,
            threshold_used=self._min_score,
            both_threshold_used=self._both_threshold,
            thresholds={
                "low_opportunity_floor": self._low_opportunity_floor,
                "high_opportunity_threshold": self._high_opportunity_threshold,
                "both_margin": self._both_margin,
                "minimum_score_gap": self._minimum_score_gap,
            },
            rationale=rationale,
            scored_at=scored_at,
            top_equity_records=top_equity,
            top_crypto_records=top_crypto,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recency_weight(self, timestamp: str) -> float:
        """Return exponential recency decay weight for an ISO-8601 timestamp.

        ``recency(t) = exp(-lambda * age_hours)``

        Unknown or malformed timestamps are treated as zero age (weight 1.0)
        with a logged warning.
        """
        try:
            event_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            age_hours = max((now - event_time).total_seconds() / 3600.0, 0.0)
            return math.exp(-self._lambda * age_hours)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "MarketSelector: could not parse timestamp %r — treating as age 0: %s",
                timestamp,
                exc,
            )
            return 1.0

    def _affinity(self, event_type: str, asset_class: str) -> float:
        """Return the asset-class affinity multiplier for *event_type*."""
        entry = EVENT_ASSET_AFFINITY.get(event_type, _DEFAULT_AFFINITY)
        return entry.get(asset_class, 0.5)

    def _score_records(
        self,
        records: list[NewsIntelligenceRecord],
        asset_class: str,
    ) -> tuple[list[float], float]:
        """Compute a weighted aggregate score for *records*.

        Returns
        -------
        tuple[list[float], float]
            ``(per_record_weights, aggregate_score)``

            *per_record_weights* has the same length as *records* and contains
            each record's unscaled contribution (used for top-N selection).

            *aggregate_score* is in ``[0.0, 1.0]``.

        Scoring formula (per record):
            ``w_i = recency(timestamp_i) * impact_score_i
                    * affinity(event_type_i, asset_class) * confidence_i``

            ``score = sum(w_i) / max(N, 1)``  clipped to ``[0.0, 1.0]``
        """
        if not records:
            return [], 0.0

        weights: list[float] = []
        for rec in records:
            w = (
                self._recency_weight(rec.timestamp)
                * rec.impact_score
                * self._affinity(rec.event_type, asset_class)
                * rec.confidence
            )
            weights.append(w)

        score = sum(weights) / max(len(weights), 1)
        score = max(0.0, min(1.0, score))
        return weights, score

    def _decide(self, equity_score: float, crypto_score: float) -> MarketRoute:
        """Apply routing thresholds and return a :class:`MarketRoute`.

        Decision order:
        1. (PANIC veto is handled by ``score()`` before this method).
        2. Both below ``low_opportunity_floor`` -> ``RUN_NONE``.
        3. Both above ``high_opportunity_threshold`` and within
           ``both_margin`` -> ``RUN_BOTH``.
        4. One side above ``high_opportunity_threshold`` and ahead by at least
           ``minimum_score_gap`` -> route only that side.
        5. Intermediate tie / near-tie handling: if one side leads by at least
           ``minimum_score_gap``, route that side, otherwise ``RUN_NONE``.
        6. Fallback -> ``RUN_NONE``.
        """
        low_floor = self._low_opportunity_floor
        high_threshold = self._high_opportunity_threshold
        both_margin = self._both_margin
        min_gap = self._minimum_score_gap
        gap = abs(equity_score - crypto_score)

        if equity_score < low_floor and crypto_score < low_floor:
            return MarketRoute.RUN_NONE

        if (
            equity_score >= high_threshold
            and crypto_score >= high_threshold
            and gap <= both_margin
        ):
            return MarketRoute.RUN_BOTH

        if equity_score >= high_threshold and (equity_score - crypto_score) >= min_gap:
            return MarketRoute.RUN_EQUITIES

        if crypto_score >= high_threshold and (crypto_score - equity_score) >= min_gap:
            return MarketRoute.RUN_CRYPTO

        if gap >= min_gap:
            if equity_score > crypto_score:
                return MarketRoute.RUN_EQUITIES
            if crypto_score > equity_score:
                return MarketRoute.RUN_CRYPTO

        return MarketRoute.RUN_NONE

    def _top_records(
        self,
        records: list[NewsIntelligenceRecord],
        weights: list[float],
    ) -> list[ScoredRecord]:
        """Return up to ``top_n`` :class:`ScoredRecord` sorted by weight desc."""
        if not records:
            return []
        paired = sorted(zip(weights, records), key=lambda x: x[0], reverse=True)
        return [
            ScoredRecord(
                canonical_symbol=rec.canonical_symbol,
                event_type=rec.event_type,
                asset=rec.asset,
                raw_weight=round(w, 6),
            )
            for w, rec in paired[: self._top_n]
        ]

    @staticmethod
    def _build_rationale(
        *,
        route: MarketRoute,
        equity_score: float,
        crypto_score: float,
        equity_records: list[NewsIntelligenceRecord],
        crypto_records: list[NewsIntelligenceRecord],
        regime_label: str,
        regime_vetoed: bool,
        multiplier: float,
    ) -> str:
        """Construct a deterministic, human-readable rationale string."""
        parts: list[str] = [f"route={route.value}"]
        parts.append(
            f"equity_score={equity_score:.4f} "
            f"({len(equity_records)} record(s))"
        )
        parts.append(
            f"crypto_score={crypto_score:.4f} "
            f"({len(crypto_records)} record(s))"
        )
        parts.append(f"regime={regime_label}")
        if regime_vetoed:
            parts.append("regime_vetoed=True (PANIC → RUN_NONE)")
        elif multiplier != 1.0:
            parts.append(f"regime_multiplier={multiplier:.2f}")
        return "; ".join(parts)
