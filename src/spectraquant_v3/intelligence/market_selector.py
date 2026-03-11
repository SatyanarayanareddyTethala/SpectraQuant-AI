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
# Adapted from the V2 event_ontology.py taxonomy (Phases 1–6 audit, §4 Idea 2)
# and extended with the full V3 taxonomy from the design spec.
# Values represent the weight multiplier applied to a record's contribution to
# the per-asset-class score.  All values are in [0.0, 1.0].
# Keys are lower-case; incoming event_type values are lowercased before lookup.
# ---------------------------------------------------------------------------

EVENT_ASSET_AFFINITY: dict[str, dict[str, float]] = {
    # Equity-dominant events
    "earnings":              {"equity": 1.00, "crypto": 0.05},
    "guidance":              {"equity": 0.95, "crypto": 0.05},
    "m_and_a":               {"equity": 0.90, "crypto": 0.10},
    "dividend":              {"equity": 0.85, "crypto": 0.00},
    "corporate_action":      {"equity": 0.80, "crypto": 0.05},
    "analyst":               {"equity": 0.75, "crypto": 0.10},
    "sector_theme":          {"equity": 0.70, "crypto": 0.55},
    "operations_disruption": {"equity": 0.70, "crypto": 0.20},
    # Cross-asset events (macro affects both; crypto reacts more to rates)
    "macro":                 {"equity": 0.60, "crypto": 0.80},
    "risk":                  {"equity": 0.60, "crypto": 0.50},
    # Mixed / cross-asset
    "regulatory":            {"equity": 0.50, "crypto": 0.85},
    "social_buzz":           {"equity": 0.20, "crypto": 0.45},
    # Crypto-dominant events
    "listing":               {"equity": 0.10, "crypto": 0.90},
    "security_incident":     {"equity": 0.30, "crypto": 0.75},
    "protocol_upgrade":      {"equity": 0.05, "crypto": 0.95},
    "exchange_hack":         {"equity": 0.00, "crypto": 1.00},
    "onchain":               {"equity": 0.00, "crypto": 0.95},
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
# Typed configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class MarketSelectorConfig:
    """Typed configuration for :class:`MarketSelector`.

    Attributes:
        min_score_to_run: Minimum per-asset-class score ``[0.0, 1.0]`` for a
            class to be eligible for routing.  Default ``0.35``.
        both_threshold:   Score above which both classes are strong enough to
            run simultaneously (bypass the 1.5× dominance rule).  Default ``0.60``.
        half_life_hours:  Exponential recency half-life in hours.  Must be > 0.
            Default ``6.0``.
        top_n:            Maximum contributing records to surface per asset
            class.  Default ``5``.
    """

    min_score_to_run: float = 0.35
    both_threshold: float = 0.60
    half_life_hours: float = 6.0
    top_n: int = 5

    def __post_init__(self) -> None:
        if self.half_life_hours <= 0.0:
            raise ValueError(
                f"half_life_hours must be > 0, got {self.half_life_hours!r}"
            )
        if self.top_n < 0:
            raise ValueError(f"top_n must be >= 0, got {self.top_n!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarketSelectorConfig:
        """Create a :class:`MarketSelectorConfig` from a plain dict."""
        return cls(
            min_score_to_run=float(data.get("min_score_to_run", 0.35)),
            both_threshold=float(data.get("both_threshold", 0.60)),
            half_life_hours=float(data.get("half_life_hours", 6.0)),
            top_n=int(data.get("top_n", 5)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "min_score_to_run": self.min_score_to_run,
            "both_threshold": self.both_threshold,
            "half_life_hours": self.half_life_hours,
            "top_n": self.top_n,
        }


# ---------------------------------------------------------------------------
# Typed input dataclass
# ---------------------------------------------------------------------------


@dataclass
class MarketSelectorInput:
    """Typed input bundle for :meth:`MarketSelector.score_input`.

    Attributes:
        records:      Provider-agnostic news intelligence records (may be empty).
        regime_label: Market regime string, e.g. ``"PANIC"``, ``"RISK_OFF"``,
            ``"EVENT_DRIVEN"``.  Case-insensitive; unknown values are neutral.
        as_of_utc:    Optional ISO-8601 UTC timestamp marking the logical
            evaluation time.  Used for backtest reproducibility; does not
            affect recency decay (which always uses wall-clock time).
    """

    records: list[NewsIntelligenceRecord] = field(default_factory=list)
    regime_label: str = "UNKNOWN"
    as_of_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (records are serialised via their own to_dict)."""
        return {
            "records": [r.to_dict() for r in self.records],
            "regime_label": self.regime_label,
            "as_of_utc": self.as_of_utc,
        }


# ---------------------------------------------------------------------------
# Dataclass: ScoredRecord and MarketSelectorDecision
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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "canonical_symbol": self.canonical_symbol,
            "event_type": self.event_type,
            "asset": self.asset,
            "raw_weight": self.raw_weight,
        }


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
        risk_off_penalty_applied: ``True`` when RISK_OFF reduced scores.
        threshold_used:      The ``min_score_to_run`` threshold applied.
        both_threshold_used: The ``both_threshold`` applied.
        rationale:           Human-readable score breakdown.
        scored_at:           ISO-8601 UTC timestamp of the scoring call.
        top_equity_records:  Up to ``top_n`` highest-contributing equity
            records for explainability.
        top_crypto_records:  Up to ``top_n`` highest-contributing crypto
            records for explainability.
        version:             Selector version string (``"v1"``).
    """

    route: MarketRoute
    equity_score: float
    crypto_score: float
    equity_record_count: int
    crypto_record_count: int
    record_count: int
    regime_label: str
    regime_vetoed: bool
    risk_off_penalty_applied: bool
    threshold_used: float
    both_threshold_used: float
    rationale: str
    scored_at: str
    top_equity_records: list[ScoredRecord] = field(default_factory=list)
    top_crypto_records: list[ScoredRecord] = field(default_factory=list)
    version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON export.

        All fields are included; enum values are converted to their string
        representation for JSON compatibility.
        """
        return {
            "route": self.route.value,
            "equity_score": self.equity_score,
            "crypto_score": self.crypto_score,
            "equity_record_count": self.equity_record_count,
            "crypto_record_count": self.crypto_record_count,
            "record_count": self.record_count,
            "regime_label": self.regime_label,
            "regime_vetoed": self.regime_vetoed,
            "risk_off_penalty_applied": self.risk_off_penalty_applied,
            "threshold_used": self.threshold_used,
            "both_threshold_used": self.both_threshold_used,
            "rationale": self.rationale,
            "scored_at": self.scored_at,
            "top_equity_records": [r.to_dict() for r in self.top_equity_records],
            "top_crypto_records": [r.to_dict() for r in self.top_crypto_records],
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# MarketSelector
# ---------------------------------------------------------------------------


class MarketSelector:
    """Deterministic news-first market selector.

    Parameters
    ----------
    config:
        Optional configuration.  Accepts either a :class:`MarketSelectorConfig`
        instance or a plain ``dict`` with the same keys (for backward
        compatibility).  Recognised dict keys:

        ``min_score_to_run`` (float, default ``0.35``)
            Minimum per-asset-class score required for that asset class to be
            considered for routing.

        ``both_threshold`` (float, default ``0.60``)
            Score above which both asset classes are considered strong enough
            to run simultaneously without applying the 1.5× dominance rule.

        ``half_life_hours`` (float, default ``6.0``)
            Half-life in hours for the exponential recency decay:
            ``recency(t) = exp(-ln(2) / half_life_hours * age_hours)``.

        ``top_n`` (int, default ``5``)
            Maximum number of scored records to include in
            :attr:`MarketSelectorDecision.top_equity_records` and
            :attr:`MarketSelectorDecision.top_crypto_records`.
    """

    def __init__(
        self,
        config: MarketSelectorConfig | dict[str, Any] | None = None,
    ) -> None:
        if isinstance(config, MarketSelectorConfig):
            cfg_obj = config
        else:
            cfg_obj = MarketSelectorConfig.from_dict(config or {})

        self._min_score: float = cfg_obj.min_score_to_run
        self._both_threshold: float = cfg_obj.both_threshold
        self._half_life_hours: float = cfg_obj.half_life_hours
        self._top_n: int = cfg_obj.top_n

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
            ``"EVENT_DRIVEN"``.  Case-insensitive; unknown values are neutral.

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
        risk_off_penalty_applied = regime_upper == "RISK_OFF"

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
            risk_off_penalty_applied=risk_off_penalty_applied,
            threshold_used=self._min_score,
            both_threshold_used=self._both_threshold,
            rationale=rationale,
            scored_at=scored_at,
            top_equity_records=top_equity,
            top_crypto_records=top_crypto,
        )

    def score_input(self, selector_input: MarketSelectorInput) -> MarketSelectorDecision:
        """Score a :class:`MarketSelectorInput` bundle and return a routing decision.

        Convenience wrapper around :meth:`score` that accepts the typed input
        model directly.

        Parameters
        ----------
        selector_input:
            Typed input bundle containing records and regime label.

        Returns
        -------
        MarketSelectorDecision
            A fully populated decision with scores, route, rationale, and
            contributing records.
        """
        return self.score(
            selector_input.records,
            regime_label=selector_input.regime_label,
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
        """Return the asset-class affinity multiplier for *event_type*.

        Event type lookup is case-insensitive; values are lowercased before
        table lookup so both ``"EARNINGS"`` and ``"earnings"`` work correctly.
        """
        entry = EVENT_ASSET_AFFINITY.get(event_type.lower(), _DEFAULT_AFFINITY)
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
                    * affinity(event_type_i, asset_class) * confidence_i
                    * sentiment_factor_i``

            where ``sentiment_factor = 0.5 + 0.5 * abs(sentiment_score)``

            ``score = sum(w_i) / max(N, 1)``  clipped to ``[0.0, 1.0]``
        """
        if not records:
            return [], 0.0

        weights: list[float] = []
        for rec in records:
            sentiment_factor = 0.5 + 0.5 * abs(rec.sentiment_score)
            w = (
                self._recency_weight(rec.timestamp)
                * rec.impact_score
                * self._affinity(rec.event_type, asset_class)
                * rec.confidence
                * sentiment_factor
            )
            weights.append(w)

        score = sum(weights) / max(len(weights), 1)
        score = max(0.0, min(1.0, score))
        return weights, score

    def _decide(self, equity_score: float, crypto_score: float) -> MarketRoute:
        """Apply routing thresholds and return a :class:`MarketRoute`.

        Logic (from audit §5.6):
        - Both above ``both_threshold``  → ``RUN_BOTH``
        - Both above ``min_score`` but neither dominates by 1.5×  → ``RUN_BOTH``
        - Both above ``min_score``, one dominates by 1.5×  → that asset only
        - Only equity above threshold  → ``RUN_EQUITIES``
        - Only crypto above threshold  → ``RUN_CRYPTO``
        - Neither above threshold  → ``RUN_NONE``
        """
        min_s = self._min_score
        both_t = self._both_threshold

        if equity_score >= both_t and crypto_score >= both_t:
            return MarketRoute.RUN_BOTH

        if equity_score >= min_s and crypto_score >= min_s:
            if equity_score >= crypto_score * 1.5:
                return MarketRoute.RUN_EQUITIES
            if crypto_score >= equity_score * 1.5:
                return MarketRoute.RUN_CRYPTO
            return MarketRoute.RUN_BOTH

        if equity_score >= min_s:
            return MarketRoute.RUN_EQUITIES

        if crypto_score >= min_s:
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
