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
# Each canonical event type carries an implicit affinity toward equities or
# crypto. Keys are intentionally UPPERCASE so lookup can canonicalize all
# inbound event_type values to a deterministic representation.
# Adapted from the V2 event_ontology.py taxonomy (Phases 1–6 audit, §4 Idea 2).
# Values represent the weight multiplier applied to a record's contribution to
# the per-asset-class score.  All values are in [0.0, 1.0].
# ---------------------------------------------------------------------------

EVENT_ASSET_AFFINITY: dict[str, dict[str, float]] = {
    # Equity-dominant events
    "EARNINGS":              {"equity": 1.00, "crypto": 0.05},
    "M_AND_A":               {"equity": 0.90, "crypto": 0.10},
    "CORPORATE_ACTION":      {"equity": 0.80, "crypto": 0.05},
    "OPERATIONS_DISRUPTION": {"equity": 0.70, "crypto": 0.20},
    # Cross-asset events (macro affects both; crypto reacts more to rates)
    "MACRO":                 {"equity": 0.60, "crypto": 0.80},
    "RISK":                  {"equity": 0.60, "crypto": 0.50},
    # Crypto-dominant events
    "REGULATORY":            {"equity": 0.50, "crypto": 0.85},
    "LISTING":               {"equity": 0.10, "crypto": 0.90},
    "SECURITY_INCIDENT":     {"equity": 0.30, "crypto": 0.75},
    # Fallback for unrecognised event types
    "UNKNOWN":               {"equity": 0.50, "crypto": 0.50},
}

# Default affinity for event types not in the lookup table
_DEFAULT_AFFINITY: dict[str, float] = {"equity": 0.50, "crypto": 0.50}

# Explicit aliases observed in provider payloads and historical datasets.
# All aliases must resolve to a canonical key in EVENT_ASSET_AFFINITY.
_EVENT_TYPE_ALIASES: dict[str, str] = {
    "M&A": "M_AND_A",
    "MERGERS_AND_ACQUISITIONS": "M_AND_A",
    "REGULATION": "REGULATORY",
    "GENERAL": "UNKNOWN",
}

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_symbol": self.canonical_symbol,
            "event_type": self.event_type,
            "asset": self.asset,
            "raw_weight": round(float(self.raw_weight), 6),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoredRecord:
        return cls(
            canonical_symbol=str(payload.get("canonical_symbol", "")),
            event_type=str(payload.get("event_type", "UNKNOWN")),
            asset=str(payload.get("asset", "unknown")),
            raw_weight=float(payload.get("raw_weight", 0.0)),
        )


@dataclass(frozen=True)
class MarketSelectorConfig:
    """Configuration for deterministic market selection."""

    min_score_to_run: float = 0.35
    both_threshold: float = 0.60
    half_life_hours: float = 6.0
    top_n: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_score_to_run": float(self.min_score_to_run),
            "both_threshold": float(self.both_threshold),
            "half_life_hours": float(self.half_life_hours),
            "top_n": int(self.top_n),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> MarketSelectorConfig:
        p = payload or {}
        return cls(
            min_score_to_run=float(p.get("min_score_to_run", 0.35)),
            both_threshold=float(p.get("both_threshold", 0.60)),
            half_life_hours=float(p.get("half_life_hours", 6.0)),
            top_n=int(p.get("top_n", 5)),
        )


@dataclass(frozen=True)
class MarketSelectorInput:
    """Input envelope for :meth:`MarketSelector.score_input`."""

    records: list[NewsIntelligenceRecord]
    regime_label: str = "UNKNOWN"
    as_of_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.model_dump() for record in self.records],
            "regime_label": self.regime_label,
            "as_of_utc": self.as_of_utc,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MarketSelectorInput:
        raw_records = payload.get("records", [])
        return cls(
            records=[NewsIntelligenceRecord.model_validate(rec) for rec in raw_records],
            regime_label=str(payload.get("regime_label", "UNKNOWN")),
            as_of_utc=payload.get("as_of_utc"),
        )


@dataclass(frozen=True)
class ScoreBreakdown:
    equity: float
    crypto: float
    equity_record_count: int
    crypto_record_count: int
    record_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "equity": round(float(self.equity), 6),
            "crypto": round(float(self.crypto), 6),
            "equity_record_count": int(self.equity_record_count),
            "crypto_record_count": int(self.crypto_record_count),
            "record_count": int(self.record_count),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreBreakdown:
        return cls(
            equity=float(payload.get("equity", 0.0)),
            crypto=float(payload.get("crypto", 0.0)),
            equity_record_count=int(payload.get("equity_record_count", 0)),
            crypto_record_count=int(payload.get("crypto_record_count", 0)),
            record_count=int(payload.get("record_count", 0)),
        )


@dataclass
class ContributingEventSummary:
    top_equity_records: list[ScoredRecord] = field(default_factory=list)
    top_crypto_records: list[ScoredRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_equity_records": [record.to_dict() for record in self.top_equity_records],
            "top_crypto_records": [record.to_dict() for record in self.top_crypto_records],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ContributingEventSummary:
        return cls(
            top_equity_records=[
                ScoredRecord.from_dict(record)
                for record in payload.get("top_equity_records", [])
            ],
            top_crypto_records=[
                ScoredRecord.from_dict(record)
                for record in payload.get("top_crypto_records", [])
            ],
        )


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

    as_of_utc: str
    decision: MarketRoute
    scores: ScoreBreakdown
    thresholds: dict[str, float] = field(default_factory=dict)
    regimes: dict[str, Any] = field(default_factory=dict)
    veto_flags: dict[str, bool] = field(default_factory=dict)
    rationale: str = ""
    contributing_events: ContributingEventSummary = field(default_factory=ContributingEventSummary)
    version: str = "v1"

    @property
    def route(self) -> MarketRoute:
        return self.decision

    @property
    def equity_score(self) -> float:
        return self.scores.equity

    @property
    def crypto_score(self) -> float:
        return self.scores.crypto

    @property
    def equity_record_count(self) -> int:
        return self.scores.equity_record_count

    @property
    def crypto_record_count(self) -> int:
        return self.scores.crypto_record_count

    @property
    def record_count(self) -> int:
        return self.scores.record_count

    @property
    def regime_label(self) -> str:
        return str(self.regimes.get("label", "UNKNOWN"))

    @property
    def regime_vetoed(self) -> bool:
        return bool(self.veto_flags.get("regime_vetoed", False))

    @property
    def threshold_used(self) -> float:
        return float(self.thresholds.get("min_score_to_run", 0.0))

    @property
    def both_threshold_used(self) -> float:
        return float(self.thresholds.get("both_threshold", 0.0))

    @property
    def scored_at(self) -> str:
        return self.as_of_utc

    @property
    def top_equity_records(self) -> list[ScoredRecord]:
        return self.contributing_events.top_equity_records

    @property
    def top_crypto_records(self) -> list[ScoredRecord]:
        return self.contributing_events.top_crypto_records

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_utc": self.as_of_utc,
            "decision": self.decision.value,
            "scores": self.scores.to_dict(),
            "thresholds": dict(self.thresholds),
            "regimes": {
                "label": self.regime_label,
                "multiplier": float(self.regimes.get("multiplier", 1.0)),
            },
            "veto_flags": {
                "regime_vetoed": self.regime_vetoed,
            },
            "rationale": self.rationale,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MarketSelectorDecision:
        decision_raw = payload.get("decision", MarketRoute.RUN_NONE.value)
        raw_thresholds = payload.get("thresholds", {})
        return cls(
            as_of_utc=str(payload.get("as_of_utc", "")),
            decision=MarketRoute(decision_raw),
            scores=ScoreBreakdown.from_dict(payload.get("scores", {})),
            thresholds={
                "min_score_to_run": float(raw_thresholds.get("min_score_to_run", 0.35)),
                "both_threshold": float(raw_thresholds.get("both_threshold", 0.60)),
                "low_opportunity_floor": float(raw_thresholds.get("low_opportunity_floor", raw_thresholds.get("min_score_to_run", 0.35))),
                "high_opportunity_threshold": float(raw_thresholds.get("high_opportunity_threshold", raw_thresholds.get("both_threshold", 0.60))),
                "both_margin": float(raw_thresholds.get("both_margin", 0.10)),
                "minimum_score_gap": float(raw_thresholds.get("minimum_score_gap", 0.15)),
            },
            regimes={
                "label": str(payload.get("regimes", {}).get("label", "UNKNOWN")),
                "multiplier": float(payload.get("regimes", {}).get("multiplier", 1.0)),
            },
            veto_flags={
                "regime_vetoed": bool(
                    payload.get("veto_flags", {}).get("regime_vetoed", False)
                )
            },
            rationale=str(payload.get("rationale", "")),
            version=str(payload.get("version", "v1")),
        )


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

    def __init__(self, config: dict[str, Any] | MarketSelectorConfig | None = None) -> None:
        cfg_dict: dict[str, Any] = config if isinstance(config, dict) else {}
        self._config = (
            config if isinstance(config, MarketSelectorConfig) else MarketSelectorConfig.from_dict(cfg_dict or None)
        )
        self._min_score: float = self._config.min_score_to_run
        self._both_threshold: float = self._config.both_threshold
        self._half_life_hours: float = self._config.half_life_hours
        self._top_n: int = self._config.top_n
        self._low_opportunity_floor: float = float(
            cfg_dict.get("low_opportunity_floor", self._min_score)
        )
        self._high_opportunity_threshold: float = float(
            cfg_dict.get("high_opportunity_threshold", self._both_threshold)
        )
        self._both_margin: float = float(cfg_dict.get("both_margin", 0.10))
        self._minimum_score_gap: float = float(cfg_dict.get("minimum_score_gap", 0.15))

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
        """Legacy adapter for ``score_input``."""
        return self.score_input(
            MarketSelectorInput(records=records, regime_label=regime_label)
        )

    def score_input(self, selector_input: MarketSelectorInput) -> MarketSelectorDecision:
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
        scored_at = selector_input.as_of_utc or datetime.now(tz=timezone.utc).isoformat()
        records = selector_input.records
        regime_label = selector_input.regime_label

        # 1. Partition by asset class
        equity_records = [r for r in records if r.asset == "equity"]
        crypto_records = [r for r in records if r.asset == "crypto"]

        # 2. Score each asset class
        equity_weights, equity_score_raw, equity_unknown_events = self._score_records(
            equity_records,
            "equity",
        )
        crypto_weights, crypto_score_raw, crypto_unknown_events = self._score_records(
            crypto_records,
            "crypto",
        )

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
            unknown_event_fallback_count=equity_unknown_events + crypto_unknown_events,
        )

        return MarketSelectorDecision(
            as_of_utc=scored_at,
            decision=route,
            scores=ScoreBreakdown(
                equity=round(equity_score, 6),
                crypto=round(crypto_score, 6),
                equity_record_count=len(equity_records),
                crypto_record_count=len(crypto_records),
                record_count=len(records),
            ),
            thresholds={
                "min_score_to_run": self._min_score,
                "both_threshold": self._both_threshold,
                "low_opportunity_floor": self._low_opportunity_floor,
                "high_opportunity_threshold": self._high_opportunity_threshold,
                "both_margin": self._both_margin,
                "minimum_score_gap": self._minimum_score_gap,
            },
            regimes={"label": regime_label, "multiplier": multiplier},
            veto_flags={"regime_vetoed": regime_vetoed},
            rationale=rationale,
            contributing_events=ContributingEventSummary(
                top_equity_records=top_equity,
                top_crypto_records=top_crypto,
            ),
            version="v1",
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

    def _canonical_event_type(self, event_type: str | None) -> str:
        """Return canonical event type key for affinity lookup.

        Policy:
        - Normalize by stripping whitespace and uppercasing.
        - Resolve explicit aliases (e.g. ``REGULATION`` → ``REGULATORY``).
        - Preserve only canonical keys from ``EVENT_ASSET_AFFINITY``.
        - Unknown values map to ``UNKNOWN``.
        """
        normalized = (event_type or "").strip().upper()
        if not normalized:
            return "UNKNOWN"
        if normalized in EVENT_ASSET_AFFINITY:
            return normalized
        return _EVENT_TYPE_ALIASES.get(normalized, "UNKNOWN")

    def _affinity(self, event_type: str, asset_class: str) -> tuple[float, bool]:
        """Return ``(affinity, used_unknown_fallback)`` for *event_type*.

        Unknown-event policy is explicitly neutral 0.5/0.5 to avoid accidental
        directional bias when upstream event typing is missing or novel.
        """
        canonical = self._canonical_event_type(event_type)
        used_unknown_fallback = canonical == "UNKNOWN"
        entry = EVENT_ASSET_AFFINITY.get(canonical, _DEFAULT_AFFINITY)
        return entry.get(asset_class, 0.5), used_unknown_fallback

    def _score_records(
        self,
        records: list[NewsIntelligenceRecord],
        asset_class: str,
    ) -> tuple[list[float], float, int]:
        """Compute a weighted aggregate score for *records*.

        Returns
        -------
        tuple[list[float], float, int]
            ``(per_record_weights, aggregate_score, unknown_event_fallback_count)``

            *per_record_weights* has the same length as *records* and contains
            each record's unscaled contribution (used for top-N selection).

            *aggregate_score* is in ``[0.0, 1.0]``.

        Scoring formula (per record):
            ``w_i = recency(timestamp_i) * impact_score_i
                    * affinity(event_type_i, asset_class) * confidence_i``

            ``score = sum(w_i) / max(N, 1)``  clipped to ``[0.0, 1.0]``
        """
        if not records:
            return [], 0.0, 0

        weights: list[float] = []
        unknown_event_fallback_count = 0
        for rec in records:
            affinity, used_unknown_fallback = self._affinity(rec.event_type, asset_class)
            if used_unknown_fallback:
                unknown_event_fallback_count += 1
            w = (
                self._recency_weight(rec.timestamp)
                * rec.impact_score
                * affinity
                * rec.confidence
            )
            weights.append(w)

        score = sum(weights) / max(len(weights), 1)
        score = max(0.0, min(1.0, score))
        return weights, score, unknown_event_fallback_count

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
        unknown_event_fallback_count: int,
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
        if unknown_event_fallback_count > 0:
            parts.append(
                "unknown_event_fallback="
                f"{unknown_event_fallback_count} record(s) "
                "(neutral_affinity=0.50/0.50)"
            )
        return "; ".join(parts)
