"""Deterministic V3 news-first market selector.

This module scores provider-agnostic :class:`NewsIntelligenceRecord` inputs for
equities and crypto, then emits a typed routing decision that can later sit
above the V3 strategy runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from spectraquant_v3.core.enums import MarketRoute
from spectraquant_v3.core.news_schema import NewsIntelligenceRecord

_DEFAULT_REFERENCE_TIME = "1970-01-01T00:00:00+00:00"
_DEFAULT_TOP_CONTRIBUTORS = 5
_TOP_K_INTENSITY = 3
_UNKNOWN_EVENT_TYPE = "UNKNOWN"
_BREADTH_NORMALIZATION_FACTOR = 3.0
_TOP_EVENT_WEIGHT = 0.85
_SECOND_EVENT_WEIGHT = 0.10
_THIRD_EVENT_WEIGHT = 0.05
_INTENSITY_SCORE_WEIGHT = 0.90
_BREADTH_SCORE_WEIGHT = 0.10
_BREADTH_CUTOFF_FLOOR_MULTIPLIER = 0.50
_WEAK_EVENT_THRESHOLD_DIVISOR = 2.0
_MIN_BREADTH_CUTOFF = 0.05
_MAX_NOISE_PENALTY = 0.05
_RISK_OFF_MULTIPLIER = 0.75
_CROSS_ASSET_STRESS_MULTIPLIER = 0.90
_EVENT_DRIVEN_MULTIPLIER = 1.10

EVENT_ASSET_AFFINITY: dict[str, dict[str, float]] = {
    "EARNINGS": {"equity": 1.00, "crypto": 0.05},
    "GUIDANCE": {"equity": 0.95, "crypto": 0.05},
    "M_AND_A": {"equity": 0.90, "crypto": 0.10},
    "CORPORATE_ACTION": {"equity": 0.80, "crypto": 0.05},
    "DIVIDEND": {"equity": 0.85, "crypto": 0.00},
    "ANALYST": {"equity": 0.75, "crypto": 0.10},
    "MACRO": {"equity": 0.60, "crypto": 0.80},
    "REGULATORY": {"equity": 0.50, "crypto": 0.85},
    "PROTOCOL_UPGRADE": {"equity": 0.05, "crypto": 0.95},
    "LISTING": {"equity": 0.10, "crypto": 0.90},
    "EXCHANGE_HACK": {"equity": 0.00, "crypto": 1.00},
    "SECURITY_INCIDENT": {"equity": 0.30, "crypto": 0.75},
    "ONCHAIN": {"equity": 0.00, "crypto": 0.95},
    "OPERATIONS_DISRUPTION": {"equity": 0.70, "crypto": 0.20},
    "SECTOR_THEME": {"equity": 0.70, "crypto": 0.55},
    "SOCIAL_BUZZ": {"equity": 0.20, "crypto": 0.45},
    _UNKNOWN_EVENT_TYPE: {"equity": 0.50, "crypto": 0.50},
}

_EVENT_TYPE_ALIASES: dict[str, str] = {
    "M&A": "M_AND_A",
    "MERGERS_AND_ACQUISITIONS": "M_AND_A",
    "REGULATION": "REGULATORY",
}


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _serialize_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _normalize_iso_datetime_string(value: str | None, *, default: str | None = None) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed is not None:
        return _serialize_datetime(parsed)
    if default is not None:
        fallback = _parse_iso_datetime(default)
        if fallback is not None:
            return _serialize_datetime(fallback)
    raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}")


def canonicalize_event_type(event_type: str | None) -> str:
    normalized = (event_type or "").strip().upper()
    if not normalized:
        return _UNKNOWN_EVENT_TYPE
    if normalized in EVENT_ASSET_AFFINITY:
        return normalized
    return _EVENT_TYPE_ALIASES.get(normalized, _UNKNOWN_EVENT_TYPE)


def get_event_asset_affinity(event_type: str | None, asset_class: str) -> float:
    canonical = canonicalize_event_type(event_type)
    return float(EVENT_ASSET_AFFINITY.get(canonical, EVENT_ASSET_AFFINITY[_UNKNOWN_EVENT_TYPE]).get(asset_class, 0.5))


def _route_from_raw(value: str | MarketRoute) -> MarketRoute:
    if isinstance(value, MarketRoute):
        return value
    text = str(value).strip()
    if not text:
        return MarketRoute.RUN_NONE
    if text in MarketRoute.__members__:
        return MarketRoute[text]
    try:
        return MarketRoute(text.lower())
    except ValueError:
        return MarketRoute.RUN_NONE


@dataclass(frozen=True)
class MarketSelectorConfig:
    low_opportunity_floor: float = 0.15
    high_opportunity_threshold: float = 0.55
    both_margin: float = 0.10
    minimum_score_gap: float = 0.08
    recency_half_life_hours_equity: float = 18.0
    recency_half_life_hours_crypto: float = 8.0

    def __post_init__(self) -> None:
        if self.recency_half_life_hours_equity <= 0:
            raise ValueError("recency_half_life_hours_equity must be > 0")
        if self.recency_half_life_hours_crypto <= 0:
            raise ValueError("recency_half_life_hours_crypto must be > 0")
        if self.low_opportunity_floor < 0 or self.high_opportunity_threshold < 0:
            raise ValueError("opportunity thresholds must be >= 0")
        if self.both_margin < 0 or self.minimum_score_gap < 0:
            raise ValueError("decision margins must be >= 0")

    def to_dict(self) -> dict[str, float]:
        return {
            "low_opportunity_floor": _round_float(self.low_opportunity_floor),
            "high_opportunity_threshold": _round_float(self.high_opportunity_threshold),
            "both_margin": _round_float(self.both_margin),
            "minimum_score_gap": _round_float(self.minimum_score_gap),
            "recency_half_life_hours_equity": _round_float(self.recency_half_life_hours_equity),
            "recency_half_life_hours_crypto": _round_float(self.recency_half_life_hours_crypto),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> MarketSelectorConfig:
        data = payload or {}
        return cls(
            low_opportunity_floor=float(data.get("low_opportunity_floor", 0.15)),
            high_opportunity_threshold=float(data.get("high_opportunity_threshold", 0.55)),
            both_margin=float(data.get("both_margin", 0.10)),
            minimum_score_gap=float(data.get("minimum_score_gap", 0.08)),
            recency_half_life_hours_equity=float(data.get("recency_half_life_hours_equity", 18.0)),
            recency_half_life_hours_crypto=float(data.get("recency_half_life_hours_crypto", 8.0)),
        )


@dataclass(frozen=True)
class MarketRegimes:
    global_regime: str = "NORMAL"
    equity_regime: str = "NORMAL"
    crypto_regime: str = "NORMAL"

    def normalized(self) -> MarketRegimes:
        return MarketRegimes(
            global_regime=self.global_regime.strip().upper(),
            equity_regime=self.equity_regime.strip().upper(),
            crypto_regime=self.crypto_regime.strip().upper(),
        )

    def to_dict(self) -> dict[str, str]:
        normalized = self.normalized()
        return {
            "global": normalized.global_regime,
            "equity": normalized.equity_regime,
            "crypto": normalized.crypto_regime,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> MarketRegimes:
        data = payload or {}
        return cls(
            global_regime=str(data.get("global", "NORMAL")),
            equity_regime=str(data.get("equity", "NORMAL")),
            crypto_regime=str(data.get("crypto", "NORMAL")),
        )

    def for_asset(self, asset_class: str) -> str:
        normalized = self.normalized()
        if asset_class == "equity":
            return normalized.equity_regime
        return normalized.crypto_regime


@dataclass(frozen=True)
class MarketRiskFlags:
    panic_mode: bool = False
    high_cross_asset_stress: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "panic_mode": bool(self.panic_mode),
            "high_cross_asset_stress": bool(self.high_cross_asset_stress),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> MarketRiskFlags:
        data = payload or {}
        return cls(
            panic_mode=bool(data.get("panic_mode", False)),
            high_cross_asset_stress=bool(data.get("high_cross_asset_stress", False)),
        )


@dataclass(frozen=True)
class MarketSelectorInput:
    as_of_utc: str
    news_events: list[NewsIntelligenceRecord] = field(default_factory=list)
    regimes: MarketRegimes = field(default_factory=MarketRegimes)
    risk_flags: MarketRiskFlags = field(default_factory=MarketRiskFlags)
    config: MarketSelectorConfig | None = None

    @property
    def records(self) -> list[NewsIntelligenceRecord]:
        return self.news_events

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_utc": self.as_of_utc,
            "news_events": [event.to_dict() for event in self.news_events],
            "regimes": self.regimes.to_dict(),
            "risk_flags": self.risk_flags.to_dict(),
            "config": None if self.config is None else self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MarketSelectorInput:
        raw_events = payload.get("news_events", payload.get("records", []))
        events = [NewsIntelligenceRecord(**event_payload) for event_payload in raw_events]
        as_of_value = payload.get("as_of_utc")
        if as_of_value is None:
            as_of_value = _derive_reference_time(events)
        else:
            as_of_value = _normalize_iso_datetime_string(str(as_of_value), default=_DEFAULT_REFERENCE_TIME)
        return cls(
            as_of_utc=str(as_of_value),
            news_events=events,
            regimes=MarketRegimes.from_dict(payload.get("regimes")),
            risk_flags=MarketRiskFlags.from_dict(payload.get("risk_flags")),
            config=None if payload.get("config") is None else MarketSelectorConfig.from_dict(payload.get("config")),
        )


@dataclass(frozen=True)
class ScoreBreakdown:
    equity_opportunity_score: float
    crypto_opportunity_score: float

    @property
    def equity(self) -> float:
        return self.equity_opportunity_score

    @property
    def crypto(self) -> float:
        return self.crypto_opportunity_score

    def to_dict(self) -> dict[str, float]:
        return {
            "equity_opportunity_score": _round_float(self.equity_opportunity_score),
            "crypto_opportunity_score": _round_float(self.crypto_opportunity_score),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ScoreBreakdown:
        data = payload or {}
        return cls(
            equity_opportunity_score=float(data.get("equity_opportunity_score", data.get("equity", 0.0))),
            crypto_opportunity_score=float(data.get("crypto_opportunity_score", data.get("crypto", 0.0))),
        )


@dataclass(frozen=True)
class ContributingEventSummary:
    canonical_symbol: str
    asset: str
    event_type: str
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_symbol": self.canonical_symbol,
            "asset": self.asset,
            "event_type": self.event_type,
            "contribution": _round_float(self.contribution),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ContributingEventSummary:
        return cls(
            canonical_symbol=str(payload.get("canonical_symbol", "")),
            asset=str(payload.get("asset", "")),
            event_type=str(payload.get("event_type", _UNKNOWN_EVENT_TYPE)),
            contribution=float(payload.get("contribution", 0.0)),
        )


@dataclass(frozen=True)
class SelectorRationale:
    primary_reason: str
    secondary_reasons: list[str] = field(default_factory=list)
    top_contributing_events: list[ContributingEventSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_reason": self.primary_reason,
            "secondary_reasons": list(self.secondary_reasons),
            "top_contributing_events": [event.to_dict() for event in self.top_contributing_events],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SelectorRationale:
        data = payload or {}
        return cls(
            primary_reason=str(data.get("primary_reason", "")),
            secondary_reasons=[str(item) for item in data.get("secondary_reasons", [])],
            top_contributing_events=[
                ContributingEventSummary.from_dict(item)
                for item in data.get("top_contributing_events", [])
            ],
        )


@dataclass(frozen=True)
class VetoFlags:
    """Structured veto/risk-penalty flags for selector output.

    ``risk_penalty_applied`` is the broader V1 flag. The legacy
    ``risk_off_penalty_applied`` alias is kept in sync for compatibility.
    """

    panic_veto: bool = False
    risk_penalty_applied: bool = False
    risk_off_penalty_applied: bool = False

    def __post_init__(self) -> None:
        # The broad flag and legacy alias intentionally collapse to one
        # effective value so serialized payloads remain backwards compatible.
        effective_risk_penalty = bool(
            self.risk_penalty_applied or self.risk_off_penalty_applied
        )
        object.__setattr__(self, "risk_penalty_applied", effective_risk_penalty)
        object.__setattr__(self, "risk_off_penalty_applied", effective_risk_penalty)

    def to_dict(self) -> dict[str, bool]:
        return {
            "panic_veto": bool(self.panic_veto),
            "risk_penalty_applied": bool(self.risk_penalty_applied),
            "risk_off_penalty_applied": bool(self.risk_off_penalty_applied),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> VetoFlags:
        data = payload or {}
        effective_risk_penalty = bool(
            data.get(
                "risk_penalty_applied",
                data.get("risk_off_penalty_applied", False),
            )
        )
        return cls(
            panic_veto=bool(data.get("panic_veto", False)),
            risk_penalty_applied=effective_risk_penalty,
            risk_off_penalty_applied=effective_risk_penalty,
        )


@dataclass(frozen=True)
class MarketSelectorDecision:
    as_of_utc: str
    decision: MarketRoute
    scores: ScoreBreakdown
    thresholds: MarketSelectorConfig
    regimes: MarketRegimes
    veto_flags: VetoFlags
    rationale: SelectorRationale
    version: str = "v1"

    @property
    def route(self) -> MarketRoute:
        return self.decision

    @property
    def equity_score(self) -> float:
        return self.scores.equity_opportunity_score

    @property
    def crypto_score(self) -> float:
        return self.scores.crypto_opportunity_score

    @property
    def scored_at(self) -> str:
        return self.as_of_utc

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_utc": self.as_of_utc,
            "decision": self.decision.name,
            "scores": self.scores.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "regimes": self.regimes.to_dict(),
            "veto_flags": self.veto_flags.to_dict(),
            "rationale": self.rationale.to_dict(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MarketSelectorDecision:
        return cls(
            as_of_utc=str(payload.get("as_of_utc", _DEFAULT_REFERENCE_TIME)),
            decision=_route_from_raw(payload.get("decision", MarketRoute.RUN_NONE.name)),
            scores=ScoreBreakdown.from_dict(payload.get("scores")),
            thresholds=MarketSelectorConfig.from_dict(payload.get("thresholds")),
            regimes=MarketRegimes.from_dict(payload.get("regimes")),
            veto_flags=VetoFlags.from_dict(payload.get("veto_flags")),
            rationale=SelectorRationale.from_dict(payload.get("rationale")),
            version=str(payload.get("version", "v1")),
        )


def _derive_reference_time(news_events: list[NewsIntelligenceRecord]) -> str:
    timestamps = [
        parsed
        for parsed in (_parse_iso_datetime(event.timestamp) for event in news_events)
        if parsed is not None
    ]
    if not timestamps:
        return _DEFAULT_REFERENCE_TIME
    return _serialize_datetime(max(timestamps))


class MarketSelector:
    """Deterministic selector that routes between equities, crypto, both, or none."""

    def __init__(self, config: dict[str, Any] | MarketSelectorConfig | None = None) -> None:
        self._config = config if isinstance(config, MarketSelectorConfig) else MarketSelectorConfig.from_dict(config)

    def score(
        self,
        records: list[NewsIntelligenceRecord],
        *,
        regime_label: str = "NORMAL",
        as_of_utc: str | None = None,
    ) -> MarketSelectorDecision:
        return self.score_input(
            MarketSelectorInput(
                as_of_utc=as_of_utc or _derive_reference_time(records),
                news_events=list(records),
                regimes=MarketRegimes(global_regime=regime_label),
                risk_flags=MarketRiskFlags(),
            )
        )

    def score_input(self, selector_input: MarketSelectorInput) -> MarketSelectorDecision:
        config = selector_input.config or self._config
        as_of_dt = _parse_iso_datetime(selector_input.as_of_utc)
        if as_of_dt is None:
            as_of_dt = _parse_iso_datetime(_derive_reference_time(selector_input.news_events))
        if as_of_dt is None:
            raise ValueError(f"Could not determine selector reference time from {selector_input.as_of_utc!r}")
        as_of_utc = _serialize_datetime(as_of_dt)

        regimes = selector_input.regimes.normalized()
        panic_veto = selector_input.risk_flags.panic_mode or self._is_panic_regime(regimes)

        equity_result, equity_penalty = self._score_asset_class(
            asset_class="equity",
            news_events=selector_input.news_events,
            as_of_utc=as_of_dt,
            config=config,
            regimes=regimes,
            risk_flags=selector_input.risk_flags,
        )
        crypto_result, crypto_penalty = self._score_asset_class(
            asset_class="crypto",
            news_events=selector_input.news_events,
            as_of_utc=as_of_dt,
            config=config,
            regimes=regimes,
            risk_flags=selector_input.risk_flags,
        )

        decision = self._decide_route(
            equity_score=equity_result["score"],
            crypto_score=crypto_result["score"],
            panic_veto=panic_veto,
            config=config,
        )

        top_events = self._select_top_contributing_events(
            decision=decision,
            equity_contributions=equity_result["top_events"],
            crypto_contributions=crypto_result["top_events"],
        )
        rationale = self._build_rationale(
            decision=decision,
            equity_score=equity_result["score"],
            crypto_score=crypto_result["score"],
            config=config,
            regimes=regimes,
            panic_veto=panic_veto,
            risk_off_penalty_applied=equity_penalty or crypto_penalty,
            top_contributing_events=top_events,
        )

        return MarketSelectorDecision(
            as_of_utc=as_of_utc,
            decision=decision,
            scores=ScoreBreakdown(
                equity_opportunity_score=equity_result["score"],
                crypto_opportunity_score=crypto_result["score"],
            ),
            thresholds=config,
            regimes=regimes,
            veto_flags=VetoFlags(
                panic_veto=panic_veto,
                risk_penalty_applied=equity_penalty or crypto_penalty,
                risk_off_penalty_applied=equity_penalty or crypto_penalty,
            ),
            rationale=rationale,
            version="v1",
        )

    def _score_asset_class(
        self,
        *,
        asset_class: str,
        news_events: list[NewsIntelligenceRecord],
        as_of_utc: datetime,
        config: MarketSelectorConfig,
        regimes: MarketRegimes,
        risk_flags: MarketRiskFlags,
    ) -> tuple[dict[str, Any], bool]:
        scored_events: list[ContributingEventSummary] = []
        contributions: list[float] = []
        for event in news_events:
            contribution = self._event_contribution(
                record=event,
                asset_class=asset_class,
                as_of_utc=as_of_utc,
                config=config,
            )
            contributions.append(contribution)
            scored_events.append(
                ContributingEventSummary(
                    canonical_symbol=event.canonical_symbol,
                    asset=event.asset,
                    event_type=canonicalize_event_type(event.event_type),
                    contribution=contribution,
                )
            )

        scored_events.sort(
            key=lambda item: (item.contribution, item.canonical_symbol, item.asset, item.event_type),
            reverse=True,
        )
        base_score = self._asset_opportunity_score(contributions, config)
        adjusted_score, risk_off_penalty_applied = self._apply_regime_modifiers(
            base_score=base_score,
            asset_class=asset_class,
            regimes=regimes,
            risk_flags=risk_flags,
        )
        return {
            "score": _round_float(adjusted_score),
            "top_events": scored_events[:_DEFAULT_TOP_CONTRIBUTORS],
        }, risk_off_penalty_applied

    def _event_contribution(
        self,
        *,
        record: NewsIntelligenceRecord,
        asset_class: str,
        as_of_utc: datetime,
        config: MarketSelectorConfig,
    ) -> float:
        event_time = _parse_iso_datetime(record.timestamp)
        if event_time is None:
            event_time = as_of_utc
        age_hours = max((as_of_utc - event_time).total_seconds() / 3600.0, 0.0)
        half_life = (
            config.recency_half_life_hours_equity
            if asset_class == "equity"
            else config.recency_half_life_hours_crypto
        )
        recency_weight = 0.5 ** (age_hours / half_life)
        # NewsIntelligenceRecord clamps sentiment_score into [-1.0, 1.0].
        # The selector is opportunity-oriented rather than direction-oriented:
        # strongly negative catalyst tone can still imply a strong short or
        # risk-management opportunity, so V1 uses absolute sentiment magnitude.
        # Formula: sentiment_factor = 0.5 + 0.5 * |sentiment_score|.
        # This keeps the factor bounded in [0.5, 1.0], where neutral sentiment
        # still preserves half of the event's base contribution.
        sentiment_factor = 0.5 + 0.5 * abs(float(record.sentiment_score))
        affinity = get_event_asset_affinity(record.event_type, asset_class)
        contribution = (
            recency_weight
            * float(record.impact_score)
            * float(record.confidence)
            * affinity
            * sentiment_factor
        )
        return max(0.0, min(1.0, contribution))

    def _asset_opportunity_score(
        self,
        contributions: list[float],
        config: MarketSelectorConfig,
    ) -> float:
        if not contributions:
            return 0.0
        ranked = sorted(contributions, reverse=True)
        top_k = ranked[:_TOP_K_INTENSITY]
        while len(top_k) < _TOP_K_INTENSITY:
            top_k.append(0.0)
        intensity = (
            (_TOP_EVENT_WEIGHT * top_k[0])
            + (_SECOND_EVENT_WEIGHT * top_k[1])
            + (_THIRD_EVENT_WEIGHT * top_k[2])
        )
        breadth_cutoff = max(
            config.low_opportunity_floor * _BREADTH_CUTOFF_FLOOR_MULTIPLIER,
            _MIN_BREADTH_CUTOFF,
        )
        breadth = min(
            sum(1 for item in contributions if item >= breadth_cutoff)
            / _BREADTH_NORMALIZATION_FACTOR,
            1.0,
        )
        weak_ratio = sum(
            1 for item in contributions if item < breadth_cutoff / _WEAK_EVENT_THRESHOLD_DIVISOR
        ) / max(len(contributions), 1)
        noise_penalty = min(weak_ratio * _MAX_NOISE_PENALTY, _MAX_NOISE_PENALTY)
        score = (_INTENSITY_SCORE_WEIGHT * intensity) + (_BREADTH_SCORE_WEIGHT * breadth) - noise_penalty
        return max(0.0, min(1.0, score))

    def _apply_regime_modifiers(
        self,
        *,
        base_score: float,
        asset_class: str,
        regimes: MarketRegimes,
        risk_flags: MarketRiskFlags,
    ) -> tuple[float, bool]:
        multiplier = 1.0
        risk_off_penalty_applied = False
        applicable_regimes = {regimes.global_regime, regimes.for_asset(asset_class)}

        if self._has_regime(applicable_regimes, "RISK_OFF"):
            # V1 keeps the penalty meaningful but not absolute so a very strong
            # catalyst cluster can still surface as actionable.
            multiplier *= _RISK_OFF_MULTIPLIER
            risk_off_penalty_applied = True
        if risk_flags.high_cross_asset_stress:
            multiplier *= _CROSS_ASSET_STRESS_MULTIPLIER
            risk_off_penalty_applied = True
        if self._has_regime(applicable_regimes, "EVENT_DRIVEN"):
            # Use only a modest boost so regime context helps ordering without
            # overwhelming the underlying event-intensity score.
            multiplier *= _EVENT_DRIVEN_MULTIPLIER

        return max(0.0, min(1.0, base_score * multiplier)), risk_off_penalty_applied

    @staticmethod
    def _has_regime(regimes: set[str], regime_name: str) -> bool:
        return regime_name in regimes

    def _is_panic_regime(self, regimes: MarketRegimes) -> bool:
        return self._has_regime(
            {
                regimes.global_regime,
                regimes.equity_regime,
                regimes.crypto_regime,
            },
            "PANIC",
        )

    def _decide_route(
        self,
        *,
        equity_score: float,
        crypto_score: float,
        panic_veto: bool,
        config: MarketSelectorConfig,
    ) -> MarketRoute:
        if panic_veto:
            return MarketRoute.RUN_NONE
        if (
            equity_score < config.low_opportunity_floor
            and crypto_score < config.low_opportunity_floor
        ):
            return MarketRoute.RUN_NONE

        score_gap = abs(equity_score - crypto_score)
        if (
            equity_score >= config.high_opportunity_threshold
            and crypto_score >= config.high_opportunity_threshold
            and score_gap <= config.both_margin
        ):
            return MarketRoute.RUN_BOTH
        if (
            equity_score >= config.high_opportunity_threshold
            and (equity_score - crypto_score) >= config.minimum_score_gap
        ):
            return MarketRoute.RUN_EQUITIES
        if (
            crypto_score >= config.high_opportunity_threshold
            and (crypto_score - equity_score) >= config.minimum_score_gap
        ):
            return MarketRoute.RUN_CRYPTO
        if (
            equity_score >= config.low_opportunity_floor
            and crypto_score >= config.low_opportunity_floor
            and score_gap <= config.minimum_score_gap
        ):
            return MarketRoute.RUN_BOTH
        if (
            equity_score >= config.low_opportunity_floor
            and (equity_score - crypto_score) >= config.minimum_score_gap
        ):
            return MarketRoute.RUN_EQUITIES
        if (
            crypto_score >= config.low_opportunity_floor
            and (crypto_score - equity_score) >= config.minimum_score_gap
        ):
            return MarketRoute.RUN_CRYPTO
        return MarketRoute.RUN_NONE

    def _select_top_contributing_events(
        self,
        *,
        decision: MarketRoute,
        equity_contributions: list[ContributingEventSummary],
        crypto_contributions: list[ContributingEventSummary],
    ) -> list[ContributingEventSummary]:
        if decision == MarketRoute.RUN_EQUITIES:
            selected = equity_contributions
        elif decision == MarketRoute.RUN_CRYPTO:
            selected = crypto_contributions
        elif decision == MarketRoute.RUN_BOTH:
            selected = equity_contributions + crypto_contributions
        else:
            selected = equity_contributions + crypto_contributions

        ranked = sorted(
            selected,
            key=lambda item: (item.contribution, item.canonical_symbol, item.asset, item.event_type),
            reverse=True,
        )
        # Secondary keys (canonical_symbol, asset, event_type) keep identical
        # contributions deterministic across Python versions and runs without
        # needing any provider- or insertion-order dependence.
        deduped: list[ContributingEventSummary] = []
        seen: set[tuple[str, str, str]] = set()
        for event in ranked:
            key = (event.canonical_symbol, event.asset, event.event_type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(event)
            if len(deduped) >= _DEFAULT_TOP_CONTRIBUTORS:
                break
        return deduped

    def _build_rationale(
        self,
        *,
        decision: MarketRoute,
        equity_score: float,
        crypto_score: float,
        config: MarketSelectorConfig,
        regimes: MarketRegimes,
        panic_veto: bool,
        risk_off_penalty_applied: bool,
        top_contributing_events: list[ContributingEventSummary],
    ) -> SelectorRationale:
        if panic_veto:
            primary_reason = "Panic conditions vetoed all market routing decisions."
        elif decision == MarketRoute.RUN_NONE and not top_contributing_events:
            primary_reason = "No actionable normalized news events were available for either market."
        elif decision == MarketRoute.RUN_NONE:
            primary_reason = "Neither market cleared the configured opportunity floor with a decisive score gap."
        elif decision == MarketRoute.RUN_BOTH:
            primary_reason = (
                "Equity and crypto opportunity scores are both actionable and remain within the configured both-market margin."
            )
        elif decision == MarketRoute.RUN_EQUITIES:
            primary_reason = "Equity catalysts are stronger and clear the routing thresholds by a decisive margin."
        else:
            primary_reason = "Crypto catalysts are stronger and clear the routing thresholds by a decisive margin."

        secondary_reasons = [
            f"Equity opportunity score={equity_score:.4f}; crypto opportunity score={crypto_score:.4f}.",
            (
                "Thresholds applied: "
                f"low_floor={config.low_opportunity_floor:.2f}, "
                f"high_threshold={config.high_opportunity_threshold:.2f}, "
                f"both_margin={config.both_margin:.2f}, "
                f"minimum_gap={config.minimum_score_gap:.2f}."
            ),
            (
                "Regimes evaluated: "
                f"global={regimes.global_regime}, equity={regimes.equity_regime}, crypto={regimes.crypto_regime}."
            ),
        ]
        if risk_off_penalty_applied:
            secondary_reasons.append("Risk-off or cross-asset stress penalties reduced at least one opportunity score.")
        if top_contributing_events:
            top_event = top_contributing_events[0]
            secondary_reasons.append(
                "Top contributing event: "
                f"{top_event.canonical_symbol} {top_event.event_type} ({top_event.asset}) "
                f"contribution={top_event.contribution:.4f}."
            )

        return SelectorRationale(
            primary_reason=primary_reason,
            secondary_reasons=secondary_reasons,
            top_contributing_events=top_contributing_events,
        )
