"""Market event ontology for SpectraQuant-AI.

Implements Section 5 of the system specification: a minimal but complete
taxonomy of market-moving events.  Each event class defines:

* required slots – fields that **must** be populated for the event to be valid
* optional uncertainty fields – confidence, source reliability, etc.

Design goals
------------
* Implementation-neutral: pure Python dataclasses, no framework dependencies.
* Extensible: add sub-types by subclassing the appropriate base.
* Validation-ready: :meth:`BaseMarketEvent.validate` returns a list of
  missing required fields so callers can decide how to handle partial data.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any

__all__ = [
    "BaseMarketEvent",
    "EarningsEvent",
    "RegulatoryEvent",
    "MacroEvent",
    "CorporateActionEvent",
    "OperationsDisruptionEvent",
    "RiskEvent",
    "MAndAEvent",
    "EVENT_REGISTRY",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class BaseMarketEvent:
    """Common attributes shared by all market events.

    Attributes
    ----------
    event_id : str
        Unique identifier (caller-supplied or generated).
    event_type : str
        Discriminator string from the ontology (e.g. ``"earnings_beat"``).
    ticker : str
        Primary affected ticker.
    detected_at : datetime
        When the event was first detected (UTC).
    event_date : datetime | None
        When the event actually occurred or is scheduled (UTC); None = unknown.
    source : str
        News source / data vendor name.
    source_reliability : float
        Prior reliability score for the source, range [0, 1].
    confidence : float
        Model confidence that the event classification is correct, [0, 1].
    sentiment : str
        High-level sentiment label: ``"positive"``, ``"negative"``, ``"neutral"``.
    uncertainty_flags : list[str]
        Free-form list of reasons the event classification may be uncertain.
    metadata : dict
        Additional key-value pairs preserved for downstream use.
    """

    event_id: str
    event_type: str
    ticker: str
    detected_at: datetime

    event_date: datetime | None = None
    source: str = ""
    source_reliability: float = 1.0
    confidence: float = 1.0
    sentiment: str = "neutral"
    uncertainty_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Slots that sub-classes declare as REQUIRED (override in subclass)
    _required_slots: tuple[str, ...] = field(
        default_factory=tuple, init=False, repr=False, compare=False
    )

    def validate(self) -> list[str]:
        """Return a list of missing required slot names.

        An empty list means the event is fully populated.
        """
        missing: list[str] = []
        for slot in self._required_slots:
            val = getattr(self, slot, None)
            if val is None or val == "":
                missing.append(slot)
        return missing

    def is_valid(self) -> bool:
        """True iff all required slots are populated."""
        return len(self.validate()) == 0


# ---------------------------------------------------------------------------
# Earnings events
# ---------------------------------------------------------------------------

@dataclass
class EarningsEvent(BaseMarketEvent):
    """Quarterly / annual earnings announcement.

    Required slots
    --------------
    reported_eps : float
        Earnings per share reported.
    consensus_eps : float
        Analyst consensus EPS estimate before the announcement.
    revenue_reported : float
        Actual revenue (in millions, local currency).
    revenue_consensus : float
        Analyst consensus revenue estimate.

    Uncertainty fields
    ------------------
    guidance_change : str | None
        ``"raised"``, ``"lowered"``, ``"maintained"``, or None if not disclosed.
    beat_or_miss : str | None
        ``"beat"``, ``"miss"``, ``"in-line"`` derived from EPS comparison.
    """

    reported_eps: float | None = None
    consensus_eps: float | None = None
    revenue_reported: float | None = None
    revenue_consensus: float | None = None
    guidance_change: str | None = None
    beat_or_miss: str | None = None

    def __post_init__(self) -> None:
        self._required_slots = ("reported_eps", "consensus_eps")
        if (
            self.reported_eps is not None
            and self.consensus_eps is not None
            and self.beat_or_miss is None
        ):
            diff = self.reported_eps - self.consensus_eps
            if diff > 0:
                self.beat_or_miss = "beat"
            elif diff < 0:
                self.beat_or_miss = "miss"
            else:
                self.beat_or_miss = "in-line"


# ---------------------------------------------------------------------------
# Regulatory events
# ---------------------------------------------------------------------------

@dataclass
class RegulatoryEvent(BaseMarketEvent):
    """Regulatory action, fine, approval, or compliance notice.

    Required slots
    --------------
    regulator : str
        Name of the regulatory body (e.g. ``"SEBI"``, ``"FDA"``, ``"FCA"``).
    action_type : str
        One of: ``"fine"``, ``"approval"``, ``"rejection"``, ``"investigation"``,
        ``"consent_order"``, ``"ban"``, ``"warning"``.

    Uncertainty fields
    ------------------
    fine_amount : float | None
        Fine in millions (local currency); None if unknown.
    outcome_probability : float | None
        Estimated probability that a pending action is adverse, [0, 1].
    """

    regulator: str = ""
    action_type: str = ""
    fine_amount: float | None = None
    outcome_probability: float | None = None

    def __post_init__(self) -> None:
        self._required_slots = ("regulator", "action_type")


# ---------------------------------------------------------------------------
# Macro events
# ---------------------------------------------------------------------------

@dataclass
class MacroEvent(BaseMarketEvent):
    """Macroeconomic data release or central bank action.

    Required slots
    --------------
    indicator : str
        Economic indicator name (e.g. ``"CPI"``, ``"RBI_RATE"``, ``"GDP"``).
    actual_value : float
        Released value.

    Uncertainty fields
    ------------------
    consensus_value : float | None
        Bloomberg / Reuters consensus estimate.
    prior_value : float | None
        Previous period's value.
    surprise_direction : str | None
        ``"positive_surprise"``, ``"negative_surprise"``, ``"in-line"``.
    """

    indicator: str = ""
    actual_value: float | None = None
    consensus_value: float | None = None
    prior_value: float | None = None
    surprise_direction: str | None = None

    def __post_init__(self) -> None:
        self._required_slots = ("indicator", "actual_value")
        if (
            self.actual_value is not None
            and self.consensus_value is not None
            and self.surprise_direction is None
        ):
            diff = self.actual_value - self.consensus_value
            if abs(diff) < 1e-9:
                self.surprise_direction = "in-line"
            elif diff > 0:
                self.surprise_direction = "positive_surprise"
            else:
                self.surprise_direction = "negative_surprise"


# ---------------------------------------------------------------------------
# Corporate actions
# ---------------------------------------------------------------------------

@dataclass
class CorporateActionEvent(BaseMarketEvent):
    """Dividend, stock split, buyback, or rights issue.

    Required slots
    --------------
    action_subtype : str
        One of: ``"dividend"``, ``"split"``, ``"buyback"``, ``"rights_issue"``,
        ``"bonus_shares"``.
    ex_date : datetime | None
        Ex-action date (UTC); required for dividend / split.

    Uncertainty fields
    ------------------
    amount : float | None
        Monetary amount (dividend per share or buyback size in millions).
    split_ratio : float | None
        Split ratio, e.g. 2.0 for a 2-for-1 split.
    """

    action_subtype: str = ""
    ex_date: datetime | None = None
    amount: float | None = None
    split_ratio: float | None = None

    def __post_init__(self) -> None:
        self._required_slots = ("action_subtype",)


# ---------------------------------------------------------------------------
# Operations disruption
# ---------------------------------------------------------------------------

@dataclass
class OperationsDisruptionEvent(BaseMarketEvent):
    """Supply chain, plant, cyber, or weather disruption.

    Required slots
    --------------
    disruption_type : str
        One of: ``"supply_chain"``, ``"plant_shutdown"``, ``"cyber_attack"``,
        ``"natural_disaster"``, ``"labour_action"``.
    severity : str
        ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.

    Uncertainty fields
    ------------------
    estimated_revenue_impact_pct : float | None
        Estimated revenue impact as % of annual revenue.
    resolution_timeline_days : int | None
        Best-estimate days to resolution.
    """

    disruption_type: str = ""
    severity: str = ""
    estimated_revenue_impact_pct: float | None = None
    resolution_timeline_days: int | None = None

    def __post_init__(self) -> None:
        self._required_slots = ("disruption_type", "severity")


# ---------------------------------------------------------------------------
# Risk events
# ---------------------------------------------------------------------------

@dataclass
class RiskEvent(BaseMarketEvent):
    """Credit, geopolitical, fraud, legal, or liquidity risk materialisation.

    Required slots
    --------------
    risk_category : str
        One of: ``"credit_downgrade"``, ``"fraud"``, ``"legal"``,
        ``"geopolitical"``, ``"liquidity_crisis"``, ``"management_departure"``.
    impact_severity : str
        ``"low"``, ``"medium"``, ``"high"``.

    Uncertainty fields
    ------------------
    credit_rating_change : str | None
        Old and new rating string, e.g. ``"BBB+ → BBB"``.
    legal_exposure_millions : float | None
        Estimated financial exposure in millions (local currency).
    """

    risk_category: str = ""
    impact_severity: str = ""
    credit_rating_change: str | None = None
    legal_exposure_millions: float | None = None

    def __post_init__(self) -> None:
        self._required_slots = ("risk_category", "impact_severity")


# ---------------------------------------------------------------------------
# M&A events
# ---------------------------------------------------------------------------

@dataclass
class MAndAEvent(BaseMarketEvent):
    """Merger, acquisition, divestiture, or joint-venture announcement.

    Required slots
    --------------
    deal_type : str
        One of: ``"acquisition"``, ``"merger"``, ``"divestiture"``,
        ``"joint_venture"``, ``"stake_sale"``.
    target_ticker : str
        Ticker of the target company (may equal ``ticker`` for the acquirer).

    Uncertainty fields
    ------------------
    deal_value_millions : float | None
        Announced deal value in millions (local currency).
    premium_pct : float | None
        Acquisition premium over last closing price, e.g. 25.0 for 25 %.
    deal_stage : str | None
        ``"rumour"``, ``"announced"``, ``"regulatory_review"``,
        ``"completed"``, ``"terminated"``.
    completion_probability : float | None
        Estimated probability of deal completion, [0, 1].
    """

    deal_type: str = ""
    target_ticker: str = ""
    deal_value_millions: float | None = None
    premium_pct: float | None = None
    deal_stage: str | None = None
    completion_probability: float | None = None

    def __post_init__(self) -> None:
        self._required_slots = ("deal_type", "target_ticker")


# ---------------------------------------------------------------------------
# Registry mapping type strings to classes
# ---------------------------------------------------------------------------

EVENT_REGISTRY: dict[str, type[BaseMarketEvent]] = {
    "earnings": EarningsEvent,
    "regulatory": RegulatoryEvent,
    "macro": MacroEvent,
    "corporate_action": CorporateActionEvent,
    "operations_disruption": OperationsDisruptionEvent,
    "risk": RiskEvent,
    "m_and_a": MAndAEvent,
}


def create_event(event_type: str, **kwargs: Any) -> BaseMarketEvent:
    """Factory function: create a typed event from the registry.

    Parameters
    ----------
    event_type : str
        Key from :data:`EVENT_REGISTRY`.
    **kwargs
        Keyword arguments forwarded to the event dataclass.

    Raises
    ------
    ValueError
        If *event_type* is not in :data:`EVENT_REGISTRY`.
    """
    cls = EVENT_REGISTRY.get(event_type)
    if cls is None:
        raise ValueError(
            f"Unknown event type: {event_type!r}. "
            f"Available: {sorted(EVENT_REGISTRY)}"
        )
    return cls(event_type=event_type, **kwargs)
