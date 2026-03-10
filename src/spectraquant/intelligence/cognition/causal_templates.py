"""Causal templates: event type → expected market mechanism mapping.

Each :class:`CausalTemplate` describes the market mechanisms expected to
follow a given event type, the typical price-reaction pattern, and a
base strength score used as a prior in the belief engine.

Design philosophy
-----------------
Markets react to *events*, not sentiment.  For every classified event we need
to know **how** the market is likely to respond structurally — not just whether
text sounds positive or negative.

Event → Mechanism examples
---------------------------
earnings_surprise  → drift + gap continuation
rate_hike          → volatility spike + sector rotation
government_contract→ delayed drift
fraud_allegation   → gap down + high uncertainty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Sequence

__all__ = [
    "MechanismTag",
    "CausalTemplate",
    "CAUSAL_TEMPLATE_REGISTRY",
    "get_causal_template",
]


# ---------------------------------------------------------------------------
# Mechanism tags
# ---------------------------------------------------------------------------

class MechanismTag(str, Enum):
    """Tags describing expected market mechanism after an event."""

    DRIFT = "drift"
    GAP = "gap"
    REVERSAL_RISK = "reversal_risk"
    VOLATILITY_EXPANSION = "volatility_expansion"
    LIQUIDITY_SHOCK = "liquidity_shock"
    SECTOR_ROTATION = "sector_rotation"
    DELAYED_DRIFT = "delayed_drift"
    UNCERTAINTY = "uncertainty"
    MOMENTUM_CONTINUATION = "momentum_continuation"
    MEAN_REVERSION = "mean_reversion"


# ---------------------------------------------------------------------------
# Template dataclass
# ---------------------------------------------------------------------------

@dataclass
class CausalTemplate:
    """Expected market-mechanism template for a given event type.

    Attributes
    ----------
    event_type : str
        The ontology event type key (e.g. ``"earnings_beat"``).
    mechanism_tags : list[MechanismTag]
        Expected structural mechanisms triggered by this event.
    base_event_strength : float
        Prior strength of this event type's price impact, in [0, 1].
        Higher values mean the event historically drives larger moves.
    direction_bias : float
        Expected directional bias: +1 = bullish, -1 = bearish, 0 = neutral.
    typical_horizon_days : int
        Typical number of trading days until the alpha decays.
    uncertainty_multiplier : float
        Factor applied to uncertainty estimate for this event type.
        Values > 1 indicate the event class is historically noisy.
    description : str
        Human-readable description of the causal chain.
    """

    event_type: str
    mechanism_tags: list[MechanismTag] = field(default_factory=list)
    base_event_strength: float = 0.5
    direction_bias: float = 0.0
    typical_horizon_days: int = 5
    uncertainty_multiplier: float = 1.0
    description: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CAUSAL_TEMPLATE_REGISTRY: Dict[str, CausalTemplate] = {
    # -----------------------------------------------------------------
    # Earnings events
    # -----------------------------------------------------------------
    "earnings_beat": CausalTemplate(
        event_type="earnings_beat",
        mechanism_tags=[MechanismTag.DRIFT, MechanismTag.GAP, MechanismTag.MOMENTUM_CONTINUATION],
        base_event_strength=0.75,
        direction_bias=1.0,
        typical_horizon_days=5,
        uncertainty_multiplier=0.8,
        description="Earnings beat triggers gap-up + post-earnings drift; momentum tends to continue.",
    ),
    "earnings_miss": CausalTemplate(
        event_type="earnings_miss",
        mechanism_tags=[MechanismTag.GAP, MechanismTag.REVERSAL_RISK, MechanismTag.VOLATILITY_EXPANSION],
        base_event_strength=0.70,
        direction_bias=-1.0,
        typical_horizon_days=3,
        uncertainty_multiplier=1.2,
        description="Earnings miss triggers gap-down; reversal risk elevated if miss is minor.",
    ),
    "earnings_in_line": CausalTemplate(
        event_type="earnings_in_line",
        mechanism_tags=[MechanismTag.MEAN_REVERSION, MechanismTag.VOLATILITY_EXPANSION],
        base_event_strength=0.30,
        direction_bias=0.0,
        typical_horizon_days=2,
        uncertainty_multiplier=1.0,
        description="In-line earnings often produce muted reaction; IV crush and mean reversion likely.",
    ),
    # -----------------------------------------------------------------
    # Regulatory events
    # -----------------------------------------------------------------
    "regulatory_approval": CausalTemplate(
        event_type="regulatory_approval",
        mechanism_tags=[MechanismTag.GAP, MechanismTag.DRIFT],
        base_event_strength=0.80,
        direction_bias=1.0,
        typical_horizon_days=10,
        uncertainty_multiplier=0.7,
        description="Regulatory approval removes major risk; sustained drift expected.",
    ),
    "regulatory_penalty": CausalTemplate(
        event_type="regulatory_penalty",
        mechanism_tags=[MechanismTag.GAP, MechanismTag.UNCERTAINTY, MechanismTag.REVERSAL_RISK],
        base_event_strength=0.65,
        direction_bias=-1.0,
        typical_horizon_days=7,
        uncertainty_multiplier=1.5,
        description="Penalty creates uncertainty overhang; potential reversal if penalty smaller than feared.",
    ),
    # -----------------------------------------------------------------
    # Macro events
    # -----------------------------------------------------------------
    "rate_hike": CausalTemplate(
        event_type="rate_hike",
        mechanism_tags=[MechanismTag.VOLATILITY_EXPANSION, MechanismTag.SECTOR_ROTATION, MechanismTag.LIQUIDITY_SHOCK],
        base_event_strength=0.60,
        direction_bias=-0.5,
        typical_horizon_days=15,
        uncertainty_multiplier=1.3,
        description="Rate hike compresses multiples; sector rotation out of growth into value.",
    ),
    "rate_cut": CausalTemplate(
        event_type="rate_cut",
        mechanism_tags=[MechanismTag.DRIFT, MechanismTag.SECTOR_ROTATION, MechanismTag.MOMENTUM_CONTINUATION],
        base_event_strength=0.65,
        direction_bias=0.8,
        typical_horizon_days=20,
        uncertainty_multiplier=0.9,
        description="Rate cut expands multiples; risk-on drift across growth sectors.",
    ),
    "macro_surprise_positive": CausalTemplate(
        event_type="macro_surprise_positive",
        mechanism_tags=[MechanismTag.DRIFT, MechanismTag.MOMENTUM_CONTINUATION],
        base_event_strength=0.50,
        direction_bias=0.6,
        typical_horizon_days=10,
        uncertainty_multiplier=1.0,
        description="Positive macro surprise supports risk-on positioning.",
    ),
    "macro_surprise_negative": CausalTemplate(
        event_type="macro_surprise_negative",
        mechanism_tags=[MechanismTag.VOLATILITY_EXPANSION, MechanismTag.REVERSAL_RISK],
        base_event_strength=0.55,
        direction_bias=-0.6,
        typical_horizon_days=10,
        uncertainty_multiplier=1.2,
        description="Negative macro surprise increases volatility; risk-off positioning.",
    ),
    # -----------------------------------------------------------------
    # Corporate actions
    # -----------------------------------------------------------------
    "dividend_announcement": CausalTemplate(
        event_type="dividend_announcement",
        mechanism_tags=[MechanismTag.DRIFT, MechanismTag.MOMENTUM_CONTINUATION],
        base_event_strength=0.40,
        direction_bias=0.5,
        typical_horizon_days=5,
        uncertainty_multiplier=0.8,
        description="Dividend signals confidence; mild positive drift.",
    ),
    "buyback_announcement": CausalTemplate(
        event_type="buyback_announcement",
        mechanism_tags=[MechanismTag.DRIFT, MechanismTag.GAP],
        base_event_strength=0.55,
        direction_bias=0.7,
        typical_horizon_days=10,
        uncertainty_multiplier=0.9,
        description="Buyback signals undervaluation; gap-up + sustained drift expected.",
    ),
    "rights_issue": CausalTemplate(
        event_type="rights_issue",
        mechanism_tags=[MechanismTag.GAP, MechanismTag.REVERSAL_RISK],
        base_event_strength=0.50,
        direction_bias=-0.5,
        typical_horizon_days=5,
        uncertainty_multiplier=1.2,
        description="Rights issue dilutes existing shareholders; initial gap-down likely.",
    ),
    # -----------------------------------------------------------------
    # M&A events
    # -----------------------------------------------------------------
    "acquisition_announced": CausalTemplate(
        event_type="acquisition_announced",
        mechanism_tags=[MechanismTag.GAP, MechanismTag.DRIFT],
        base_event_strength=0.80,
        direction_bias=1.0,
        typical_horizon_days=30,
        uncertainty_multiplier=1.1,
        description="Acquisition target typically gaps to deal premium; acquirer uncertainty.",
    ),
    "merger_failed": CausalTemplate(
        event_type="merger_failed",
        mechanism_tags=[MechanismTag.GAP, MechanismTag.REVERSAL_RISK, MechanismTag.VOLATILITY_EXPANSION],
        base_event_strength=0.70,
        direction_bias=-0.8,
        typical_horizon_days=5,
        uncertainty_multiplier=1.4,
        description="Failed merger causes target to retrace to pre-deal price.",
    ),
    # -----------------------------------------------------------------
    # Risk / fraud events
    # -----------------------------------------------------------------
    "fraud_allegation": CausalTemplate(
        event_type="fraud_allegation",
        mechanism_tags=[MechanismTag.GAP, MechanismTag.UNCERTAINTY, MechanismTag.LIQUIDITY_SHOCK],
        base_event_strength=0.85,
        direction_bias=-1.0,
        typical_horizon_days=20,
        uncertainty_multiplier=2.0,
        description="Fraud allegation triggers gap-down with high uncertainty; liquidity dries up.",
    ),
    "insider_buying": CausalTemplate(
        event_type="insider_buying",
        mechanism_tags=[MechanismTag.DELAYED_DRIFT, MechanismTag.MOMENTUM_CONTINUATION],
        base_event_strength=0.55,
        direction_bias=0.7,
        typical_horizon_days=15,
        uncertainty_multiplier=0.9,
        description="Insider buying signals confidence; delayed drift as market reprices.",
    ),
    "insider_selling": CausalTemplate(
        event_type="insider_selling",
        mechanism_tags=[MechanismTag.REVERSAL_RISK, MechanismTag.UNCERTAINTY],
        base_event_strength=0.45,
        direction_bias=-0.4,
        typical_horizon_days=10,
        uncertainty_multiplier=1.3,
        description="Insider selling raises concern but is often routine; mild negative bias.",
    ),
    # -----------------------------------------------------------------
    # Operations events
    # -----------------------------------------------------------------
    "government_contract": CausalTemplate(
        event_type="government_contract",
        mechanism_tags=[MechanismTag.DELAYED_DRIFT, MechanismTag.MOMENTUM_CONTINUATION],
        base_event_strength=0.60,
        direction_bias=0.8,
        typical_horizon_days=15,
        uncertainty_multiplier=0.9,
        description="Government contract win provides revenue certainty; delayed drift.",
    ),
    "supply_shock": CausalTemplate(
        event_type="supply_shock",
        mechanism_tags=[MechanismTag.VOLATILITY_EXPANSION, MechanismTag.UNCERTAINTY, MechanismTag.SECTOR_ROTATION],
        base_event_strength=0.65,
        direction_bias=-0.5,
        typical_horizon_days=10,
        uncertainty_multiplier=1.5,
        description="Supply disruption raises costs/uncertainty; volatility expansion expected.",
    ),
    "plant_shutdown": CausalTemplate(
        event_type="plant_shutdown",
        mechanism_tags=[MechanismTag.GAP, MechanismTag.UNCERTAINTY],
        base_event_strength=0.60,
        direction_bias=-0.7,
        typical_horizon_days=7,
        uncertainty_multiplier=1.4,
        description="Plant shutdown impairs capacity; gap-down + uncertainty overhang.",
    ),
    # -----------------------------------------------------------------
    # Geopolitical / macro
    # -----------------------------------------------------------------
    "geopolitical_escalation": CausalTemplate(
        event_type="geopolitical_escalation",
        mechanism_tags=[MechanismTag.VOLATILITY_EXPANSION, MechanismTag.LIQUIDITY_SHOCK, MechanismTag.SECTOR_ROTATION],
        base_event_strength=0.70,
        direction_bias=-0.8,
        typical_horizon_days=5,
        uncertainty_multiplier=1.8,
        description="Geopolitical escalation triggers risk-off; broad volatility spike.",
    ),
    "sector_rerating": CausalTemplate(
        event_type="sector_rerating",
        mechanism_tags=[MechanismTag.DRIFT, MechanismTag.SECTOR_ROTATION, MechanismTag.MOMENTUM_CONTINUATION],
        base_event_strength=0.55,
        direction_bias=0.5,
        typical_horizon_days=20,
        uncertainty_multiplier=1.0,
        description="Sector rerating drives multi-week drift in affected names.",
    ),
    # -----------------------------------------------------------------
    # Fallback / unknown
    # -----------------------------------------------------------------
    "unknown": CausalTemplate(
        event_type="unknown",
        mechanism_tags=[MechanismTag.UNCERTAINTY],
        base_event_strength=0.20,
        direction_bias=0.0,
        typical_horizon_days=5,
        uncertainty_multiplier=1.5,
        description="Unknown event type; low confidence prior.",
    ),
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_causal_template(event_type: str) -> CausalTemplate:
    """Return the :class:`CausalTemplate` for *event_type*.

    Falls back to the ``"unknown"`` template if the type is not registered.

    Parameters
    ----------
    event_type : str
        Ontology event type string.

    Returns
    -------
    CausalTemplate
        The matching template, or the ``"unknown"`` fallback.
    """
    return CAUSAL_TEMPLATE_REGISTRY.get(event_type, CAUSAL_TEMPLATE_REGISTRY["unknown"])


def get_mechanism_tags(event_type: str) -> Sequence[MechanismTag]:
    """Convenience: return just the mechanism tags for *event_type*."""
    return get_causal_template(event_type).mechanism_tags
