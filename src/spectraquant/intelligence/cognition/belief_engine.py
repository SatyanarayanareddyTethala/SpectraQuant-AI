"""Belief Engine — multi-tower prediction model for SpectraQuant-AI-V3.

Combines four signal sources into a single *belief_score* per candidate:

    belief_score =
        w_price  * price_signal   +
        w_event  * event_strength +
        w_analog * analog_prior   +
        w_regime * regime_align

Weights are regime-adaptive: RISK_ON regimes raise w_event and w_analog;
high-volatility regimes raise w_price (momentum is less reliable when
markets are choppy).

Ranking-based execution
-----------------------
Instead of fixed BUY thresholds the engine ranks all candidates by
``belief_score`` and selects the top-K where K is adaptive:

    K = min(max_positions, ceil(adaptive_fraction * N))

This ensures small universes still produce trades.

Uncertainty
-----------
Each belief score carries an uncertainty estimate derived from:
- news_ambiguity     (confidence of event classification)
- analog_variance    (standard deviation of analog outcomes)
- volatility_spike   (current vol vs historical vol ratio)

Conviction = belief_score / (1 + uncertainty)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from spectraquant.intelligence.cognition.causal_templates import (
    MechanismTag,
    get_causal_template,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default regime-adaptive weights
# ---------------------------------------------------------------------------

# weights[regime] = {source: weight}  (must sum to 1.0)
_DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "RISK_ON": {
        "price": 0.25,
        "event": 0.35,
        "analog": 0.25,
        "regime": 0.15,
    },
    "RISK_OFF": {
        "price": 0.35,
        "event": 0.20,
        "analog": 0.20,
        "regime": 0.25,
    },
    "TRENDING": {
        "price": 0.35,
        "event": 0.30,
        "analog": 0.20,
        "regime": 0.15,
    },
    "CHOPPY": {
        "price": 0.20,
        "event": 0.30,
        "analog": 0.35,
        "regime": 0.15,
    },
    "EVENT_DRIVEN": {
        "price": 0.15,
        "event": 0.45,
        "analog": 0.30,
        "regime": 0.10,
    },
    "PANIC": {
        "price": 0.40,
        "event": 0.15,
        "analog": 0.15,
        "regime": 0.30,
    },
    # Fallback used when regime is unknown
    "DEFAULT": {
        "price": 0.30,
        "event": 0.30,
        "analog": 0.25,
        "regime": 0.15,
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BeliefScore:
    """Decomposed belief score for a single candidate.

    Attributes
    ----------
    ticker : str
        Candidate ticker symbol.
    belief_score : float
        Final combined score in [0, 1].
    conviction : float
        Uncertainty-adjusted score: belief_score / (1 + uncertainty).
    price_signal : float
        Normalised price-model signal [0, 1].
    event_strength : float
        Normalised event signal [0, 1].
    analog_prior : float
        Normalised analog memory signal [0, 1].
    regime_alignment : float
        Regime compatibility score [0, 1].
    uncertainty : float
        Aggregate uncertainty estimate ≥ 0; higher = less reliable.
    weights_used : dict
        The four weights applied.
    event_type : str
        Classified event type (or ``"unknown"``).
    mechanism_tags : list[str]
        Expected mechanism tags from the causal template.
    rank : int
        Rank among all candidates (1 = highest belief_score).
    signal : str
        BUY / HOLD / SELL derived from rank-based execution policy.
    metadata : dict
        Extra fields for traceability.
    """

    ticker: str
    belief_score: float
    conviction: float
    price_signal: float
    event_strength: float
    analog_prior: float
    regime_alignment: float
    uncertainty: float
    weights_used: Dict[str, float] = field(default_factory=dict)
    event_type: str = "unknown"
    mechanism_tags: List[str] = field(default_factory=list)
    rank: int = 0
    signal: str = "HOLD"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _normalize_price_signal(
    expected_return: float,
    score_raw: float,
    min_return: float = -0.40,
    max_return: float = 0.40,
) -> float:
    """Map expected_return ∈ [min_return, max_return] to [0, 1]."""
    span = max_return - min_return
    if span <= 0:
        return 0.5
    signal = (expected_return - min_return) / span
    return _clamp(signal)


def _regime_alignment(regime: str, mechanism_tags: Sequence[MechanismTag]) -> float:
    """Compute compatibility between market regime and event mechanism.

    Returns a score in [0, 1]:
    - RISK_ON + bullish mechanisms → high alignment
    - RISK_OFF + bearish mechanisms → moderate (we can still SHORT or AVOID)
    - PANIC → low alignment for most event types
    """
    tags = set(str(t) for t in mechanism_tags)
    bullish = {MechanismTag.DRIFT, MechanismTag.GAP, MechanismTag.MOMENTUM_CONTINUATION, MechanismTag.DELAYED_DRIFT}
    bearish = {MechanismTag.REVERSAL_RISK, MechanismTag.UNCERTAINTY, MechanismTag.LIQUIDITY_SHOCK}

    bullish_count = sum(1 for t in bullish if str(t) in tags)
    bearish_count = sum(1 for t in bearish if str(t) in tags)
    dominant = "bullish" if bullish_count >= bearish_count else "bearish"

    if regime in ("RISK_ON", "TRENDING"):
        return 0.85 if dominant == "bullish" else 0.35
    if regime in ("RISK_OFF", "PANIC"):
        return 0.20 if dominant == "bullish" else 0.60
    if regime == "EVENT_DRIVEN":
        return 0.70  # most events relevant in event-driven regime
    if regime == "CHOPPY":
        return 0.40 if dominant == "bullish" else 0.50
    return 0.50  # unknown regime


def _compute_uncertainty(
    news_confidence: float,
    analog_variance: float,
    vol_ratio: float,
    uncertainty_multiplier: float,
) -> float:
    """Aggregate uncertainty from three sources.

    Returns an uncertainty value ≥ 0.
    A value of 0 means maximum certainty.
    """
    news_ambiguity = 1.0 - _clamp(news_confidence)
    analog_unc = _clamp(analog_variance)  # already 0–1 range
    vol_spike = max(0.0, vol_ratio - 1.0)  # excess vol above baseline
    raw = (0.4 * news_ambiguity + 0.3 * analog_unc + 0.3 * vol_spike)
    return raw * uncertainty_multiplier


# ---------------------------------------------------------------------------
# BeliefEngine
# ---------------------------------------------------------------------------

class BeliefEngine:
    """Multi-tower belief scorer for all candidates in the universe.

    Parameters
    ----------
    regime_weights : dict, optional
        Override regime-adaptive weights.
    adaptive_fraction : float
        Fraction of ranked candidates to designate BUY (default 0.4).
    max_positions : int
        Hard cap on number of BUY signals.
    """

    def __init__(
        self,
        regime_weights: Optional[Dict[str, Dict[str, float]]] = None,
        adaptive_fraction: float = 0.40,
        max_positions: int = 20,
    ) -> None:
        self._weights = {**_DEFAULT_WEIGHTS, **(regime_weights or {})}
        self.adaptive_fraction = adaptive_fraction
        self.max_positions = max_positions

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_candidate(
        self,
        ticker: str,
        *,
        expected_return: float,
        score_raw: float,
        event_type: str = "unknown",
        event_confidence: float = 1.0,
        analog_prior: float = 0.5,
        analog_variance: float = 0.0,
        regime: str = "DEFAULT",
        vol_ratio: float = 1.0,
    ) -> BeliefScore:
        """Compute :class:`BeliefScore` for a single candidate.

        Parameters
        ----------
        ticker : str
            Candidate ticker.
        expected_return : float
            Annual expected return from the price model (e.g. 0.15 = 15%).
        score_raw : float
            Raw composite score from predictions (unnormalised).
        event_type : str
            Classified event type from event ontology.
        event_confidence : float
            Confidence in the event classification [0, 1].
        analog_prior : float
            Analog memory prior return normalised to [0, 1].
        analog_variance : float
            Variance of analog outcomes (0 = perfect consistency).
        regime : str
            Current market regime label.
        vol_ratio : float
            Ratio of current realised vol to historical vol (1 = normal).
        """
        template = get_causal_template(event_type)
        weights = self._weights.get(regime, self._weights["DEFAULT"])

        # Tower 1: price signal
        price_sig = _normalize_price_signal(expected_return, score_raw)

        # Tower 2: event strength
        event_str = _clamp(
            template.base_event_strength
            * event_confidence
            * (0.5 + 0.5 * (template.direction_bias + 1.0) / 2.0)
        )

        # Tower 3: analog prior (already normalised by caller)
        analog = _clamp(analog_prior)

        # Tower 4: regime alignment
        regime_align = _regime_alignment(regime, template.mechanism_tags)

        # Combine
        belief = (
            weights["price"] * price_sig
            + weights["event"] * event_str
            + weights["analog"] * analog
            + weights["regime"] * regime_align
        )
        belief = _clamp(belief)

        # Uncertainty
        uncertainty = _compute_uncertainty(
            event_confidence,
            analog_variance,
            vol_ratio,
            template.uncertainty_multiplier,
        )
        conviction = belief / (1.0 + uncertainty)

        return BeliefScore(
            ticker=ticker,
            belief_score=belief,
            conviction=conviction,
            price_signal=price_sig,
            event_strength=event_str,
            analog_prior=analog,
            regime_alignment=regime_align,
            uncertainty=uncertainty,
            weights_used=dict(weights),
            event_type=event_type,
            mechanism_tags=[str(t) for t in template.mechanism_tags],
            metadata={"vol_ratio": vol_ratio, "regime": regime},
        )

    # ------------------------------------------------------------------
    # Ranking & signal assignment
    # ------------------------------------------------------------------

    def rank_and_assign(self, scores: List[BeliefScore]) -> List[BeliefScore]:
        """Sort candidates by belief_score and assign BUY/HOLD signals.

        K = min(max_positions, ceil(adaptive_fraction * N))

        This guarantees at least one trade for N ≥ 1.
        """
        if not scores:
            return scores

        n = len(scores)
        k = min(self.max_positions, max(1, math.ceil(self.adaptive_fraction * n)))

        sorted_scores = sorted(scores, key=lambda s: s.belief_score, reverse=True)
        for rank, score in enumerate(sorted_scores, start=1):
            score.rank = rank
            score.signal = "BUY" if rank <= k else "HOLD"

        logger.debug(
            "BeliefEngine: %d candidates → BUY top %d (adaptive_fraction=%.2f)",
            n,
            k,
            self.adaptive_fraction,
        )
        return sorted_scores

    # ------------------------------------------------------------------
    # Convenience: score + rank in one call
    # ------------------------------------------------------------------

    def evaluate(
        self,
        candidates: Sequence[Mapping[str, Any]],
        regime: str = "DEFAULT",
    ) -> List[BeliefScore]:
        """Score and rank a list of candidate dicts.

        Each dict must contain at minimum:
            - ``ticker`` : str
            - ``expected_return_annual`` : float (price model output)
            - ``score`` : float (raw composite score)

        Optional fields (used when present):
            - ``event_type`` : str
            - ``event_confidence`` : float
            - ``analog_prior`` : float        (normalised 0–1)
            - ``analog_variance`` : float
            - ``vol_ratio`` : float

        Returns
        -------
        list[BeliefScore]
            Ranked and signal-annotated scores.
        """
        scored: List[BeliefScore] = []
        for c in candidates:
            bs = self.score_candidate(
                ticker=str(c.get("ticker", "UNKNOWN")),
                expected_return=_safe_float(
                c.get("expected_return_horizon",
                      c.get("expected_return",
                            c.get("predicted_return",
                                  c.get("expected_return_annual", 0.0))))
            ),
                score_raw=_safe_float(c.get("score", 0.0)),
                event_type=str(c.get("event_type", "unknown")),
                event_confidence=_safe_float(c.get("event_confidence", 1.0)),
                analog_prior=_safe_float(c.get("analog_prior", 0.5)),
                analog_variance=_safe_float(c.get("analog_variance", 0.0)),
                regime=regime,
                vol_ratio=_safe_float(c.get("vol_ratio", 1.0), default=1.0),
            )
            scored.append(bs)

        return self.rank_and_assign(scored)

    # ------------------------------------------------------------------
    # Online weight adaptation
    # ------------------------------------------------------------------

    def adapt_weights(
        self,
        regime: str,
        tower: str,
        direction: float,
        learning_rate: float = 0.05,
    ) -> None:
        """Update one tower's weight for a given regime.

        Called by the learning module when a trade outcome is observed.

        Parameters
        ----------
        regime : str
            Which regime's weight set to update.
        tower : str
            One of: ``price``, ``event``, ``analog``, ``regime``.
        direction : float
            +1 to increase tower weight; -1 to decrease.
        learning_rate : float
            Step size.
        """
        weights = self._weights.get(regime, self._weights["DEFAULT"])
        if tower not in weights:
            logger.warning("adapt_weights: unknown tower '%s'", tower)
            return

        weights[tower] = _clamp(weights[tower] + direction * learning_rate)
        # Renormalize to sum = 1
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total
        self._weights[regime] = weights
        logger.debug(
            "adapt_weights: regime=%s tower=%s → %.3f",
            regime,
            tower,
            weights[tower],
        )
