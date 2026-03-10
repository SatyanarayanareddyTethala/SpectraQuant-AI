"""Explanation Engine — human-readable rationale for every candidate.

Every candidate that passes through the V3 pipeline receives a structured
explanation showing:

- What event was detected (event_type)
- What market mechanism is expected (mechanism_tags)
- What historical analogs support the view (analog_examples)
- How confident each tower is (confidence_components)
- The final natural-language reason (final_reason)

Example output
--------------
"BUY because:
 earnings_beat detected (confidence: 0.88),
 14 similar events avg +6.2% 5d drift,
 regime=RISK_ON (strong alignment),
 momentum_continuation + drift mechanisms expected."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from spectraquant.intelligence.cognition.belief_engine import BeliefScore
from spectraquant.intelligence.cognition.causal_templates import get_causal_template

logger = logging.getLogger(__name__)

__all__ = ["CandidateExplanation", "ExplanationEngine"]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CandidateExplanation:
    """Full explanation record for one candidate.

    Attributes
    ----------
    ticker : str
    signal : str
        BUY / HOLD / SELL
    event_type : str
        Classified event type.
    mechanism_tags : list[str]
        Expected market mechanisms.
    analog_examples : list[dict]
        Up to N representative analog events with their outcomes.
    confidence_components : dict
        Per-tower confidence breakdown (price, event, analog, regime).
    final_reason : str
        Natural-language summary of why this signal was generated.
    belief_score : float
    conviction : float
    uncertainty : float
    """

    ticker: str
    signal: str
    event_type: str
    mechanism_tags: List[str]
    analog_examples: List[Dict[str, Any]]
    confidence_components: Dict[str, float]
    final_reason: str
    belief_score: float
    conviction: float
    uncertainty: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ExplanationEngine
# ---------------------------------------------------------------------------

class ExplanationEngine:
    """Generates :class:`CandidateExplanation` objects for a list of beliefs.

    Parameters
    ----------
    max_analog_examples : int
        Maximum number of analog examples to include in explanations.
    """

    def __init__(self, max_analog_examples: int = 5) -> None:
        self.max_analog_examples = max_analog_examples

    def explain(
        self,
        belief: BeliefScore,
        analog_examples: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CandidateExplanation:
        """Build a :class:`CandidateExplanation` for one candidate.

        Parameters
        ----------
        belief : BeliefScore
            The belief score computed by :class:`BeliefEngine`.
        analog_examples : list[dict], optional
            Retrieved analog events (from AnalogMemory).  Each dict may
            contain ``event_type``, ``ticker``, ``observed_return``,
            ``horizon_days`` fields.
        """
        examples = list(analog_examples or [])[:self.max_analog_examples]
        template = get_causal_template(belief.event_type)

        confidence_components = {
            "price_signal": round(belief.price_signal, 4),
            "event_strength": round(belief.event_strength, 4),
            "analog_prior": round(belief.analog_prior, 4),
            "regime_alignment": round(belief.regime_alignment, 4),
        }

        final_reason = self._build_reason(belief, examples, template.description)

        return CandidateExplanation(
            ticker=belief.ticker,
            signal=belief.signal,
            event_type=belief.event_type,
            mechanism_tags=belief.mechanism_tags,
            analog_examples=examples,
            confidence_components=confidence_components,
            final_reason=final_reason,
            belief_score=round(belief.belief_score, 4),
            conviction=round(belief.conviction, 4),
            uncertainty=round(belief.uncertainty, 4),
            metadata=dict(belief.metadata),
        )

    def explain_all(
        self,
        beliefs: Sequence[BeliefScore],
        analogs_by_ticker: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
    ) -> List[CandidateExplanation]:
        """Build explanations for all candidates.

        Parameters
        ----------
        beliefs : list[BeliefScore]
        analogs_by_ticker : dict, optional
            Maps ticker → list of analog dicts.
        """
        analogs_map = analogs_by_ticker or {}
        return [
            self.explain(b, analogs_map.get(b.ticker))
            for b in beliefs
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_reason(
        self,
        belief: BeliefScore,
        examples: Sequence[Dict[str, Any]],
        template_description: str,
    ) -> str:
        """Build a natural-language reason string."""
        parts: List[str] = []

        # Signal
        parts.append(f"{belief.signal}")

        # Event
        if belief.event_type and belief.event_type != "unknown":
            confidence_str = f"{belief.event_strength:.0%}"
            parts.append(
                f"{belief.event_type} detected (event_strength: {confidence_str})"
            )
        else:
            parts.append("no specific event detected")

        # Analog evidence
        if examples:
            returns = [
                e.get("observed_return", e.get("observed_return_shortterm"))
                for e in examples
                if e.get("observed_return", e.get("observed_return_shortterm")) is not None
            ]
            if returns:
                avg_ret = sum(returns) / len(returns)
                horizon = examples[0].get("horizon_days", 5)
                parts.append(
                    f"{len(examples)} similar past events avg {avg_ret:+.1%} "
                    f"{horizon}d outcome"
                )
        else:
            parts.append("no analog history available")

        # Regime
        regime = belief.metadata.get("regime", "UNKNOWN")
        regime_str = _describe_regime_alignment(belief.regime_alignment)
        parts.append(f"regime={regime} ({regime_str})")

        # Mechanisms
        if belief.mechanism_tags:
            mechanisms = " + ".join(belief.mechanism_tags[:3])
            parts.append(f"{mechanisms} expected")

        # Causal description
        if template_description:
            parts.append(template_description)

        return "because: " + ", ".join(parts[1:]) + "."


def _describe_regime_alignment(alignment: float) -> str:
    if alignment >= 0.75:
        return "strong alignment"
    if alignment >= 0.50:
        return "moderate alignment"
    if alignment >= 0.30:
        return "weak alignment"
    return "poor alignment"
