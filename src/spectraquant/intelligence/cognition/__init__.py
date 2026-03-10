"""SpectraQuant-AI-V3 Cognition Layer.

Provides causal, adaptive, news-aware trading intelligence through:

- causal_templates  : maps event types to expected market mechanisms
- belief_engine     : combines price, event, analog, and regime signals
- explanation_engine: generates human-readable explanations for every candidate

NEWS → EVENT → MARKET MECHANISM → PRICE RESPONSE → LEARNING
"""
from __future__ import annotations

from spectraquant.intelligence.cognition.belief_engine import BeliefEngine, BeliefScore
from spectraquant.intelligence.cognition.causal_templates import (
    CausalTemplate,
    MechanismTag,
    get_causal_template,
)
from spectraquant.intelligence.cognition.explanation_engine import (
    CandidateExplanation,
    ExplanationEngine,
)

__all__ = [
    "BeliefEngine",
    "BeliefScore",
    "CausalTemplate",
    "CandidateExplanation",
    "ExplanationEngine",
    "MechanismTag",
    "get_causal_template",
]
