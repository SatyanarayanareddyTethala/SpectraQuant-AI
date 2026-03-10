"""Equity meta-policy layer.

Blends agent signals into a single consensus signal per symbol,
applying regime-aware scaling and risk gates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from spectraquant.core.enums import SignalStatus
from spectraquant.equities.signals._base_agent import AgentOutput

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    """Blended signal decision for a single equity symbol."""

    canonical_symbol: str
    blended_score: float = 0.0
    confidence: float = 0.0
    contributing_agents: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class EquityMetaPolicy:
    """Blend agent outputs into policy decisions for the equity pipeline.

    Args:
        agent_weights: Optional mapping of agent_id → weight.
            Defaults to equal weighting.
        min_confidence: Minimum blended confidence to emit a non-zero signal.
        regime_scale: Scale factor applied when regime is bearish (negative score).
    """

    def __init__(
        self,
        agent_weights: dict[str, float] | None = None,
        min_confidence: float = 0.2,
        regime_scale: float = 0.5,
    ) -> None:
        self._agent_weights = agent_weights or {}
        self._min_confidence = min_confidence
        self._regime_scale = regime_scale

    def blend(
        self,
        agent_outputs: list[AgentOutput],
        symbol: str,
    ) -> PolicyDecision:
        """Blend a list of agent outputs for *symbol* into a policy decision.

        Returns:
            PolicyDecision with blended_score in [-1, 1].
        """
        valid = [o for o in agent_outputs if o.status == SignalStatus.OK]

        if not valid:
            reasons = list({o.error_reason for o in agent_outputs if o.error_reason})
            return PolicyDecision(
                canonical_symbol=symbol,
                blended_score=0.0,
                confidence=0.0,
                blocked=True,
                block_reason=f"No valid signals. Reasons: {reasons}",
            )

        # Weighted blend
        total_weight = 0.0
        weighted_score = 0.0
        weighted_conf = 0.0
        contributing = []

        for out in valid:
            w = self._agent_weights.get(out.agent_id, 1.0)
            weighted_score += out.signal_score * w * out.confidence
            weighted_conf += out.confidence * w
            total_weight += w
            contributing.append(out.agent_id)

        if total_weight == 0 or weighted_conf == 0:
            return PolicyDecision(
                canonical_symbol=symbol,
                blocked=True,
                block_reason="Zero total weight/confidence",
            )

        blended = weighted_score / weighted_conf
        blended = max(-1.0, min(1.0, blended))
        conf = weighted_conf / total_weight

        # Apply regime scaling if regime agent is very negative
        regime_outputs = [o for o in valid if "regime" in o.agent_id]
        if regime_outputs and regime_outputs[0].signal_score < -0.2:
            blended *= self._regime_scale

        if conf < self._min_confidence:
            return PolicyDecision(
                canonical_symbol=symbol,
                blended_score=blended,
                confidence=conf,
                contributing_agents=contributing,
                blocked=True,
                block_reason=f"Confidence too low: {conf:.3f} < {self._min_confidence}",
            )

        return PolicyDecision(
            canonical_symbol=symbol,
            blended_score=blended,
            confidence=conf,
            contributing_agents=contributing,
            metadata={"n_agents": len(valid)},
        )

    def run(
        self,
        signals_by_symbol: dict[str, list[AgentOutput]],
    ) -> dict[str, PolicyDecision]:
        """Run meta-policy for all symbols."""
        return {
            sym: self.blend(outputs, sym)
            for sym, outputs in signals_by_symbol.items()
        }
