"""Equity portfolio allocator.

Converts policy decisions into target portfolio weights.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from spectraquant.equities.policy.meta_policy import PolicyDecision

logger = logging.getLogger(__name__)

# Maximum iterations for the iterative weight-cap algorithm.  For any
# feasible portfolio (n * max_weight >= 1) the algorithm converges in at
# most n steps in theory; 100 is a generous upper bound for practical use.
_MAX_CAP_ITERATIONS = 100


@dataclass
class AllocationResult:
    """Target weights produced by the equity allocator."""

    target_weights: dict[str, float] = field(default_factory=dict)
    blocked_assets: list[str] = field(default_factory=list)
    reason_codes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class EquityAllocator:
    """Convert policy decisions into target portfolio weights.

    Args:
        max_weight: Maximum weight per asset (default 0.20).
        min_signal_threshold: Minimum |blended_score| to include asset.
        volatility_target: Optional annualised portfolio volatility target.
    """

    def __init__(
        self,
        max_weight: float = 0.20,
        min_signal_threshold: float = 0.05,
        volatility_target: float | None = None,
    ) -> None:
        self._max_weight = max_weight
        self._min_signal_threshold = min_signal_threshold
        self._vol_target = volatility_target

    def allocate(
        self,
        decisions: dict[str, PolicyDecision],
    ) -> AllocationResult:
        """Convert decisions into target weights.

        Returns:
            AllocationResult with normalised weights summing to ≤ 1.0.
        """
        result = AllocationResult()

        # Select eligible symbols
        eligible: dict[str, float] = {}
        for sym, dec in decisions.items():
            if dec.blocked:
                result.blocked_assets.append(sym)
                result.reason_codes[sym] = dec.block_reason
                continue
            if abs(dec.blended_score) < self._min_signal_threshold:
                result.blocked_assets.append(sym)
                result.reason_codes[sym] = f"signal_too_small ({dec.blended_score:.4f})"
                continue
            if dec.blended_score > 0:
                eligible[sym] = dec.blended_score * dec.confidence

        if not eligible:
            logger.warning("No eligible symbols for equity allocation")
            return result

        # Normalise to sum = 1, cap at max_weight
        total = sum(eligible.values())
        raw_weights = {sym: v / total for sym, v in eligible.items()}

        # Apply max_weight cap iteratively
        weights = self._cap_weights(raw_weights)
        result.target_weights = weights
        result.metadata["n_allocated"] = len(weights)
        result.metadata["total_weight"] = sum(weights.values())
        return result

    def _cap_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply max_weight cap with iterative redistribution.

        Iterates until all normalised weights satisfy ``max_weight``.

        When the number of eligible symbols is too small to enforce
        ``max_weight`` while keeping weights summed to 1 (i.e.
        ``n * max_weight < 1``), an equal distribution is returned as the
        best feasible approximation.
        """
        if not weights:
            return {}
        n = len(weights)
        # Infeasible constraint: not enough symbols to enforce cap at sum = 1.
        if n * self._max_weight < 1.0 - 1e-9:
            return {k: 1.0 / n for k in weights}

        w = dict(weights)
        for _ in range(_MAX_CAP_ITERATIONS):  # converges well within the limit
            total = sum(w.values())
            if total <= 0:
                return w
            normed = {k: v / total for k, v in w.items()}
            if max(normed.values()) <= self._max_weight + 1e-9:
                return normed
            w = {k: min(v, self._max_weight) for k, v in normed.items()}
        # Normalise final state (loop exhaustion is not expected in practice).
        logger.warning(
            "_cap_weights: iteration limit (%d) reached; result may not satisfy max_weight",
            _MAX_CAP_ITERATIONS,
        )
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else w
