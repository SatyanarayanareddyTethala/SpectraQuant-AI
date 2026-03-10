"""Focused edge-case tests for EquityAllocator.

Covers:
- Negative blended_score → symbol excluded from allocation
- Signal below min_signal_threshold → symbol blocked
- max_weight invariant maintained after iterative re-normalisation
- Infeasible max_weight (too few symbols) → equal distribution fallback
- All-blocked input → empty result
- All-negative scores → empty result
- Single symbol → full weight
- Metadata fields populated correctly
- Signed-weight capping/normalisation bug guard
"""
from __future__ import annotations

import pytest

from spectraquant.equities.policy.allocator import AllocationResult, EquityAllocator
from spectraquant.equities.policy.meta_policy import PolicyDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision(
    sym: str,
    score: float = 0.5,
    confidence: float = 0.8,
    blocked: bool = False,
    block_reason: str = "",
) -> PolicyDecision:
    return PolicyDecision(
        canonical_symbol=sym,
        blended_score=score,
        confidence=confidence,
        blocked=blocked,
        block_reason=block_reason,
    )


def _allocator(max_weight: float = 0.25, min_signal_threshold: float = 0.05) -> EquityAllocator:
    return EquityAllocator(
        max_weight=max_weight,
        min_signal_threshold=min_signal_threshold,
    )


# ---------------------------------------------------------------------------
# Negative scores
# ---------------------------------------------------------------------------

class TestNegativeScoreExclusion:
    """Symbols with non-positive blended_score must not receive a weight."""

    def test_negative_score_not_in_target_weights(self) -> None:
        """Negative-score symbol must be absent from target weights."""
        alloc = _allocator()
        decisions = {
            "INFY.NS": _decision("INFY.NS", score=0.6),
            "TCS.NS": _decision("TCS.NS", score=-0.4),
        }
        result = alloc.allocate(decisions)
        assert "TCS.NS" not in result.target_weights

    def test_negative_score_positive_symbol_still_allocated(self) -> None:
        """Excluding a negative-score symbol must not prevent other symbols."""
        alloc = _allocator()
        decisions = {
            "INFY.NS": _decision("INFY.NS", score=0.6),
            "TCS.NS": _decision("TCS.NS", score=-0.4),
        }
        result = alloc.allocate(decisions)
        assert "INFY.NS" in result.target_weights
        assert result.target_weights["INFY.NS"] > 0

    def test_zero_score_excluded(self) -> None:
        alloc = _allocator(min_signal_threshold=0.05)
        decisions = {
            "INFY.NS": _decision("INFY.NS", score=0.6),
            "TCS.NS": _decision("TCS.NS", score=0.0),
        }
        result = alloc.allocate(decisions)
        assert "TCS.NS" not in result.target_weights

    def test_all_negative_scores_returns_empty(self) -> None:
        alloc = _allocator()
        decisions = {
            "INFY.NS": _decision("INFY.NS", score=-0.3),
            "TCS.NS": _decision("TCS.NS", score=-0.7),
        }
        result = alloc.allocate(decisions)
        assert result.target_weights == {}


# ---------------------------------------------------------------------------
# min_signal_threshold filtering
# ---------------------------------------------------------------------------

class TestSignalThreshold:
    """Symbols with |blended_score| < min_signal_threshold must be blocked."""

    def test_small_positive_score_blocked(self) -> None:
        alloc = _allocator(min_signal_threshold=0.10)
        decisions = {
            "INFY.NS": _decision("INFY.NS", score=0.6),
            "TCS.NS": _decision("TCS.NS", score=0.03),  # below threshold
        }
        result = alloc.allocate(decisions)
        assert "TCS.NS" not in result.target_weights
        assert "TCS.NS" in result.blocked_assets

    def test_small_negative_score_blocked(self) -> None:
        alloc = _allocator(min_signal_threshold=0.10)
        decisions = {
            "INFY.NS": _decision("INFY.NS", score=0.6),
            "TCS.NS": _decision("TCS.NS", score=-0.02),  # |score| below threshold
        }
        result = alloc.allocate(decisions)
        assert "TCS.NS" not in result.target_weights
        assert "TCS.NS" in result.blocked_assets

    def test_reason_code_set_for_threshold_blocked(self) -> None:
        alloc = _allocator(min_signal_threshold=0.10)
        decisions = {"INFY.NS": _decision("INFY.NS", score=0.03)}
        result = alloc.allocate(decisions)
        assert "INFY.NS" in result.reason_codes
        assert "signal_too_small" in result.reason_codes["INFY.NS"]

    def test_exactly_at_threshold_included(self) -> None:
        """Score exactly equal to threshold is not blocked because the
        implementation uses a strict < comparison (abs(score) < threshold)."""
        alloc = _allocator(min_signal_threshold=0.10)
        decisions = {"INFY.NS": _decision("INFY.NS", score=0.10)}
        result = alloc.allocate(decisions)
        # abs(0.10) < 0.10 is False → not excluded by threshold filter
        assert "INFY.NS" in result.target_weights


# ---------------------------------------------------------------------------
# max_weight invariant after iterative re-normalisation
# ---------------------------------------------------------------------------

class TestMaxWeightCapInvariant:
    """Weights must never exceed max_weight after the iterative cap."""

    def test_feasible_two_symbols_cap_enforced(self) -> None:
        """With max_weight=0.60, 2 symbols (n*0.60=1.20>=1): cap is feasible."""
        alloc = _allocator(max_weight=0.60)
        decisions = {
            "INFY.NS": _decision("INFY.NS", score=0.8, confidence=1.0),
            "TCS.NS": _decision("TCS.NS", score=0.2, confidence=1.0),
        }
        result = alloc.allocate(decisions)
        for sym, w in result.target_weights.items():
            assert w <= 0.60 + 1e-9, f"{sym} weight {w:.4f} exceeds max_weight=0.60"
        # Weights should still sum to 1.0
        assert abs(sum(result.target_weights.values()) - 1.0) < 1e-6

    def test_infeasible_cap_falls_back_to_equal_weight(self) -> None:
        """With max_weight=0.20 and only 2 symbols (n*0.20=0.40<1), cap cannot be
        enforced; equal distribution (0.50 each) is returned instead."""
        alloc = _allocator(max_weight=0.20)
        decisions = {
            "INFY.NS": _decision("INFY.NS", score=0.9, confidence=1.0),
            "TCS.NS": _decision("TCS.NS", score=0.1, confidence=1.0),
        }
        result = alloc.allocate(decisions)
        weights = list(result.target_weights.values())
        # Equal distribution: each symbol gets 1/n = 0.50
        for w in weights:
            assert abs(w - 0.5) < 1e-6, f"expected equal weight 0.50, got {w:.4f}"

    def test_max_weight_many_symbols(self) -> None:
        """With enough symbols, max_weight must be strictly enforced."""
        alloc = _allocator(max_weight=0.15)
        decisions = {
            f"SYM{i}.NS": _decision(f"SYM{i}.NS", score=0.3 + i * 0.05, confidence=0.8)
            for i in range(10)
        }
        result = alloc.allocate(decisions)
        for sym, w in result.target_weights.items():
            assert w <= 0.15 + 1e-9, f"{sym} weight {w:.4f} exceeds max_weight=0.15"

    def test_weights_sum_to_one(self) -> None:
        """Output weights must sum to 1.0 regardless of capping."""
        alloc = _allocator(max_weight=0.50)
        decisions = {
            "A.NS": _decision("A.NS", score=0.7),
            "B.NS": _decision("B.NS", score=0.3),
        }
        result = alloc.allocate(decisions)
        total = sum(result.target_weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_sum_of_weights_at_most_one(self) -> None:
        """Sum of positive weights must not exceed 1.0."""
        alloc = _allocator(max_weight=0.25)
        decisions = {
            f"SYM{i}.NS": _decision(f"SYM{i}.NS", score=0.1 + i * 0.1, confidence=0.9)
            for i in range(6)
        }
        result = alloc.allocate(decisions)
        total = sum(result.target_weights.values())
        assert total <= 1.0 + 1e-6

    def test_regression_single_pass_cap_normalized_above_max_weight(self) -> None:
        """Regression: original single-pass cap+normalize violated max_weight for
        under-diversified portfolios. Iterative approach must converge correctly."""
        alloc = _allocator(max_weight=0.20)
        # 10 symbols with highly skewed scores — the top symbol gets a heavy raw weight
        decisions = {
            f"SYM{i}.NS": _decision(
                f"SYM{i}.NS", score=0.1 if i > 0 else 5.0, confidence=1.0
            )
            for i in range(10)
        }
        result = alloc.allocate(decisions)
        for sym, w in result.target_weights.items():
            assert w <= 0.20 + 1e-9, (
                f"{sym} weight {w:.4f} exceeds max_weight=0.20"
            )


# ---------------------------------------------------------------------------
# All-blocked inputs
# ---------------------------------------------------------------------------

class TestAllBlockedInput:
    def test_all_explicitly_blocked(self) -> None:
        alloc = _allocator()
        decisions = {
            "INFY.NS": _decision("INFY.NS", blocked=True, block_reason="risk gate"),
            "TCS.NS": _decision("TCS.NS", blocked=True, block_reason="risk gate"),
        }
        result = alloc.allocate(decisions)
        assert result.target_weights == {}
        assert set(result.blocked_assets) == {"INFY.NS", "TCS.NS"}

    def test_reason_codes_populated_for_blocked(self) -> None:
        alloc = _allocator()
        decisions = {
            "INFY.NS": _decision("INFY.NS", blocked=True, block_reason="circuit_breaker"),
        }
        result = alloc.allocate(decisions)
        assert result.reason_codes["INFY.NS"] == "circuit_breaker"


# ---------------------------------------------------------------------------
# Single symbol
# ---------------------------------------------------------------------------

class TestSingleSymbol:
    def test_single_symbol_gets_full_weight(self) -> None:
        """A single eligible symbol receives 1.0 regardless of max_weight
        (the infeasibility fallback distributes equally → 1/1 = 1.0)."""
        alloc = _allocator(max_weight=0.20)
        decisions = {"INFY.NS": _decision("INFY.NS", score=0.7)}
        result = alloc.allocate(decisions)
        assert abs(result.target_weights["INFY.NS"] - 1.0) < 1e-6

    def test_single_symbol_weights_sum_to_one(self) -> None:
        alloc = _allocator(max_weight=0.50)
        decisions = {"INFY.NS": _decision("INFY.NS", score=0.7)}
        result = alloc.allocate(decisions)
        total = sum(result.target_weights.values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_n_allocated_populated(self) -> None:
        alloc = _allocator()
        decisions = {
            "A.NS": _decision("A.NS", score=0.6),
            "B.NS": _decision("B.NS", score=0.4),
        }
        result = alloc.allocate(decisions)
        assert result.metadata["n_allocated"] == 2

    def test_total_weight_sums_to_one(self) -> None:
        alloc = _allocator(max_weight=0.25)
        decisions = {
            "A.NS": _decision("A.NS", score=0.6),
            "B.NS": _decision("B.NS", score=0.4),
        }
        result = alloc.allocate(decisions)
        assert abs(result.metadata["total_weight"] - 1.0) < 1e-6

    def test_empty_result_no_metadata_crash(self) -> None:
        alloc = _allocator()
        decisions = {"A.NS": _decision("A.NS", score=-0.5)}
        result = alloc.allocate(decisions)
        # No metadata set when no eligible symbols — should not raise
        assert isinstance(result, AllocationResult)

