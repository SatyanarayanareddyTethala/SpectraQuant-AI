from __future__ import annotations

import pytest

from spectraquant_v3.strategies.allocators.rank_vol_target_allocator import (
    RankVolTargetAllocator,
)


def test_rank_to_base_weights_inverse_rank_with_confidence():
    alloc = RankVolTargetAllocator()
    ranked = {
        "A": {"rank": 1, "confidence": 1.0, "vol": 0.2},
        "B": {"rank": 2, "confidence": 0.5, "vol": 0.2},
    }
    base = alloc._ranks_to_base_weights(ranked, ["A", "B"])
    assert base["A"] == pytest.approx(1.0)
    assert base["B"] == pytest.approx(0.25)


def test_normalize_weights_unit_gross():
    alloc = RankVolTargetAllocator()
    out = alloc._normalize_weights({"A": 2.0, "B": 1.0}, target_gross=1.0)
    assert out["A"] == pytest.approx(2 / 3)
    assert out["B"] == pytest.approx(1 / 3)
    assert sum(abs(v) for v in out.values()) == pytest.approx(1.0)


def test_apply_vol_targeting_uses_fallback_for_missing_vol():
    alloc = RankVolTargetAllocator(target_vol=0.10, missing_vol=0.25)
    w = {"A": 0.6, "B": 0.4}
    ranked = {
        "A": {"rank": 1, "confidence": 1.0, "vol": None},
        "B": {"rank": 2, "confidence": 1.0, "vol": 0.20},
    }
    scaled, diag = alloc._apply_vol_targeting(w, ranked)
    assert diag["portfolio_vol"] > 0
    assert 0 < diag["scale"] <= alloc.max_gross_leverage
    assert scaled["A"] == pytest.approx(w["A"] * diag["scale"])


def test_clip_weights_caps_single_name():
    alloc = RankVolTargetAllocator(max_weight=0.20)
    clipped, clipped_symbols = alloc._clip_weights({"A": 0.45, "B": 0.10})
    assert clipped["A"] == pytest.approx(0.20)
    assert clipped["B"] == pytest.approx(0.10)
    assert clipped_symbols == ["A"]


def test_drop_tiny_positions_then_renormalize():
    alloc = RankVolTargetAllocator(min_tradable_weight=0.18)
    final, diag = alloc.allocate(
        {
            "A": {"rank": 1, "confidence": 1.0, "vol": 0.30},
            "B": {"rank": 2, "confidence": 1.0, "vol": 0.30},
            "C": {"rank": 3, "confidence": 1.0, "vol": 0.30},
        }
    )
    assert "C" in diag["dropped_symbols"]
    assert set(final) == {"A", "B"}
    assert sum(abs(v) for v in final.values()) == pytest.approx(sum(abs(v) for v in diag["stage_thresholded"].values()))


def test_single_symbol_edge_case():
    alloc = RankVolTargetAllocator(target_vol=0.15, max_weight=1.0)
    final, diag = alloc.allocate({"ONLY": {"rank": 1, "confidence": 1.0, "vol": 0.2}})
    assert set(final) == {"ONLY"}
    assert final["ONLY"] > 0
    assert diag["risk"]["portfolio_vol"] > 0


def test_ties_are_deterministic_via_symbol_ordering():
    alloc = RankVolTargetAllocator()
    inp = {
        "B": {"rank": 1, "confidence": 1.0, "vol": 0.2},
        "A": {"rank": 1, "confidence": 1.0, "vol": 0.2},
    }
    out1, _ = alloc.allocate(inp)
    out2, _ = alloc.allocate(dict(reversed(list(inp.items()))))
    assert out1 == out2
    assert out1["A"] == pytest.approx(out1["B"])


def test_all_clipped_edge_case_results_in_capped_weights():
    alloc = RankVolTargetAllocator(target_vol=1.0, max_weight=0.10, max_gross_leverage=1.0)
    final, diag = alloc.allocate(
        {
            "A": {"rank": 1, "confidence": 1.0, "vol": 0.01},
            "B": {"rank": 2, "confidence": 1.0, "vol": 0.01},
            "C": {"rank": 3, "confidence": 1.0, "vol": 0.01},
        }
    )
    assert all(abs(v) <= 0.10 + 1e-12 for v in final.values())
    assert len(diag["clipped_symbols"]) >= 1


def test_invalid_rank_raises():
    alloc = RankVolTargetAllocator()
    with pytest.raises(ValueError):
        alloc.allocate({"A": {"rank": 0, "confidence": 1.0, "vol": 0.2}})
