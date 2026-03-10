"""Tests for portfolio allocation and constraints."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectraquant.portfolio.allocator import (
    allocate,
    allocate_risk_parity,
    allocate_vol_target,
)
from spectraquant.portfolio.constraints import (
    PortfolioConstraints,
    apply_constraints,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Synthetic daily returns for 3 assets over 100 days."""
    np.random.seed(42)
    n = 100
    ret = pd.DataFrame(
        {
            "BTC": np.random.normal(0.001, 0.03, n),
            "ETH": np.random.normal(0.0005, 0.04, n),
            "SOL": np.random.normal(0.0, 0.05, n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
    )
    return ret


@pytest.fixture
def sample_cov(sample_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized covariance matrix."""
    return sample_returns.cov() * 365


class TestRiskParity:
    """Risk parity allocation tests."""

    def test_weights_sum_to_one(self, sample_cov: pd.DataFrame) -> None:
        w = allocate_risk_parity(sample_cov)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_lower_vol_gets_higher_weight(self, sample_cov: pd.DataFrame) -> None:
        w = allocate_risk_parity(sample_cov)
        # BTC has lowest vol (0.03) so should get highest weight
        assert w["BTC"] > w["SOL"]

    def test_all_positive_weights(self, sample_cov: pd.DataFrame) -> None:
        w = allocate_risk_parity(sample_cov)
        assert (w > 0).all()


class TestVolTarget:
    """Volatility targeting tests."""

    def test_nonzero_output_with_valid_scores(
        self, sample_cov: pd.DataFrame
    ) -> None:
        scores = pd.Series({"BTC": 0.5, "ETH": -0.2, "SOL": 0.3})
        w = allocate_vol_target(scores, sample_cov, target_vol=0.15)
        assert w.abs().sum() > 0

    def test_zero_scores_fallback(self, sample_cov: pd.DataFrame) -> None:
        """All-zero scores should fall back to equal weights."""
        scores = pd.Series({"BTC": 0.0, "ETH": 0.0, "SOL": 0.0})
        w = allocate_vol_target(scores, sample_cov, target_vol=0.15)
        assert w.abs().sum() > 0  # Should NOT be all zeros

    def test_negative_score_produces_negative_weight(
        self, sample_cov: pd.DataFrame
    ) -> None:
        """A negative score must yield a negative (short) weight."""
        scores = pd.Series({"BTC": -0.5, "ETH": 0.0, "SOL": 0.0})
        w = allocate_vol_target(scores, sample_cov, target_vol=0.15)
        assert w["BTC"] < 0, "negative score should produce a negative weight"

    def test_mixed_sign_scores_preserve_signs(
        self, sample_cov: pd.DataFrame
    ) -> None:
        """Signs of scores must be preserved in output weights."""
        scores = pd.Series({"BTC": 0.5, "ETH": -0.3, "SOL": 0.2})
        w = allocate_vol_target(scores, sample_cov, target_vol=0.15)
        assert w["BTC"] > 0
        assert w["ETH"] < 0
        assert w["SOL"] > 0

    def test_all_negative_scores_produce_negative_weights(
        self, sample_cov: pd.DataFrame
    ) -> None:
        """All-negative scores should yield all-negative (short) weights."""
        scores = pd.Series({"BTC": -0.5, "ETH": -0.3, "SOL": -0.2})
        w = allocate_vol_target(scores, sample_cov, target_vol=0.15)
        assert (w < 0).all(), "all negative scores should yield all negative weights"

    def test_vol_scaling_applied(self, sample_cov: pd.DataFrame) -> None:
        """Higher target_vol should produce proportionally larger weights."""
        scores = pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.2})
        w_low = allocate_vol_target(scores, sample_cov, target_vol=0.10)
        w_high = allocate_vol_target(scores, sample_cov, target_vol=0.30)
        assert w_high.abs().sum() > w_low.abs().sum()


class TestAllocateDispatcher:
    """High-level allocate() function tests."""

    def test_risk_parity_method(self, sample_returns: pd.DataFrame) -> None:
        scores = pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.1})
        w = allocate(scores, sample_returns, method="risk_parity")
        assert abs(w.sum() - 1.0) < 1e-9

    def test_vol_target_method(self, sample_returns: pd.DataFrame) -> None:
        scores = pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.1})
        w = allocate(scores, sample_returns, method="vol_target", target_vol=0.15)
        assert w.abs().sum() > 0

    def test_no_overlap_returns_empty(self) -> None:
        scores = pd.Series({"DOGE": 0.5})
        returns = pd.DataFrame({"BTC": [0.01, -0.01]})
        w = allocate(scores, returns)
        assert len(w) == 0


class TestConstraints:
    """Portfolio constraint enforcement tests."""

    def test_max_weight_respected(self) -> None:
        w = pd.Series({"BTC": 0.6, "ETH": 0.3, "SOL": 0.1})
        c = PortfolioConstraints(max_weight=0.25, max_gross_leverage=1.0)
        cw = apply_constraints(w, c)
        assert cw.max() <= 0.25 + 1e-9

    def test_gross_leverage_respected(self) -> None:
        w = pd.Series({"BTC": 0.5, "ETH": 0.4, "SOL": 0.3})
        c = PortfolioConstraints(max_weight=1.0, max_gross_leverage=1.0)
        cw = apply_constraints(w, c)
        assert cw.abs().sum() <= 1.0 + 1e-9

    def test_max_positions(self) -> None:
        w = pd.Series({"BTC": 0.3, "ETH": 0.3, "SOL": 0.2, "DOGE": 0.1, "AVAX": 0.1})
        c = PortfolioConstraints(max_positions=3, max_weight=1.0, max_gross_leverage=5.0)
        cw = apply_constraints(w, c)
        assert (cw != 0).sum() <= 3

    def test_no_all_zeros_from_valid_input(self) -> None:
        """Constraints should not zero out all weights when input is valid."""
        w = pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.2})
        c = PortfolioConstraints(max_weight=0.25, max_gross_leverage=1.0)
        cw = apply_constraints(w, c)
        assert cw.abs().sum() > 0

    def test_turnover_cap(self) -> None:
        prev = pd.Series({"BTC": 0.4, "ETH": 0.3, "SOL": 0.3})
        new = pd.Series({"BTC": 0.1, "ETH": 0.5, "SOL": 0.4})
        c = PortfolioConstraints(
            max_weight=1.0,
            max_gross_leverage=5.0,
            max_turnover=0.2,
        )
        cw = apply_constraints(new, c, prev_weights=prev)
        turnover = (cw - prev).abs().sum()
        assert turnover <= 0.2 + 1e-9

    def test_negative_weights_clipped_to_min_weight(self) -> None:
        """Weights below min_weight must be clipped upward."""
        w = pd.Series({"BTC": -0.5, "ETH": 0.3, "SOL": -0.1})
        c = PortfolioConstraints(min_weight=-0.20, max_weight=1.0, max_gross_leverage=5.0)
        cw = apply_constraints(w, c)
        assert cw.min() >= -0.20 - 1e-9

    def test_gross_leverage_enforced_with_mixed_sign_weights(self) -> None:
        """Gross leverage must use sum(|w|) so sign cancellation cannot mask violations."""
        w = pd.Series({"BTC": 0.5, "ETH": -0.4, "SOL": 0.3})
        c = PortfolioConstraints(
            min_weight=-1.0, max_weight=1.0, max_gross_leverage=0.8
        )
        cw = apply_constraints(w, c)
        assert cw.abs().sum() <= 0.8 + 1e-9

    def test_max_weight_clip_does_not_affect_negative_side(self) -> None:
        """max_weight clips positive weights; min_weight clips negative weights."""
        w = pd.Series({"BTC": 0.8, "ETH": -0.8})
        c = PortfolioConstraints(min_weight=-0.25, max_weight=0.25, max_gross_leverage=2.0)
        cw = apply_constraints(w, c)
        assert cw["BTC"] <= 0.25 + 1e-9
        assert cw["ETH"] >= -0.25 - 1e-9
