"""Portfolio allocator skeleton for SpectraQuant-AI-V3.

Translates meta-policy decisions into target portfolio weights.

Two allocation modes:
- ``equal_weight``  – equal positive weight for all passed symbols.
- ``vol_target``    – weight inversely proportional to realised volatility
                      scaled to a target portfolio volatility.

All weights are constrained by ``max_weight`` and sum to ≤ 1.
Blocked symbols always receive a weight of 0.

This is a skeleton – live execution, transaction cost models, and
position sizing beyond the basics are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spectraquant_v3.core.schema import AllocationRow
from spectraquant_v3.pipeline.meta_policy import PolicyDecision


@dataclass
class AllocatorConfig:
    """Configuration for the portfolio allocator."""

    mode: str = "equal_weight"
    """Allocation mode: ``equal_weight`` or ``vol_target``."""

    max_weight: float = 0.20
    """Maximum weight per symbol."""

    max_gross_leverage: float = 1.0
    """Sum of all weights must not exceed this value."""

    target_vol: float = 0.15
    """Target annualised portfolio volatility (used in ``vol_target`` mode)."""

    min_weight: float = 0.0
    """Minimum weight per symbol (0 = allow exclusion)."""

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AllocatorConfig":
        """Build from merged pipeline config."""
        portfolio_cfg = cfg.get("portfolio", {})
        return cls(
            mode=portfolio_cfg.get("allocator", "equal_weight"),
            max_weight=float(portfolio_cfg.get("max_weight", 0.20)),
            max_gross_leverage=float(portfolio_cfg.get("max_gross_leverage", 1.0)),
            target_vol=float(portfolio_cfg.get("target_vol", 0.15)),
        )


class Allocator:
    """Converts meta-policy decisions to target portfolio weights.

    Args:
        config: :class:`AllocatorConfig` controlling allocation behaviour.
        run_id: Parent run identifier.
    """

    def __init__(self, config: AllocatorConfig, run_id: str = "unknown") -> None:
        self.config = config
        self.run_id = run_id

    @classmethod
    def from_config(cls, cfg: dict[str, Any], run_id: str = "unknown") -> "Allocator":
        """Build from merged pipeline config."""
        return cls(config=AllocatorConfig.from_config(cfg), run_id=run_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        decisions: list[PolicyDecision],
        vol_map: dict[str, float] | None = None,
    ) -> list[AllocationRow]:
        """Convert policy decisions to target weights.

        Args:
            decisions: List of :class:`~spectraquant_v3.pipeline.meta_policy.PolicyDecision`
                       objects from the meta-policy stage.
            vol_map:   Optional dict mapping canonical symbol → annualised
                       realised volatility.  Required for ``vol_target`` mode.

        Returns:
            List of :class:`~spectraquant_v3.core.schema.AllocationRow` objects,
            one per symbol (including blocked symbols with weight=0).
        """
        passed = [d for d in decisions if d.passed]
        blocked = [d for d in decisions if not d.passed]

        if self.config.mode == "vol_target" and vol_map:
            raw_weights = self._vol_target_weights(passed, vol_map)
        else:
            raw_weights = self._equal_weights(passed)

        # Apply max_weight constraint using |weight| so negative (short) weights
        # are capped symmetrically. Then enforce max_gross_leverage on sum(|w|).
        capped: dict[str, float] = {}
        for sym, w in raw_weights.items():
            if abs(w) > self.config.max_weight:
                capped[sym] = self.config.max_weight * (1.0 if w >= 0 else -1.0)
            else:
                capped[sym] = w

        total_abs = sum(abs(w) for w in capped.values())
        if total_abs > self.config.max_gross_leverage and total_abs > 0:
            factor = self.config.max_gross_leverage / total_abs
            capped = {sym: w * factor for sym, w in capped.items()}

        rows: list[AllocationRow] = []
        for d in passed:
            rows.append(
                AllocationRow(
                    run_id=self.run_id,
                    canonical_symbol=d.canonical_symbol,
                    asset_class=d.asset_class,
                    target_weight=round(capped.get(d.canonical_symbol, 0.0), 6),
                    blocked=False,
                )
            )
        for d in blocked:
            rows.append(
                AllocationRow(
                    run_id=self.run_id,
                    canonical_symbol=d.canonical_symbol,
                    asset_class=d.asset_class,
                    target_weight=0.0,
                    blocked=True,
                    blocked_reason=d.reason,
                )
            )
        return rows

    # ------------------------------------------------------------------
    # Weighting schemes
    # ------------------------------------------------------------------

    def _equal_weights(
        self, decisions: list[PolicyDecision]
    ) -> dict[str, float]:
        """Return equal positive weights for all passed decisions."""
        if not decisions:
            return {}
        w = 1.0 / len(decisions)
        return {d.canonical_symbol: w for d in decisions}

    def _vol_target_weights(
        self,
        decisions: list[PolicyDecision],
        vol_map: dict[str, float],
    ) -> dict[str, float]:
        """Return inverse-volatility weights scaled to target portfolio vol.

        Symbols with missing, non-finite, or zero vol fall back to equal weight.
        """
        if not decisions:
            return {}

        import math

        raw: dict[str, float] = {}
        default_vol = 0.20  # fallback when vol is missing, zero, or non-finite
        for d in decisions:
            vol = vol_map.get(d.canonical_symbol, 0.0)
            if not math.isfinite(vol) or vol <= 0:
                vol = default_vol
            # Weight proportional to signal score × inverse vol
            score_sign = 1.0 if d.composite_score >= 0 else -1.0
            raw[d.canonical_symbol] = score_sign * (self.config.target_vol / vol)

        # Normalise so sum of |weights| ≤ max_gross_leverage
        total_abs = sum(abs(w) for w in raw.values())
        if total_abs > self.config.max_gross_leverage:
            factor = self.config.max_gross_leverage / total_abs
            raw = {sym: w * factor for sym, w in raw.items()}

        return raw
