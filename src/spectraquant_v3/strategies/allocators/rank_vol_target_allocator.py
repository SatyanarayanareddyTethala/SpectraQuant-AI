"""Rank-based volatility-targeting allocator.

This allocator expects *ranked* signals with confidence and volatility metadata
for each symbol and produces deterministic long-only target weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import math

if TYPE_CHECKING:
    from spectraquant_v3.core.schema import AllocationRow
    from spectraquant_v3.pipeline.meta_policy import PolicyDecision


@dataclass
class RankVolTargetAllocator:
    """Allocate weights from ranked signals with portfolio-level vol targeting."""

    target_vol: float = 0.15
    max_weight: float = 0.20
    max_gross_leverage: float = 1.0
    min_tradable_weight: float = 0.0
    missing_vol: float = 0.20
    run_id: str = "unknown"

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        run_id: str | None = None,
    ) -> "RankVolTargetAllocator":
        portfolio = cfg.get("portfolio", {})
        return cls(
            target_vol=float(portfolio.get("target_vol", 0.15)),
            max_weight=float(portfolio.get("max_weight", 0.20)),
            max_gross_leverage=float(portfolio.get("max_gross_leverage", 1.0)),
            min_tradable_weight=float(portfolio.get("min_weight", 0.0)),
            run_id=run_id or "unknown",
        )

    def allocate(
        self,
        ranked_signals: dict[str, dict[str, float]],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Allocate deterministic target weights from ranked signal metadata.

        Input schema per symbol:
          {"rank": int/float, "confidence": float, "vol": float}
        """
        symbols = sorted(ranked_signals)
        base_weights = self._ranks_to_base_weights(ranked_signals, symbols)
        normalized = self._normalize_weights(base_weights, target_gross=1.0)
        vol_targeted, risk_diag = self._apply_vol_targeting(normalized, ranked_signals)
        clipped, clipped_symbols = self._clip_weights(vol_targeted)
        thresholded, dropped_symbols = self._drop_tiny_positions(clipped)
        final = self._normalize_weights(
            thresholded,
            target_gross=min(self.max_gross_leverage, sum(abs(v) for v in thresholded.values())),
        )

        diagnostics: dict[str, Any] = {
            "stage_base": base_weights,
            "stage_normalized": normalized,
            "stage_vol_targeted": vol_targeted,
            "stage_clipped": clipped,
            "stage_thresholded": thresholded,
            "risk": risk_diag,
            "clipped_symbols": clipped_symbols,
            "dropped_symbols": dropped_symbols,
        }
        return final, diagnostics

    def allocate_decisions(
        self,
        decisions: list[PolicyDecision],
        vol_map: dict[str, float] | None = None,
    ) -> list[AllocationRow]:
        """Uniform dispatch interface compatible with :class:`~spectraquant_v3.pipeline.allocator.Allocator`.

        Converts :class:`~spectraquant_v3.pipeline.meta_policy.PolicyDecision` objects
        into the ``ranked_signals`` format expected by :meth:`allocate`, runs the
        allocation, and wraps the output as
        :class:`~spectraquant_v3.core.schema.AllocationRow` objects.

        Passed symbols are ranked by descending ``|composite_score|``.
        Blocked symbols are included with ``target_weight=0``.
        """
        from spectraquant_v3.core.schema import AllocationRow

        vol_map = vol_map or {}
        passed = sorted(
            (d for d in decisions if d.passed),
            key=lambda d: abs(d.composite_score),
            reverse=True,
        )

        ranked_input = {
            d.canonical_symbol: {
                "rank": float(i + 1),
                "confidence": d.composite_confidence,
                "vol": vol_map.get(d.canonical_symbol, 0.0),
            }
            for i, d in enumerate(passed)
        }

        weights: dict[str, float]
        if ranked_input:
            weights, _ = self.allocate(ranked_input)
        else:
            weights = {}

        rows: list[AllocationRow] = []
        for d in decisions:
            rows.append(
                AllocationRow(
                    run_id=self.run_id,
                    canonical_symbol=d.canonical_symbol,
                    asset_class=d.asset_class,
                    target_weight=float(weights.get(d.canonical_symbol, 0.0)),
                    blocked=not d.passed,
                    blocked_reason="" if d.passed else d.reason,
                )
            )
        return rows

    def _ranks_to_base_weights(
        self,
        ranked_signals: dict[str, dict[str, float]],
        symbols: list[str],
    ) -> dict[str, float]:
        base: dict[str, float] = {}
        for sym in symbols:
            payload = ranked_signals[sym]
            rank = float(payload["rank"])
            confidence = float(payload.get("confidence", 1.0))
            if not math.isfinite(rank) or rank <= 0:
                raise ValueError(f"Invalid rank for {sym}: {rank!r}")
            if not math.isfinite(confidence) or confidence < 0:
                raise ValueError(f"Invalid confidence for {sym}: {confidence!r}")
            base[sym] = confidence / rank
        return base

    def _normalize_weights(
        self,
        weights: dict[str, float],
        target_gross: float,
    ) -> dict[str, float]:
        if not weights:
            return {}
        gross = sum(abs(v) for v in weights.values())
        if gross <= 0 or target_gross <= 0:
            return {sym: 0.0 for sym in sorted(weights)}
        factor = target_gross / gross
        return {sym: weights[sym] * factor for sym in sorted(weights)}

    def _apply_vol_targeting(
        self,
        weights: dict[str, float],
        ranked_signals: dict[str, dict[str, float]],
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not weights:
            return {}, {"portfolio_vol": 0.0, "scale": 0.0}

        exposures: list[float] = []
        for sym in sorted(weights):
            vol = self._safe_vol(ranked_signals[sym].get("vol"))
            exposures.append(weights[sym] * vol)

        portfolio_vol = math.sqrt(sum(x * x for x in exposures))
        if portfolio_vol <= 0:
            return {sym: 0.0 for sym in sorted(weights)}, {
                "portfolio_vol": 0.0,
                "scale": 0.0,
            }

        scale = min(self.max_gross_leverage, self.target_vol / portfolio_vol)
        scaled = {sym: weights[sym] * scale for sym in sorted(weights)}
        return scaled, {"portfolio_vol": portfolio_vol, "scale": scale}

    def _clip_weights(self, weights: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        clipped_symbols: list[str] = []
        clipped: dict[str, float] = {}
        for sym in sorted(weights):
            w = weights[sym]
            bounded = max(-self.max_weight, min(self.max_weight, w))
            if not math.isclose(bounded, w):
                clipped_symbols.append(sym)
            clipped[sym] = bounded
        return clipped, clipped_symbols

    def _drop_tiny_positions(self, weights: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        if self.min_tradable_weight <= 0:
            return {sym: weights[sym] for sym in sorted(weights)}, []

        dropped: list[str] = []
        filtered: dict[str, float] = {}
        for sym in sorted(weights):
            if abs(weights[sym]) < self.min_tradable_weight:
                dropped.append(sym)
            else:
                filtered[sym] = weights[sym]
        return filtered, dropped

    def _safe_vol(self, vol: float | None) -> float:
        if vol is None:
            return self.missing_vol
        vol_f = float(vol)
        if not math.isfinite(vol_f) or vol_f <= 0:
            return self.missing_vol
        return vol_f
