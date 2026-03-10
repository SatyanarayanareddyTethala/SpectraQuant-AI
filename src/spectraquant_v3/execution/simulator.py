"""Execution simulator for SpectraQuant-AI-V3.

Simulates order execution with configurable transaction costs, slippage,
spread proxies, and turnover penalties.  Designed for paper-trading
simulations and backtesting cost modelling.

Usage::

    from spectraquant_v3.execution import ExecutionSimulator

    sim = ExecutionSimulator(
        slippage_bps=5,
        transaction_cost_bps=5,
        spread_bps=2,
        turnover_penalty_bps=1,
    )

    results = sim.execute_weights(
        target_weights={"BTC": 0.5, "ETH": 0.3},
        prev_weights={"BTC": 0.4, "ETH": 0.4},
        prices={"BTC": 45000.0, "ETH": 3000.0},
    )
    for r in results:
        print(r.symbol, r.net_return_impact)
"""

from __future__ import annotations

import logging
from typing import Any

from spectraquant_v3.execution.result import ExecutionResult

logger = logging.getLogger(__name__)

# 1 basis point = 0.0001
_BPS = 1e-4


class ExecutionSimulator:
    """Paper-trading execution simulator with cost modelling.

    Args:
        slippage_bps:          Market impact slippage in basis points per trade.
        transaction_cost_bps:  Brokerage / exchange fee in basis points per trade.
        spread_bps:            Half-spread proxy in basis points.
        turnover_penalty_bps:  Additional penalty per unit of weight change.
        max_position_size:     Maximum allowed weight for any single symbol.
        mode:                  ``"paper"`` (default) or ``"live"``
                               (live raises NotImplementedError).
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        transaction_cost_bps: float = 5.0,
        spread_bps: float = 2.0,
        turnover_penalty_bps: float = 1.0,
        max_position_size: float = 1.0,
        mode: str = "paper",
    ) -> None:
        if mode == "live":
            raise NotImplementedError(
                "Live execution is not supported in this release.  "
                "Set mode='paper' for simulation."
            )
        if mode != "paper":
            raise ValueError(f"mode must be 'paper', got {mode!r}")

        self.slippage_bps = slippage_bps
        self.transaction_cost_bps = transaction_cost_bps
        self.spread_bps = spread_bps
        self.turnover_penalty_bps = turnover_penalty_bps
        self.max_position_size = max_position_size
        self.mode = mode

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def execute_weights(
        self,
        target_weights: dict[str, float],
        prev_weights: dict[str, float] | None = None,
        prices: dict[str, float] | None = None,
    ) -> list[ExecutionResult]:
        """Simulate execution of target portfolio weights.

        For each symbol a fill is simulated by applying slippage,
        transaction costs, spread proxies, and a turnover penalty
        proportional to the absolute weight change.

        Args:
            target_weights: Dict mapping symbol → desired portfolio weight.
            prev_weights:   Previous portfolio weights (used for turnover
                            calculation).  Defaults to all-zero.
            prices:         Reference prices per symbol.  Used for fill
                            price simulation.  Defaults to 1.0 per symbol.

        Returns:
            List of :class:`~spectraquant_v3.execution.result.ExecutionResult`
            objects, one per symbol that has a non-zero target weight.

        Raises:
            ValueError: When *target_weights* is empty.
        """
        if not target_weights:
            raise ValueError("execute_weights: target_weights must not be empty")

        prev = prev_weights or {}
        px = prices or {}

        results: list[ExecutionResult] = []

        for symbol, target_w in target_weights.items():
            # Cap weight
            executed_w = min(abs(target_w), self.max_position_size)
            if target_w < 0:
                executed_w = -executed_w

            prev_w = prev.get(symbol, 0.0)
            turnover = abs(executed_w - prev_w)
            side = "buy" if executed_w >= prev_w else "sell"

            ref_price = px.get(symbol, 1.0)
            slippage_factor = self.slippage_bps * _BPS
            if side == "buy":
                fill_price = ref_price * (1.0 + slippage_factor + self.spread_bps * _BPS)
            else:
                fill_price = ref_price * (1.0 - slippage_factor - self.spread_bps * _BPS)

            tc = self.transaction_cost_bps * _BPS
            tp = turnover * self.turnover_penalty_bps * _BPS
            net_impact = -(tc + slippage_factor + self.spread_bps * _BPS + tp)

            results.append(
                ExecutionResult(
                    symbol=symbol,
                    side=side,
                    target_weight=target_w,
                    executed_weight=executed_w,
                    fill_price=fill_price,
                    transaction_cost=self.transaction_cost_bps,  # stored in bps (same units as slippage_bps)
                    slippage_bps=self.slippage_bps,
                    spread_bps=self.spread_bps,
                    turnover_bps=turnover * self.turnover_penalty_bps,
                    net_return_impact=net_impact,
                    metadata={"turnover": turnover, "ref_price": ref_price},
                )
            )

        logger.debug(
            "ExecutionSimulator: processed %d symbols", len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Cost aggregation
    # ------------------------------------------------------------------

    def total_cost_bps(
        self,
        results: list[ExecutionResult],
    ) -> float:
        """Return the total execution cost of a list of results in basis points.

        Args:
            results: List of :class:`ExecutionResult` objects.

        Returns:
            Total cost in basis points (positive = cost).
        """
        return sum(abs(r.net_return_impact) * 1e4 for r in results)

    def summary(self, results: list[ExecutionResult]) -> dict[str, Any]:
        """Return an aggregate cost summary.

        Args:
            results: List of :class:`ExecutionResult` objects.

        Returns:
            Dict with keys: ``symbols``, ``total_cost_bps``,
            ``avg_slippage_bps``, ``total_turnover``.
        """
        if not results:
            return {
                "symbols": 0,
                "total_cost_bps": 0.0,
                "avg_slippage_bps": 0.0,
                "total_turnover": 0.0,
            }
        total_turnover = sum(r.metadata.get("turnover", 0.0) for r in results)
        return {
            "symbols": len(results),
            "total_cost_bps": self.total_cost_bps(results),
            "avg_slippage_bps": self.slippage_bps,
            "total_turnover": total_turnover,
        }
