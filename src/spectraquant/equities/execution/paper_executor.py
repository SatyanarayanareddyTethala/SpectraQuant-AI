"""Equity paper executor.

Simulates order execution for the equity pipeline (paper trading only).
No live trading is enabled by default.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EquityOrder:
    """A simulated equity order."""

    symbol: str
    action: str                   # "BUY" | "SELL" | "HOLD"
    target_weight: float
    slippage_bps: float = 5.0
    transaction_cost_bps: float = 10.0
    executed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)


class EquityPaperExecutor:
    """Paper-trade equity orders (no live execution).

    Args:
        slippage_bps: Assumed slippage in basis points.
        transaction_cost_bps: Assumed round-trip transaction cost in bps.
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        transaction_cost_bps: float = 10.0,
    ) -> None:
        self._slippage_bps = slippage_bps
        self._transaction_cost_bps = transaction_cost_bps
        self._log: list[EquityOrder] = []

    def execute(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float] | None = None,
    ) -> list[EquityOrder]:
        """Generate paper orders from target weights.

        Args:
            target_weights: Desired portfolio weights {symbol → weight}.
            current_weights: Current holdings (default: all zero).

        Returns:
            List of EquityOrder records.
        """
        current_weights = current_weights or {}
        orders: list[EquityOrder] = []

        all_symbols = set(target_weights) | set(current_weights)
        for sym in sorted(all_symbols):
            target = target_weights.get(sym, 0.0)
            current = current_weights.get(sym, 0.0)
            delta = target - current

            if abs(delta) < 1e-4:
                continue

            action = "BUY" if delta > 0 else "SELL"
            order = EquityOrder(
                symbol=sym,
                action=action,
                target_weight=target,
                slippage_bps=self._slippage_bps,
                transaction_cost_bps=self._transaction_cost_bps,
                metadata={"delta_weight": delta},
            )
            orders.append(order)
            self._log.append(order)
            logger.info(
                "PAPER %s %s target_weight=%.4f delta=%.4f",
                action, sym, target, delta,
            )

        return orders

    @property
    def order_log(self) -> list[EquityOrder]:
        """All paper orders placed in this session."""
        return list(self._log)
