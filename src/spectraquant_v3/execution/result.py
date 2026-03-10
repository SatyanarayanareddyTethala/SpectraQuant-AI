"""Execution result for SpectraQuant-AI-V3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """Result of an order execution simulation.

    Attributes:
        symbol:              Canonical symbol traded.
        side:                ``"buy"`` or ``"sell"``.
        target_weight:       Desired portfolio weight.
        executed_weight:     Actual weight after execution constraints.
        fill_price:          Simulated fill price (including slippage).
        transaction_cost:    Total transaction cost in basis points.
        slippage_bps:        Slippage applied in basis points.
        spread_bps:          Half-spread proxy in basis points.
        turnover_bps:        Turnover penalty in basis points.
        net_return_impact:   Net return impact of costs (negative = cost).
        metadata:            Free-form dict for additional fields.
    """

    symbol: str
    side: str
    target_weight: float
    executed_weight: float
    fill_price: float
    transaction_cost: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    turnover_bps: float = 0.0
    net_return_impact: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "target_weight": self.target_weight,
            "executed_weight": self.executed_weight,
            "fill_price": self.fill_price,
            "transaction_cost": self.transaction_cost,
            "slippage_bps": self.slippage_bps,
            "spread_bps": self.spread_bps,
            "turnover_bps": self.turnover_bps,
            "net_return_impact": self.net_return_impact,
            "metadata": self.metadata,
        }
