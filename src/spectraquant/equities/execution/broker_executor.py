"""Equity broker executor placeholder.

In production this would connect to a live broker API.
Currently a stub that raises NotImplementedError to prevent
accidental live trading.
"""
from __future__ import annotations


class EquityBrokerExecutor:
    """Abstract broker executor for equity live trading.

    NOT IMPLEMENTED. Live trading is disabled by default.
    """

    def execute(self, target_weights: dict, **kwargs):
        raise NotImplementedError(
            "Live equity execution is not enabled. "
            "Use EquityPaperExecutor for paper trading."
        )
