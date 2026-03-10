"""Equity breakout agent."""
from __future__ import annotations

import pandas as pd

from spectraquant.core.enums import SignalStatus
from spectraquant.equities.signals._base_agent import AgentOutput, BaseEquityAgent


class BreakoutAgent(BaseEquityAgent):
    """Breakout signal: fires when price exceeds N-period high."""

    agent_id = "equity_breakout"

    def __init__(self, window: int = 52) -> None:
        self._window = window

    def _compute(self, df: pd.DataFrame, symbol: str) -> AgentOutput:
        high = pd.to_numeric(df["high"], errors="coerce").dropna()
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        period = min(self._window, len(high))
        rolling_high = float(high.rolling(period).max().iloc[-1])
        current = float(close.iloc[-1])

        if rolling_high <= 0:
            return AgentOutput(
                canonical_symbol=symbol,
                agent_id=self.agent_id,
                status=SignalStatus.NO_SIGNAL,
                error_reason="invalid price data",
            )

        proximity = (current / rolling_high) - 0.9  # positive near breakout
        score = max(-0.5, min(1.0, proximity * 5))

        return AgentOutput(
            canonical_symbol=symbol,
            agent_id=self.agent_id,
            signal_score=score,
            confidence=min(0.85, abs(score) + 0.1),
            status=SignalStatus.OK,
            rationale=f"price={current:.2f} {period}d_high={rolling_high:.2f}",
            available_inputs=["OHLCV"],
            metadata={"rolling_high": rolling_high, "period": period},
        )
