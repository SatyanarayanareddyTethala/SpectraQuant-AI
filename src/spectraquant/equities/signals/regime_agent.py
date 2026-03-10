"""Equity regime agent (placeholder)."""
from __future__ import annotations

import pandas as pd

from spectraquant.core.enums import SignalStatus
from spectraquant.equities.signals._base_agent import AgentOutput, BaseEquityAgent


class RegimeAgent(BaseEquityAgent):
    """Simple regime detector based on 200-day moving average."""

    agent_id = "equity_regime"

    def __init__(self, long_window: int = 200, short_window: int = 50) -> None:
        self._long_window = long_window
        self._short_window = short_window

    def _compute(self, df: pd.DataFrame, symbol: str) -> AgentOutput:
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        ma_long = close.rolling(min(self._long_window, len(close))).mean().iloc[-1]
        ma_short = close.rolling(min(self._short_window, len(close))).mean().iloc[-1]
        current = float(close.iloc[-1])

        if current > ma_long:
            regime = "BULL"
            score = 0.3
        else:
            regime = "BEAR"
            score = -0.3

        if ma_short > ma_long:
            score += 0.1

        score = max(-1.0, min(1.0, score))

        return AgentOutput(
            canonical_symbol=symbol,
            agent_id=self.agent_id,
            signal_score=score,
            confidence=0.6,
            status=SignalStatus.OK,
            rationale=f"regime={regime}",
            available_inputs=["OHLCV"],
            metadata={"regime": regime},
        )
