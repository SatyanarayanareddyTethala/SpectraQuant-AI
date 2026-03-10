"""Equity volatility agent."""
from __future__ import annotations

import math

import pandas as pd

from spectraquant.core.enums import SignalStatus
from spectraquant.equities.signals._base_agent import AgentOutput, BaseEquityAgent


class VolatilityAgent(BaseEquityAgent):
    """Volatility regime agent: negative signal in high-vol environments."""

    agent_id = "equity_volatility"

    def __init__(self, window: int = 20, high_vol_threshold: float = 0.03) -> None:
        self._window = window
        self._high_vol_threshold = high_vol_threshold

    def _compute(self, df: pd.DataFrame, symbol: str) -> AgentOutput:
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        returns = close.pct_change().dropna()
        vol = float(returns.rolling(self._window).std().iloc[-1])
        ann_vol = vol * math.sqrt(252)

        # High vol → risk-off → negative score
        score = max(-1.0, min(0.0, -(ann_vol - 0.15) / 0.3))
        if ann_vol <= 0.15:
            score = 0.0  # neutral in normal vol

        return AgentOutput(
            canonical_symbol=symbol,
            agent_id=self.agent_id,
            horizon=self.horizon,
            signal_score=score,
            confidence=0.7,
            status=SignalStatus.OK,
            rationale=f"ann_vol={ann_vol:.3f}",
            available_inputs=["OHLCV"],
            metadata={"ann_vol": ann_vol},
        )
