"""Equity momentum agent.

Generates a momentum signal based on N-day return and RSI.
"""
from __future__ import annotations

import pandas as pd

from spectraquant.core.enums import SignalStatus
from spectraquant.equities.signals._base_agent import AgentOutput, BaseEquityAgent


class MomentumAgent(BaseEquityAgent):
    """Momentum signal: positive when price trend is upward."""

    agent_id = "equity_momentum"

    def __init__(self, lookback: int = 20, rsi_period: int = 14) -> None:
        self._lookback = lookback
        self._rsi_period = rsi_period

    def _compute(self, df: pd.DataFrame, symbol: str) -> AgentOutput:
        close = df["close"] if "close" in df.columns else df[df.columns[-1]]
        close = pd.to_numeric(close, errors="coerce").dropna()

        # N-day return
        ret = float(close.iloc[-1] / close.iloc[-min(self._lookback, len(close))] - 1)

        # Simplified RSI
        delta = close.diff().dropna()
        gain = delta.clip(lower=0).rolling(self._rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self._rsi_period).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

        # Signal: blend momentum and RSI
        momentum_score = max(-1.0, min(1.0, ret * 5))
        rsi_score = (rsi - 50) / 50.0  # -1 to 1
        score = 0.6 * momentum_score + 0.4 * rsi_score
        score = max(-1.0, min(1.0, score))
        confidence = min(0.9, abs(score) + 0.1)

        return AgentOutput(
            canonical_symbol=symbol,
            agent_id=self.agent_id,
            horizon=self.horizon,
            signal_score=score,
            confidence=confidence,
            status=SignalStatus.OK,
            rationale=f"momentum={ret:.3f} rsi={rsi:.1f}",
            available_inputs=["OHLCV"],
            metadata={"n_day_return": ret, "rsi": rsi},
        )
