"""Equity mean reversion agent."""
from __future__ import annotations

import pandas as pd

from spectraquant.core.enums import SignalStatus
from spectraquant.equities.signals._base_agent import AgentOutput, BaseEquityAgent


class MeanReversionAgent(BaseEquityAgent):
    """Mean reversion: buy when price is below rolling mean, sell above."""

    agent_id = "equity_mean_reversion"

    def __init__(self, window: int = 20, z_threshold: float = 1.0) -> None:
        self._window = window
        self._z_threshold = z_threshold

    def _compute(self, df: pd.DataFrame, symbol: str) -> AgentOutput:
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        rolling_mean = close.rolling(self._window).mean()
        rolling_std = close.rolling(self._window).std()
        z = (close - rolling_mean) / rolling_std.replace(0, 1e-9)
        z_score = float(z.iloc[-1])

        # Negative z-score → below mean → positive (buy) signal
        score = max(-1.0, min(1.0, -z_score / (self._z_threshold * 2)))
        confidence = min(0.9, abs(z_score) / 3 + 0.1)

        return AgentOutput(
            canonical_symbol=symbol,
            agent_id=self.agent_id,
            horizon=self.horizon,
            signal_score=score,
            confidence=confidence,
            status=SignalStatus.OK,
            rationale=f"z_score={z_score:.3f}",
            available_inputs=["OHLCV"],
            metadata={"z_score": z_score},
        )
