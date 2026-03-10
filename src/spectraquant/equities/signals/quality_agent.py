"""Equity quality filter agent.

Rejects symbols with poor price history quality.
"""
from __future__ import annotations

import pandas as pd

from spectraquant.core.enums import NoSignalReason, SignalStatus
from spectraquant.equities.signals._base_agent import AgentOutput, BaseEquityAgent


class QualityAgent(BaseEquityAgent):
    """Quality filter: neutral signal for high-quality data, NO_SIGNAL for poor."""

    agent_id = "equity_quality"

    def __init__(
        self,
        min_rows: int = 60,
        max_zero_return_fraction: float = 0.1,
    ) -> None:
        self._min_rows = min_rows
        self._max_zero_fraction = max_zero_return_fraction

    def _compute(self, df: pd.DataFrame, symbol: str) -> AgentOutput:
        close = pd.to_numeric(df["close"], errors="coerce").dropna()

        if len(close) < self._min_rows:
            return AgentOutput(
                canonical_symbol=symbol,
                agent_id=self.agent_id,
                status=SignalStatus.NO_SIGNAL,
                error_reason=str(NoSignalReason.INSUFFICIENT_HISTORY),
                rationale=f"Only {len(close)} rows (need {self._min_rows})",
            )

        returns = close.pct_change().dropna()
        zero_frac = float((returns == 0).sum() / len(returns))
        if zero_frac > self._max_zero_fraction:
            return AgentOutput(
                canonical_symbol=symbol,
                agent_id=self.agent_id,
                status=SignalStatus.NO_SIGNAL,
                error_reason=str(NoSignalReason.FILTER_REJECTED),
                rationale=f"Too many zero returns: {zero_frac:.1%}",
            )

        return AgentOutput(
            canonical_symbol=symbol,
            agent_id=self.agent_id,
            signal_score=0.0,
            confidence=0.9,
            status=SignalStatus.OK,
            rationale="Quality gate passed",
            available_inputs=["OHLCV"],
        )
