"""Base class for equity signal agents.

Every equity signal agent must:
1. Accept a pd.DataFrame of OHLCV data with UTC DatetimeIndex.
2. Return an AgentOutput with explicit status and reason on missing data.
3. Never silently return an empty signal when data is available.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from spectraquant.core.enums import AssetClass, NoSignalReason, SignalStatus

logger = logging.getLogger(__name__)

_REQUIRED_OHLCV_COLS = {"open", "high", "low", "close", "volume"}
_MIN_ROWS = 20  # minimum rows needed for most calculations


@dataclass
class AgentOutput:
    """Standardised output schema for equity signal agents."""

    run_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    canonical_symbol: str = ""
    asset_class: str = str(AssetClass.EQUITY)
    agent_id: str = ""
    horizon: str = "20d"
    signal_score: float = 0.0     # -1.0 to 1.0
    confidence: float = 0.0       # 0.0 to 1.0
    rationale: str = ""
    required_inputs: list[str] = field(default_factory=lambda: ["OHLCV"])
    available_inputs: list[str] = field(default_factory=list)
    status: SignalStatus = SignalStatus.NO_SIGNAL
    error_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseEquityAgent:
    """Abstract base for equity signal agents.

    Subclasses must implement ``_compute(df, symbol)`` which returns an
    AgentOutput.  The base class handles data validation and graceful
    degradation.
    """

    agent_id: str = "base_equity_agent"
    horizon: str = "20d"

    def __call__(self, df: pd.DataFrame, symbol: str = "") -> AgentOutput:
        """Run the agent on *df* for *symbol*."""
        return self.run(df, symbol)

    def run(self, df: pd.DataFrame, symbol: str = "") -> AgentOutput:
        """Validate data and run the agent.

        Returns NO_SIGNAL with NO_PRICE_DATA if OHLCV is missing/empty.
        """
        if df is None or df.empty:
            return AgentOutput(
                canonical_symbol=symbol,
                agent_id=self.agent_id,
                horizon=self.horizon,
                status=SignalStatus.NO_SIGNAL,
                error_reason=str(NoSignalReason.NO_PRICE_DATA),
                rationale=f"{self.agent_id}: no OHLCV data",
            )

        cols = {str(c).lower() for c in df.columns}
        missing_cols = _REQUIRED_OHLCV_COLS - cols
        if missing_cols:
            return AgentOutput(
                canonical_symbol=symbol,
                agent_id=self.agent_id,
                horizon=self.horizon,
                status=SignalStatus.NO_SIGNAL,
                error_reason=str(NoSignalReason.NO_PRICE_DATA),
                rationale=f"{self.agent_id}: missing columns {missing_cols}",
            )

        if len(df) < _MIN_ROWS:
            return AgentOutput(
                canonical_symbol=symbol,
                agent_id=self.agent_id,
                horizon=self.horizon,
                status=SignalStatus.NO_SIGNAL,
                error_reason=str(NoSignalReason.INSUFFICIENT_HISTORY),
                rationale=f"{self.agent_id}: only {len(df)} rows (min {_MIN_ROWS})",
            )

        try:
            return self._compute(df.copy(), symbol)
        except Exception as exc:
            logger.error("%s failed for %r: %s", self.agent_id, symbol, exc)
            return AgentOutput(
                canonical_symbol=symbol,
                agent_id=self.agent_id,
                horizon=self.horizon,
                status=SignalStatus.ERROR,
                error_reason=str(exc),
                rationale=f"{self.agent_id}: computation error",
            )

    def _compute(self, df: pd.DataFrame, symbol: str) -> AgentOutput:
        raise NotImplementedError
