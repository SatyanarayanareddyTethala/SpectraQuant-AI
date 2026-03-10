"""Momentum-based trading agent."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant.agents.registry import AgentSignal, BaseAgent

logger = logging.getLogger(__name__)


class MomentumAgent(BaseAgent):
    """Generates signals from price momentum over a configurable lookback."""

    def __init__(self, lookback: int = 20, horizon: str = "1d") -> None:
        self.lookback = lookback
        self.horizon = horizon

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[AgentSignal]:
        if "close" not in market_data.columns:
            logger.warning("MomentumAgent: 'close' column missing")
            return []

        close = pd.to_numeric(market_data["close"], errors="coerce").astype(float)
        if len(close.dropna()) < self.lookback + 1:
            return []

        returns = close.pct_change(self.lookback)
        latest_return = returns.iloc[-1]
        if np.isnan(latest_return):
            return []

        # Normalise return to [-1, 1] using tanh
        score = float(np.clip(np.tanh(latest_return * 10), -1.0, 1.0))

        # Confidence: consistency of sign over the lookback window
        recent = close.pct_change().iloc[-self.lookback :]
        sign_consistency = float(np.abs(recent.mean()) / (recent.std() + 1e-9))
        confidence = float(np.clip(sign_consistency, 0.0, 1.0))

        symbol = str(kwargs.get("symbol", market_data["symbol"].iloc[0] if "symbol" in market_data.columns else "UNKNOWN"))
        now = datetime.now(timezone.utc)

        return [
            AgentSignal(
                symbol=symbol,
                score=score,
                confidence=confidence,
                horizon=self.horizon,
                rationale_tags=["momentum", f"lookback_{self.lookback}"],
                asof_utc=now,
            ),
        ]
