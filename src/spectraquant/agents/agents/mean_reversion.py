"""Mean-reversion trading agent."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant.agents.registry import AgentSignal, BaseAgent

logger = logging.getLogger(__name__)


class MeanReversionAgent(BaseAgent):
    """Generates signals from z-score deviation of price vs. rolling mean."""

    def __init__(self, window: int = 20, horizon: str = "1d") -> None:
        self.window = window
        self.horizon = horizon

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[AgentSignal]:
        if "close" not in market_data.columns:
            logger.warning("MeanReversionAgent: 'close' column missing")
            return []

        close = pd.to_numeric(market_data["close"], errors="coerce").astype(float)
        if len(close.dropna()) < self.window + 1:
            return []

        rolling_mean = close.rolling(self.window).mean()
        rolling_std = close.rolling(self.window).std()

        z_score = (close - rolling_mean) / (rolling_std + 1e-9)
        latest_z = z_score.iloc[-1]
        if np.isnan(latest_z):
            return []

        # Negative z → buy signal (price below mean); positive z → sell
        score = float(np.clip(-latest_z, -1.0, 1.0))
        confidence = float(1.0 - np.exp(-abs(latest_z) / 2.0))

        symbol = str(kwargs.get("symbol", market_data["symbol"].iloc[0] if "symbol" in market_data.columns else "UNKNOWN"))
        now = datetime.now(timezone.utc)

        return [
            AgentSignal(
                symbol=symbol,
                score=score,
                confidence=confidence,
                horizon=self.horizon,
                rationale_tags=["mean_reversion", f"z_score_{latest_z:.2f}"],
                asof_utc=now,
            ),
        ]
