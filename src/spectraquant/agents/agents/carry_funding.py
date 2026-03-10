"""Carry / funding-rate trading agent."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant.agents.registry import AgentSignal, BaseAgent

logger = logging.getLogger(__name__)


class CarryFundingAgent(BaseAgent):
    """Generates signals from perpetual-swap funding rates.

    Positive funding (crowded long) → short signal.
    Negative funding (crowded short) → long signal.
    """

    def __init__(self, window: int = 20, horizon: str = "4h") -> None:
        self.window = window
        self.horizon = horizon

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[AgentSignal]:
        if "funding_rate" not in market_data.columns:
            logger.debug("CarryFundingAgent: 'funding_rate' column missing")
            return []

        funding = pd.to_numeric(
            market_data["funding_rate"], errors="coerce",
        ).astype(float)
        valid = funding.dropna()
        if len(valid) < self.window:
            return []

        rolling_mean = funding.rolling(self.window).mean()
        rolling_std = funding.rolling(self.window).std()

        z_score = (funding.iloc[-1] - rolling_mean.iloc[-1]) / (
            rolling_std.iloc[-1] + 1e-9
        )
        if np.isnan(z_score):
            return []

        score = float(np.clip(-z_score, -1.0, 1.0))
        confidence = float(np.clip(abs(z_score) / 3.0, 0.0, 1.0))

        symbol = str(kwargs.get("symbol", market_data["symbol"].iloc[0] if "symbol" in market_data.columns else "UNKNOWN"))
        now = datetime.now(timezone.utc)

        return [
            AgentSignal(
                symbol=symbol,
                score=score,
                confidence=confidence,
                horizon=self.horizon,
                rationale_tags=["carry", "funding_rate", f"z_{z_score:.2f}"],
                asof_utc=now,
            ),
        ]
