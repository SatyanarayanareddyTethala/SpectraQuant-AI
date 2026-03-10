"""On-chain flow trading agent."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant.agents.registry import AgentSignal, BaseAgent

logger = logging.getLogger(__name__)

# Columns used to derive on-chain flow direction signals.
_FLOW_Z_COLS = [
    "exchange_inflow_zscore",
    "exchange_outflow_zscore",
    "active_addresses_zscore",
]


class OnchainFlowAgent(BaseAgent):
    """Generates signals from on-chain flow features.

    Uses anomaly scores and z-score columns when present.
    """

    def __init__(self, horizon: str = "1d") -> None:
        self.horizon = horizon

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[AgentSignal]:
        available_z = [c for c in _FLOW_Z_COLS if c in market_data.columns]
        has_anomaly = "anomaly_score" in market_data.columns

        if not available_z and not has_anomaly:
            logger.debug("OnchainFlowAgent: no on-chain columns found")
            return []

        scores: list[float] = []
        for col in available_z:
            val = pd.to_numeric(
                market_data[col], errors="coerce",
            ).astype(float).iloc[-1]
            if not np.isnan(val):
                # Large exchange inflow → bearish; outflow/active addr → bullish
                direction = -1.0 if "inflow" in col else 1.0
                scores.append(float(np.clip(direction * val / 3.0, -1.0, 1.0)))

        if not scores:
            composite_score = 0.0
        else:
            composite_score = float(np.clip(np.mean(scores), -1.0, 1.0))

        # Anomaly score boosts confidence
        if has_anomaly:
            anomaly = pd.to_numeric(
                market_data["anomaly_score"], errors="coerce",
            ).astype(float).iloc[-1]
            if np.isnan(anomaly):
                anomaly = 0.0
            confidence = float(np.clip(anomaly, 0.0, 1.0))
        else:
            confidence = float(np.clip(abs(composite_score), 0.0, 1.0))

        if composite_score == 0.0 and confidence == 0.0:
            return []

        symbol = str(kwargs.get("symbol", market_data["symbol"].iloc[0] if "symbol" in market_data.columns else "UNKNOWN"))
        now = datetime.now(timezone.utc)

        return [
            AgentSignal(
                symbol=symbol,
                score=composite_score,
                confidence=confidence,
                horizon=self.horizon,
                rationale_tags=["onchain", "flow"] + available_z,
                asof_utc=now,
            ),
        ]
