"""News / catalyst sentiment trading agent."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant.agents.registry import AgentSignal, BaseAgent

logger = logging.getLogger(__name__)


class NewsCatalystAgent(BaseAgent):
    """Generates signals from pre-computed news sentiment and impact columns."""

    def __init__(self, horizon: str = "1d") -> None:
        self.horizon = horizon

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[AgentSignal]:
        required = {"news_impact_mean", "news_sentiment_mean"}
        if not required.issubset(market_data.columns):
            logger.debug(
                "NewsCatalystAgent: missing columns %s",
                required - set(market_data.columns),
            )
            return []

        impact = pd.to_numeric(
            market_data["news_impact_mean"], errors="coerce",
        ).astype(float)
        sentiment = pd.to_numeric(
            market_data["news_sentiment_mean"], errors="coerce",
        ).astype(float)

        latest_impact = impact.iloc[-1]
        latest_sentiment = sentiment.iloc[-1]
        if np.isnan(latest_impact) or np.isnan(latest_sentiment):
            return []

        raw_score = latest_sentiment * latest_impact
        score = float(np.clip(raw_score, -1.0, 1.0))

        # Confidence from article count when available
        if "news_article_count" in market_data.columns:
            count = pd.to_numeric(
                market_data["news_article_count"], errors="coerce",
            ).astype(float).iloc[-1]
            confidence = float(np.clip(count / 10.0, 0.0, 1.0))
        else:
            confidence = float(np.clip(abs(latest_impact), 0.0, 1.0))

        symbol = str(kwargs.get("symbol", market_data["symbol"].iloc[0] if "symbol" in market_data.columns else "UNKNOWN"))
        now = datetime.now(timezone.utc)

        return [
            AgentSignal(
                symbol=symbol,
                score=score,
                confidence=confidence,
                horizon=self.horizon,
                rationale_tags=["news", "catalyst", f"sentiment_{latest_sentiment:.2f}"],
                asof_utc=now,
            ),
        ]
