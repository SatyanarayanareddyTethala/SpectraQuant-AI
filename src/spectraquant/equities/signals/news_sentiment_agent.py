"""Equity news sentiment agent (stub).

Returns NO_SIGNAL with NO_NEWS_DATA reason when no news data is available.
When news data is supplied, blends sentiment scores into a signal.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from spectraquant.core.enums import NoSignalReason, SignalStatus
from spectraquant.equities.signals._base_agent import AgentOutput, BaseEquityAgent


class NewsSentimentAgent(BaseEquityAgent):
    """News sentiment: returns NO_SIGNAL gracefully when no news is available."""

    agent_id = "equity_news_sentiment"

    def __init__(self, news_data: dict[str, Any] | None = None) -> None:
        self._news_data = news_data or {}

    def _compute(self, df: pd.DataFrame, symbol: str) -> AgentOutput:
        if symbol not in self._news_data:
            return AgentOutput(
                canonical_symbol=symbol,
                agent_id=self.agent_id,
                status=SignalStatus.NO_SIGNAL,
                error_reason=str(NoSignalReason.NO_NEWS_DATA),
                rationale=f"No news data available for {symbol!r}",
                required_inputs=["OHLCV", "news"],
                available_inputs=["OHLCV"],
            )

        news = self._news_data[symbol]
        score = float(news.get("sentiment_score", 0.0))
        score = max(-1.0, min(1.0, score))
        confidence = float(news.get("confidence", 0.5))

        return AgentOutput(
            canonical_symbol=symbol,
            agent_id=self.agent_id,
            signal_score=score,
            confidence=confidence,
            status=SignalStatus.OK,
            rationale=f"news_sentiment={score:.3f}",
            required_inputs=["OHLCV", "news"],
            available_inputs=["OHLCV", "news"],
            metadata={"news_score": score},
        )
