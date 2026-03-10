"""Volatility-regime trading agent."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant.agents.registry import AgentSignal, BaseAgent

logger = logging.getLogger(__name__)


class VolatilityAgent(BaseAgent):
    """Generates signals based on realised volatility vs. historical regime.

    Short when volatility is expanding; long when contracting.
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 60,
        horizon: str = "1d",
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.horizon = horizon

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[AgentSignal]:
        if "close" not in market_data.columns:
            logger.warning("VolatilityAgent: 'close' column missing")
            return []

        close = pd.to_numeric(market_data["close"], errors="coerce").astype(float)
        if len(close.dropna()) < self.long_window + 1:
            return []

        daily_returns = close.pct_change()
        short_vol = daily_returns.rolling(self.short_window).std()
        long_vol = daily_returns.rolling(self.long_window).std()

        # Percentile rank of recent vol within the long window
        vol_series = daily_returns.rolling(self.long_window).std()
        pct_rank = vol_series.rank(pct=True).iloc[-1]
        if np.isnan(pct_rank):
            return []

        # Vol expanding (high rank) → short; contracting (low rank) → long
        score = float(np.clip(1.0 - 2.0 * pct_rank, -1.0, 1.0))

        latest_short = short_vol.iloc[-1]
        latest_long = long_vol.iloc[-1]
        if np.isnan(latest_short) or np.isnan(latest_long) or latest_long == 0:
            return []

        vol_ratio = latest_short / latest_long
        confidence = float(np.clip(abs(vol_ratio - 1.0), 0.0, 1.0))

        symbol = str(kwargs.get("symbol", market_data["symbol"].iloc[0] if "symbol" in market_data.columns else "UNKNOWN"))
        now = datetime.now(timezone.utc)

        return [
            AgentSignal(
                symbol=symbol,
                score=score,
                confidence=confidence,
                horizon=self.horizon,
                rationale_tags=["volatility", f"vol_pct_{pct_rank:.2f}"],
                asof_utc=now,
            ),
        ]
