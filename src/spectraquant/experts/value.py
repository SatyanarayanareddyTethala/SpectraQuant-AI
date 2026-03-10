"""Value/fundamental expert."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from spectraquant.experts.base import BaseExpert, ExpertSignal


class ValueExpert(BaseExpert):
    """Expert that identifies value opportunities using price ratios."""
    
    def __init__(self, config: dict):
        super().__init__(config, "value")
        self.lookback_period = 252  # 1 year
        self.percentile_low = 0.2  # 20th percentile = cheap
        self.percentile_high = 0.8  # 80th percentile = expensive
    
    def get_min_data_rows(self) -> int:
        return self.lookback_period + 5
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        news_data: pd.DataFrame | None = None,
    ) -> list[ExpertSignal]:
        """Generate value-based signals."""
        if not self.validate_data(prices):
            return []
        
        signals = []
        timestamp = datetime.now(timezone.utc)
        
        for ticker, group in prices.groupby("ticker"):
            df = group.sort_values("date").copy()
            
            if len(df) < self.get_min_data_rows():
                continue
            
            # Simple value metric: current price vs historical range
            recent_prices = df.tail(self.lookback_period)["close"]
            current_price = df.iloc[-1]["close"]
            
            percentile = (recent_prices <= current_price).sum() / len(recent_prices)
            
            # Determine action based on historical percentile
            if percentile < self.percentile_low:
                action = "BUY"
                score = 70 + (self.percentile_low - percentile) * 100
                reason = f"Undervalued: {percentile:.0%} percentile vs 1Y range"
            elif percentile > self.percentile_high:
                action = "SELL"
                score = 70 + (percentile - self.percentile_high) * 100
                reason = f"Overvalued: {percentile:.0%} percentile vs 1Y range"
            else:
                action = "HOLD"
                score = 50
                reason = f"Fair value: {percentile:.0%} percentile"
            
            signals.append(ExpertSignal(
                ticker=ticker,
                action=action,
                score=min(100, score),
                reason=reason,
                timestamp=timestamp,
            ))
        
        return signals
