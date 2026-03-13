"""Trend-following expert."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from spectraquant.experts.base import BaseExpert, ExpertSignal


class TrendExpert(BaseExpert):
    """Expert that follows price trends using moving averages."""
    
    def __init__(self, config: dict):
        super().__init__(config, "trend")
        self.fast_period = 20
        self.slow_period = 50
        self.signal_threshold = 0.02  # 2% spread for strong signal
    
    def get_min_data_rows(self) -> int:
        return self.slow_period + 5
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        news_data: pd.DataFrame | None = None,
    ) -> list[ExpertSignal]:
        """Generate trend-following signals."""
        if not self.validate_data(prices):
            return []
        
        signals = []
        timestamp = datetime.now(timezone.utc)
        
        # Group by ticker
        for ticker, group in prices.groupby("ticker"):
            # Sort by date
            df = group.sort_values("date").copy()
            
            if len(df) < self.get_min_data_rows():
                continue
            
            # Compute moving averages
            df["sma_fast"] = df["close"].rolling(window=self.fast_period).mean()
            df["sma_slow"] = df["close"].rolling(window=self.slow_period).mean()
            
            # Get latest values
            latest = df.iloc[-1]
            if pd.isna(latest["sma_fast"]) or pd.isna(latest["sma_slow"]):
                continue
            
            sma_fast = latest["sma_fast"]
            sma_slow = latest["sma_slow"]
            
            # Calculate signal strength
            spread = (sma_fast - sma_slow) / sma_slow
            
            # Determine action
            if spread > self.signal_threshold:
                action = "BUY"
                score = min(100, 50 + spread * 1000)  # Scale to 0-100
                reason = f"Fast SMA ({sma_fast:.2f}) > Slow SMA ({sma_slow:.2f}), uptrend"
            elif spread < -self.signal_threshold:
                action = "SELL"
                score = min(100, 50 + abs(spread) * 1000)
                reason = f"Fast SMA ({sma_fast:.2f}) < Slow SMA ({sma_slow:.2f}), downtrend"
            else:
                action = "HOLD"
                score = 50
                reason = "Neutral trend"
            
            signals.append(ExpertSignal(
                ticker=ticker,
                action=action,
                score=score,
                reason=reason,
                timestamp=timestamp,
            ))
        
        return signals
