"""Volatility/defensive expert."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from spectraquant.experts.base import BaseExpert, ExpertSignal


class VolatilityExpert(BaseExpert):
    """Expert that favors low-volatility, defensive positions."""
    
    def __init__(self, config: dict):
        super().__init__(config, "volatility")
        self.vol_period = 20
        self.vol_threshold_low = 0.01  # 1% daily vol
        self.vol_threshold_high = 0.04  # 4% daily vol
    
    def get_min_data_rows(self) -> int:
        return self.vol_period + 5
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        news_data: pd.DataFrame | None = None,
    ) -> list[ExpertSignal]:
        """Generate volatility-based signals."""
        if not self.validate_data(prices):
            return []
        
        signals = []
        timestamp = datetime.now(timezone.utc)
        
        for ticker, group in prices.groupby("ticker"):
            df = group.sort_values("date").copy()
            
            if len(df) < self.get_min_data_rows():
                continue
            
            # Compute returns and volatility
            df["returns"] = df["close"].pct_change()
            df["volatility"] = df["returns"].rolling(window=self.vol_period).std()
            
            latest = df.iloc[-1]
            if pd.isna(latest["volatility"]):
                continue
            
            vol = latest["volatility"]
            
            # Determine action - favor low volatility in defensive mode
            if vol < self.vol_threshold_low:
                action = "BUY"
                score = 75
                reason = f"Low volatility ({vol:.2%}), defensive position"
            elif vol > self.vol_threshold_high:
                action = "SELL"
                score = 70
                reason = f"High volatility ({vol:.2%}), avoid risk"
            else:
                action = "HOLD"
                score = 50
                reason = f"Moderate volatility ({vol:.2%})"
            
            signals.append(ExpertSignal(
                ticker=ticker,
                action=action,
                score=score,
                reason=reason,
                timestamp=timestamp,
            ))
        
        return signals
