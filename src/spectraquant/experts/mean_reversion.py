"""Mean reversion expert."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from spectraquant.experts.base import BaseExpert, ExpertSignal


class MeanReversionExpert(BaseExpert):
    """Expert that identifies mean reversion opportunities using Bollinger Bands."""
    
    def __init__(self, config: dict):
        super().__init__(config, "mean_reversion")
        self.bb_period = 20
        self.bb_std = 2.0
    
    def get_min_data_rows(self) -> int:
        return self.bb_period + 5
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        news_data: pd.DataFrame | None = None,
    ) -> list[ExpertSignal]:
        """Generate mean reversion signals."""
        if not self.validate_data(prices):
            return []
        
        signals = []
        timestamp = datetime.now(timezone.utc)
        
        for ticker, group in prices.groupby("ticker"):
            df = group.sort_values("date").copy()
            
            if len(df) < self.get_min_data_rows():
                continue
            
            # Compute Bollinger Bands
            df["sma"] = df["close"].rolling(window=self.bb_period).mean()
            df["std"] = df["close"].rolling(window=self.bb_period).std()
            df["bb_upper"] = df["sma"] + (self.bb_std * df["std"])
            df["bb_lower"] = df["sma"] - (self.bb_std * df["std"])
            
            latest = df.iloc[-1]
            if pd.isna(latest["bb_upper"]) or pd.isna(latest["bb_lower"]):
                continue
            
            close = latest["close"]
            bb_upper = latest["bb_upper"]
            bb_lower = latest["bb_lower"]
            sma = latest["sma"]
            
            # Calculate position relative to bands
            bb_range = bb_upper - bb_lower
            if bb_range == 0:
                continue
            
            position = (close - sma) / bb_range
            
            # Determine action
            if position < -0.8:  # Near lower band
                action = "BUY"
                score = min(100, 70 + abs(position) * 30)
                reason = f"Oversold: price near lower BB ({close:.2f} vs {bb_lower:.2f})"
            elif position > 0.8:  # Near upper band
                action = "SELL"
                score = min(100, 70 + position * 30)
                reason = f"Overbought: price near upper BB ({close:.2f} vs {bb_upper:.2f})"
            else:
                action = "HOLD"
                score = 50
                reason = "Price within normal range"
            
            signals.append(ExpertSignal(
                ticker=ticker,
                action=action,
                score=score,
                reason=reason,
                timestamp=timestamp,
            ))
        
        return signals
