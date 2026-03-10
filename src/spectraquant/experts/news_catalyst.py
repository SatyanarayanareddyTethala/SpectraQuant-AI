"""News catalyst expert."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from spectraquant.experts.base import BaseExpert, ExpertSignal


class NewsCatalystExpert(BaseExpert):
    """Expert that generates signals based on news sentiment and catalysts."""
    
    def __init__(self, config: dict):
        super().__init__(config, "news_catalyst")
        self.sentiment_threshold = 0.3
        self.min_mentions = 2
    
    def get_min_data_rows(self) -> int:
        return 5  # Minimal price history needed
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        news_data: pd.DataFrame | None = None,
    ) -> list[ExpertSignal]:
        """Generate news catalyst signals."""
        if not self.validate_data(prices):
            return []
        
        if news_data is None or news_data.empty:
            # No news data available
            return []
        
        signals = []
        timestamp = datetime.now(timezone.utc)
        
        # Ensure news_data has required columns
        required_cols = ["ticker", "score", "mentions"]
        if not all(col in news_data.columns for col in required_cols):
            return []
        
        for ticker in news_data["ticker"].unique():
            ticker_news = news_data[news_data["ticker"] == ticker].iloc[0]
            
            score_val = ticker_news["score"]
            mentions = ticker_news["mentions"]
            
            if mentions < self.min_mentions:
                continue
            
            # Normalize score to sentiment-like scale
            # Assuming score is positive impact score
            if score_val > self.sentiment_threshold:
                action = "BUY"
                score = min(100, 60 + score_val * 20)
                reason = f"Positive news catalyst: {mentions} mentions"
            elif score_val < -self.sentiment_threshold:
                action = "SELL"
                score = min(100, 60 + abs(score_val) * 20)
                reason = f"Negative news catalyst: {mentions} mentions"
            else:
                action = "HOLD"
                score = 50
                reason = f"Neutral news: {mentions} mentions"
            
            signals.append(ExpertSignal(
                ticker=ticker,
                action=action,
                score=score,
                reason=reason,
                timestamp=timestamp,
            ))
        
        return signals
