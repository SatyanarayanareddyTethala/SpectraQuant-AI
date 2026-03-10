"""Momentum expert."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from spectraquant.experts.base import BaseExpert, ExpertSignal


class MomentumExpert(BaseExpert):
    """Expert that identifies momentum using RSI and rate of change."""
    
    def __init__(self, config: dict):
        super().__init__(config, "momentum")
        self.rsi_period = 14
        self.roc_period = 10
        self.rsi_buy_threshold = 60
        self.rsi_sell_threshold = 40
    
    def get_min_data_rows(self) -> int:
        return max(self.rsi_period, self.roc_period) + 5
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        news_data: pd.DataFrame | None = None,
    ) -> list[ExpertSignal]:
        """Generate momentum signals."""
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
            
            # Compute RSI
            df["rsi"] = self._compute_rsi(df["close"], self.rsi_period)
            
            # Compute rate of change
            df["roc"] = df["close"].pct_change(periods=self.roc_period) * 100
            
            # Get latest values
            latest = df.iloc[-1]
            if pd.isna(latest["rsi"]) or pd.isna(latest["roc"]):
                continue
            
            rsi = latest["rsi"]
            roc = latest["roc"]
            
            # Determine action based on RSI and ROC
            if rsi > self.rsi_buy_threshold and roc > 0:
                action = "BUY"
                score = min(100, 50 + (rsi - 50) + roc)
                reason = f"Strong momentum: RSI={rsi:.1f}, ROC={roc:.1f}%"
            elif rsi < self.rsi_sell_threshold and roc < 0:
                action = "SELL"
                score = min(100, 50 + (50 - rsi) + abs(roc))
                reason = f"Weak momentum: RSI={rsi:.1f}, ROC={roc:.1f}%"
            else:
                action = "HOLD"
                score = 50
                reason = f"Neutral momentum: RSI={rsi:.1f}"
            
            signals.append(ExpertSignal(
                ticker=ticker,
                action=action,
                score=score,
                reason=reason,
                timestamp=timestamp,
            ))
        
        return signals
