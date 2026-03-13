"""Base expert interface for signal generation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class ExpertSignal:
    """Signal from a single expert."""
    ticker: str
    action: str  # "BUY", "HOLD", "SELL"
    score: float  # 0-100 confidence score
    reason: str  # Brief explanation
    timestamp: datetime


class BaseExpert(ABC):
    """Base class for all trading experts."""
    
    def __init__(self, config: dict, name: str):
        """Initialize expert.
        
        Args:
            config: Full configuration dict
            name: Expert name identifier
        """
        self.config = config
        self.name = name
        self.experts_cfg = config.get("experts", {})
    
    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        news_data: pd.DataFrame | None = None,
    ) -> list[ExpertSignal]:
        """Generate trading signals for given data.
        
        Args:
            prices: DataFrame with OHLCV data (ticker, date, open, high, low, close, volume)
            features: Optional precomputed features
            news_data: Optional news/sentiment data
            
        Returns:
            List of ExpertSignal objects
        """
        pass
    
    def get_min_data_rows(self) -> int:
        """Minimum number of data rows required for this expert."""
        return 30  # Default: 30 trading days
    
    def to_dataframe(self, signals: list[ExpertSignal]) -> pd.DataFrame:
        """Convert signals to DataFrame format.
        
        Args:
            signals: List of ExpertSignal objects
            
        Returns:
            DataFrame with columns: ticker, action, score, reason, expert, timestamp
        """
        if not signals:
            return pd.DataFrame(columns=["ticker", "action", "score", "reason", "expert", "timestamp"])
        
        rows = []
        for sig in signals:
            rows.append({
                "ticker": sig.ticker,
                "action": sig.action,
                "score": sig.score,
                "reason": sig.reason,
                "expert": self.name,
                "timestamp": sig.timestamp,
            })
        
        return pd.DataFrame(rows)
    
    def validate_data(self, prices: pd.DataFrame) -> bool:
        """Validate that input data meets minimum requirements.
        
        Args:
            prices: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if prices is None or prices.empty:
            return False
        
        required_cols = ["ticker", "date", "close"]
        for col in required_cols:
            if col not in prices.columns:
                return False
        
        return True
