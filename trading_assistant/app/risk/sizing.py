"""
Position sizing logic with risk management.
"""
from typing import Dict, Any, Tuple
import numpy as np


class PositionSizer:
    """Calculates position sizes with risk constraints"""
    
    def __init__(self, config: dict):
        """
        Initialize position sizer.
        
        Args:
            config: Risk configuration dictionary
        """
        self.config = config
        self.equity_base = config.get('equity_base', 1000000)
        self.alpha_risk_fraction = config.get('alpha_risk_fraction', 0.02)
        self.b_max = config.get('b_max', 50000)
        self.adv_participation_cap = config.get('adv_participation_cap', 0.1)
    
    def calculate_size(
        self,
        entry_price: float,
        stop_price: float,
        avg_daily_volume: float,
        current_equity: float = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Calculate position size using risk-based sizing.
        
        Args:
            entry_price: Entry price for trade
            stop_price: Stop loss price
            avg_daily_volume: Average daily volume in shares
            current_equity: Current account equity (optional)
            
        Returns:
            Tuple of (shares, metadata dict)
        """
        if current_equity is None:
            current_equity = self.equity_base
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share == 0:
            return 0, {'reason': 'zero_risk'}
        
        # Method 1: Fixed fraction of equity at risk
        max_dollar_risk = current_equity * self.alpha_risk_fraction
        shares_by_risk = int(max_dollar_risk / risk_per_share)
        
        # Method 2: Maximum position size
        shares_by_max_position = int(self.b_max / entry_price)
        
        # Method 3: ADV participation limit
        shares_by_adv = int(avg_daily_volume * self.adv_participation_cap)
        
        # Take minimum of all constraints
        shares = min(shares_by_risk, shares_by_max_position, shares_by_adv)
        
        # Ensure at least 1 share or 0
        if shares < 1:
            shares = 0
        
        metadata = {
            'shares_by_risk': shares_by_risk,
            'shares_by_max_position': shares_by_max_position,
            'shares_by_adv': shares_by_adv,
            'final_shares': shares,
            'dollar_risk': shares * risk_per_share,
            'position_value': shares * entry_price,
            'risk_per_share': risk_per_share
        }
        
        return shares, metadata
    
    def calculate_size_for_target_return(
        self,
        entry_price: float,
        target_price: float,
        target_return_dollars: float
    ) -> int:
        """
        Calculate position size to achieve target dollar return.
        
        Args:
            entry_price: Entry price
            target_price: Target price
            target_return_dollars: Desired dollar profit
            
        Returns:
            Number of shares
        """
        gain_per_share = target_price - entry_price
        
        if gain_per_share <= 0:
            return 0
        
        shares = int(target_return_dollars / gain_per_share)
        return max(0, shares)
