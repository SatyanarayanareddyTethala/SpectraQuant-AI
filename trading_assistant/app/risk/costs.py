"""
Transaction cost and slippage models.
"""
from typing import Dict, Any
import numpy as np


class CostModel:
    """Models transaction costs and slippage"""
    
    def __init__(self, config: dict):
        """
        Initialize cost model.
        
        Args:
            config: Costs configuration dictionary
        """
        self.config = config
        self.commission_per_trade = config.get('commission_per_trade', 20.0)
        self.slippage_config = config.get('slippage_model', {})
        self.base_bps = self.slippage_config.get('base_bps', 5)
        self.spread_weight = self.slippage_config.get('spread_weight', 0.5)
        self.vol_weight = self.slippage_config.get('vol_weight', 0.3)
        self.participation_weight = self.slippage_config.get('participation_weight', 0.2)
    
    def calculate_total_cost(
        self,
        qty: int,
        price: float,
        spread_bps: float = 10.0,
        volatility: float = 0.02,
        adv_participation: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate total transaction costs including slippage.
        
        Args:
            qty: Number of shares
            price: Execution price
            spread_bps: Bid-ask spread in basis points
            volatility: Recent volatility (daily)
            adv_participation: Fraction of ADV being traded
            
        Returns:
            Dictionary with cost breakdown
        """
        # Fixed commission
        commission = self.commission_per_trade
        
        # Slippage in bps
        slippage_bps = self._calculate_slippage_bps(
            spread_bps, volatility, adv_participation
        )
        
        # Convert slippage to dollars
        slippage_dollars = (slippage_bps / 10000) * qty * price
        
        # Total cost
        total_cost = commission + slippage_dollars
        
        return {
            'commission': commission,
            'slippage_bps': slippage_bps,
            'slippage_dollars': slippage_dollars,
            'total_cost': total_cost,
            'cost_per_share': total_cost / qty if qty > 0 else 0,
            'cost_bps': (total_cost / (qty * price) * 10000) if qty * price > 0 else 0
        }
    
    def _calculate_slippage_bps(
        self,
        spread_bps: float,
        volatility: float,
        adv_participation: float
    ) -> float:
        """
        Calculate slippage in basis points using multi-factor model.
        
        Args:
            spread_bps: Bid-ask spread
            volatility: Daily volatility
            adv_participation: ADV participation rate
            
        Returns:
            Slippage in basis points
        """
        # Base slippage
        slippage = self.base_bps
        
        # Add spread component
        slippage += self.spread_weight * spread_bps
        
        # Add volatility component (convert daily vol to bps)
        vol_bps = volatility * 10000
        slippage += self.vol_weight * vol_bps
        
        # Add market impact component (non-linear in participation)
        impact_factor = np.sqrt(adv_participation / 0.1)  # Normalized to 10% participation
        slippage += self.participation_weight * impact_factor * 10  # 10 bps at 10% participation
        
        return slippage
    
    def estimate_net_return(
        self,
        entry_price: float,
        exit_price: float,
        qty: int,
        entry_spread_bps: float = 10.0,
        exit_spread_bps: float = 10.0,
        volatility: float = 0.02,
        adv_participation: float = 0.05
    ) -> Dict[str, float]:
        """
        Estimate net return after costs for a round-trip trade.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            qty: Number of shares
            entry_spread_bps: Spread at entry
            exit_spread_bps: Spread at exit
            volatility: Volatility
            adv_participation: ADV participation
            
        Returns:
            Dictionary with return metrics
        """
        # Gross P&L
        gross_pnl = (exit_price - entry_price) * qty
        gross_return = (exit_price / entry_price - 1) if entry_price > 0 else 0
        
        # Entry costs
        entry_costs = self.calculate_total_cost(
            qty, entry_price, entry_spread_bps, volatility, adv_participation
        )
        
        # Exit costs
        exit_costs = self.calculate_total_cost(
            qty, exit_price, exit_spread_bps, volatility, adv_participation
        )
        
        # Total costs
        total_costs = entry_costs['total_cost'] + exit_costs['total_cost']
        
        # Net P&L
        net_pnl = gross_pnl - total_costs
        net_return = net_pnl / (qty * entry_price) if qty * entry_price > 0 else 0
        
        return {
            'gross_pnl': gross_pnl,
            'gross_return': gross_return,
            'entry_costs': entry_costs['total_cost'],
            'exit_costs': exit_costs['total_cost'],
            'total_costs': total_costs,
            'net_pnl': net_pnl,
            'net_return': net_return,
            'cost_drag_bps': (total_costs / (qty * entry_price) * 10000) if qty * entry_price > 0 else 0
        }
