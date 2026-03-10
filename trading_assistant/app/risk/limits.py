"""
Risk limits and portfolio constraints.
"""
from typing import Dict, Any, List, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

from ..db import models


class RiskLimits:
    """Enforces risk limits and constraints"""
    
    def __init__(self, config: dict):
        """
        Initialize risk limits.
        
        Args:
            config: Risk configuration dictionary
        """
        self.config = config
        self.max_daily_loss = config.get('max_daily_loss', 20000)
        self.max_gross_exposure = config.get('max_gross_exposure', 500000)
        self.max_name_exposure = config.get('max_name_exposure', 100000)
        self.max_sector_exposure = config.get('max_sector_exposure', 200000)
        self.turnover_cap = config.get('turnover_cap', 0.5)
        self.mae_threshold = config.get('mae_threshold', 0.03)
    
    def check_daily_loss_limit(
        self,
        db: Session,
        plan_date: datetime
    ) -> Tuple[bool, float]:
        """
        Check if daily loss limit has been breached.
        
        Args:
            db: Database session
            plan_date: Current trading date
            
        Returns:
            Tuple of (can_trade, current_loss)
        """
        # Query fills for today
        fills = db.query(models.Fill).join(
            models.PremarketPlan
        ).filter(
            models.PremarketPlan.plan_date == plan_date
        ).all()
        
        # Calculate realized P&L
        pnl = 0.0
        positions = {}
        
        for fill in fills:
            key = fill.symbol
            if fill.action == 'BUY':
                if key not in positions:
                    positions[key] = {'qty': 0, 'avg_cost': 0}
                
                old_qty = positions[key]['qty']
                old_cost = positions[key]['avg_cost']
                new_qty = old_qty + fill.qty
                new_cost = ((old_qty * old_cost) + (fill.qty * fill.price)) / new_qty if new_qty > 0 else 0
                
                positions[key]['qty'] = new_qty
                positions[key]['avg_cost'] = new_cost
            
            elif fill.action == 'SELL':
                if key in positions and positions[key]['qty'] > 0:
                    sell_qty = min(fill.qty, positions[key]['qty'])
                    realized_pnl = sell_qty * (fill.price - positions[key]['avg_cost'])
                    pnl += realized_pnl
                    
                    positions[key]['qty'] -= sell_qty
        
        current_loss = -pnl if pnl < 0 else 0
        can_trade = current_loss < self.max_daily_loss
        
        return can_trade, current_loss
    
    def check_gross_exposure(
        self,
        current_positions: List[Dict[str, Any]],
        new_position_value: float
    ) -> bool:
        """
        Check if adding new position would exceed gross exposure limit.
        
        Args:
            current_positions: List of current position dictionaries
            new_position_value: Value of new position to add
            
        Returns:
            True if within limits
        """
        current_exposure = sum(abs(p['value']) for p in current_positions)
        new_exposure = current_exposure + abs(new_position_value)
        
        return new_exposure <= self.max_gross_exposure
    
    def check_name_exposure(
        self,
        symbol: str,
        current_positions: List[Dict[str, Any]],
        new_position_value: float
    ) -> bool:
        """
        Check if position in this symbol would exceed name exposure limit.
        
        Args:
            symbol: Symbol to check
            current_positions: List of current position dictionaries
            new_position_value: Value of new position to add
            
        Returns:
            True if within limits
        """
        current_exposure = sum(
            abs(p['value']) for p in current_positions 
            if p['symbol'] == symbol
        )
        new_exposure = current_exposure + abs(new_position_value)
        
        return new_exposure <= self.max_name_exposure
    
    def check_sector_exposure(
        self,
        sector: str,
        current_positions: List[Dict[str, Any]],
        new_position_value: float
    ) -> bool:
        """
        Check if position would exceed sector exposure limit.
        
        Args:
            sector: Sector to check
            current_positions: List of current position dictionaries
            new_position_value: Value of new position to add
            
        Returns:
            True if within limits
        """
        current_exposure = sum(
            abs(p['value']) for p in current_positions 
            if p.get('sector') == sector
        )
        new_exposure = current_exposure + abs(new_position_value)
        
        return new_exposure <= self.max_sector_exposure
