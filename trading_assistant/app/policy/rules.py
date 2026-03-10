"""
Do-not-trade rules and policy enforcement.
"""
from typing import Dict, Any, List
from datetime import datetime
from sqlalchemy.orm import Session
import pandas as pd

from ..db import models


class PolicyRules:
    """Enforces do-not-trade rules"""
    
    def __init__(self, config: dict):
        """
        Initialize policy rules.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_adv = config.get('universe', {}).get('min_adv', 1000000)
        self.max_spread_bps = config.get('universe', {}).get('max_spread_bps', 50)
        self.max_daily_loss = config.get('risk', {}).get('max_daily_loss', 20000)
    
    def check_trade_allowed(
        self,
        db: Session,
        symbol: str,
        trade: models.PlanTrade,
        current_price: float,
        market_data: Dict[str, Any],
        news_context: List[Dict[str, Any]]
    ) -> tuple:
        """
        Check if a trade is allowed based on all policy rules.
        
        Args:
            db: Database session
            symbol: Symbol to check
            trade: Trade from plan
            current_price: Current market price
            market_data: Market data dictionary
            news_context: Recent news for symbol
            
        Returns:
            Tuple of (allowed, reasons_blocked)
        """
        reasons_blocked = []
        
        # Check liquidity
        if not self._check_liquidity(market_data):
            reasons_blocked.append('liquidity_below_minimum')
        
        # Check spread
        if not self._check_spread(market_data):
            reasons_blocked.append('spread_too_wide')
        
        # Check news shock
        if not self._check_news_shock(news_context):
            reasons_blocked.append('news_shock_detected')
        
        # Check daily loss limit
        if not self._check_daily_loss_limit(db, trade.plan_id):
            reasons_blocked.append('daily_loss_limit_breached')
        
        # Check regime
        if not self._check_regime(market_data):
            reasons_blocked.append('unfavorable_regime')
        
        # Check do-not-trade conditions from plan
        plan_conditions = trade.do_not_trade_if or {}
        if plan_conditions:
            for condition, threshold in plan_conditions.items():
                if self._evaluate_condition(condition, threshold, market_data):
                    reasons_blocked.append(f'plan_condition_{condition}')
        
        allowed = len(reasons_blocked) == 0
        return allowed, reasons_blocked
    
    def _check_liquidity(self, market_data: Dict[str, Any]) -> bool:
        """Check if liquidity meets minimum requirements"""
        avg_volume = market_data.get('avg_daily_volume', 0)
        return avg_volume >= self.min_adv
    
    def _check_spread(self, market_data: Dict[str, Any]) -> bool:
        """Check if spread is within acceptable limits"""
        spread_bps = market_data.get('spread_bps', 0)
        return spread_bps <= self.max_spread_bps
    
    def _check_news_shock(self, news_context: List[Dict[str, Any]]) -> bool:
        """Check for recent high-risk news"""
        for news in news_context:
            risk_score = news.get('risk_score', 0)
            if risk_score > 0.7:  # High risk threshold
                return False
        return True
    
    def _check_daily_loss_limit(self, db: Session, plan_id: int) -> bool:
        """Check if daily loss limit has been breached"""
        # Get plan
        plan = db.query(models.PremarketPlan).filter(
            models.PremarketPlan.plan_id == plan_id
        ).first()
        
        if not plan:
            return True
        
        # Get today's outcomes
        outcomes = db.query(models.TradeOutcome).filter(
            models.TradeOutcome.plan_id == plan_id,
            models.TradeOutcome.exit_ts.isnot(None)
        ).all()
        
        total_pnl = sum(o.pnl_net or 0 for o in outcomes)
        current_loss = -total_pnl if total_pnl < 0 else 0
        
        return current_loss < self.max_daily_loss
    
    def _check_regime(self, market_data: Dict[str, Any]) -> bool:
        """Check if market regime is favorable"""
        regime = market_data.get('regime', {})
        volatility_regime = regime.get('volatility', 'normal')
        
        # Don't trade in extreme volatility
        if volatility_regime == 'extreme':
            return False
        
        return True
    
    def _evaluate_condition(
        self,
        condition: str,
        threshold: Any,
        market_data: Dict[str, Any]
    ) -> bool:
        """Evaluate a specific condition"""
        if condition == 'volatility_above':
            current_vol = market_data.get('volatility', 0)
            return current_vol > threshold
        
        elif condition == 'volume_below':
            current_volume = market_data.get('current_volume', float('inf'))
            return current_volume < threshold
        
        elif condition == 'gap_above':
            gap = market_data.get('gap', 0)
            return abs(gap) > threshold
        
        return False
