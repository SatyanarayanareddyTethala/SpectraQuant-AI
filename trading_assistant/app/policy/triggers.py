"""
Trigger evaluation for intraday monitoring.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from ..db import models


class TriggerEvaluator:
    """Evaluates trading triggers from premarket plan"""
    
    def __init__(self, config: dict):
        """
        Initialize trigger evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def evaluate_triggers(
        self,
        db: Session,
        plan_id: int,
        current_prices: Dict[str, float],
        current_bars: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all triggers for a plan against current market data.
        
        Args:
            db: Database session
            plan_id: Plan ID to evaluate
            current_prices: Current prices for symbols
            current_bars: Recent bar data for technical analysis
            
        Returns:
            List of triggered trades
        """
        # Get plan trades
        trades = db.query(models.PlanTrade).filter(
            models.PlanTrade.plan_id == plan_id
        ).all()
        
        triggered = []
        
        for trade in trades:
            symbol = trade.symbol
            
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            trigger_config = trade.trigger_json
            
            # Evaluate trigger based on type
            is_triggered, trigger_details = self._evaluate_trigger(
                trigger_config,
                current_price,
                current_bars.get(symbol)
            )
            
            if is_triggered:
                triggered.append({
                    'plan_id': plan_id,
                    'symbol': symbol,
                    'trade': trade,
                    'current_price': current_price,
                    'trigger_details': trigger_details,
                    'triggered_at': datetime.utcnow()
                })
        
        return triggered
    
    def _evaluate_trigger(
        self,
        trigger_config: Dict[str, Any],
        current_price: float,
        bars: Optional[pd.DataFrame]
    ) -> tuple:
        """
        Evaluate a specific trigger.
        
        Args:
            trigger_config: Trigger configuration
            current_price: Current market price
            bars: Recent bar data
            
        Returns:
            Tuple of (is_triggered, details)
        """
        trigger_type = trigger_config.get('type', 'price')
        
        if trigger_type == 'price':
            return self._evaluate_price_trigger(trigger_config, current_price)
        elif trigger_type == 'breakout':
            return self._evaluate_breakout_trigger(trigger_config, current_price, bars)
        elif trigger_type == 'momentum':
            return self._evaluate_momentum_trigger(trigger_config, bars)
        elif trigger_type == 'volume':
            return self._evaluate_volume_trigger(trigger_config, bars)
        else:
            return False, {}
    
    def _evaluate_price_trigger(
        self,
        trigger_config: Dict[str, Any],
        current_price: float
    ) -> tuple:
        """Evaluate price-based trigger"""
        target_price = trigger_config.get('target_price')
        direction = trigger_config.get('direction', 'above')
        
        if direction == 'above' and current_price >= target_price:
            return True, {
                'type': 'price',
                'target': target_price,
                'actual': current_price,
                'direction': direction
            }
        elif direction == 'below' and current_price <= target_price:
            return True, {
                'type': 'price',
                'target': target_price,
                'actual': current_price,
                'direction': direction
            }
        
        return False, {}
    
    def _evaluate_breakout_trigger(
        self,
        trigger_config: Dict[str, Any],
        current_price: float,
        bars: Optional[pd.DataFrame]
    ) -> tuple:
        """Evaluate breakout trigger"""
        if bars is None or len(bars) < 20:
            return False, {}
        
        lookback = trigger_config.get('lookback', 20)
        resistance = bars['high'].tail(lookback).max()
        support = bars['low'].tail(lookback).min()
        
        breakout_type = trigger_config.get('breakout_type', 'resistance')
        
        if breakout_type == 'resistance' and current_price > resistance:
            return True, {
                'type': 'breakout',
                'breakout_type': 'resistance',
                'level': resistance,
                'current_price': current_price
            }
        elif breakout_type == 'support' and current_price < support:
            return True, {
                'type': 'breakout',
                'breakout_type': 'support',
                'level': support,
                'current_price': current_price
            }
        
        return False, {}
    
    def _evaluate_momentum_trigger(
        self,
        trigger_config: Dict[str, Any],
        bars: Optional[pd.DataFrame]
    ) -> tuple:
        """Evaluate momentum trigger"""
        if bars is None or len(bars) < 10:
            return False, {}
        
        # Calculate short-term momentum
        period = trigger_config.get('period', 5)
        threshold = trigger_config.get('threshold', 0.02)
        
        momentum = (bars['close'].iloc[-1] / bars['close'].iloc[-period] - 1)
        
        direction = trigger_config.get('direction', 'up')
        
        if direction == 'up' and momentum >= threshold:
            return True, {
                'type': 'momentum',
                'momentum': momentum,
                'threshold': threshold,
                'direction': 'up'
            }
        elif direction == 'down' and momentum <= -threshold:
            return True, {
                'type': 'momentum',
                'momentum': momentum,
                'threshold': threshold,
                'direction': 'down'
            }
        
        return False, {}
    
    def _evaluate_volume_trigger(
        self,
        trigger_config: Dict[str, Any],
        bars: Optional[pd.DataFrame]
    ) -> tuple:
        """Evaluate volume trigger"""
        if bars is None or len(bars) < 20:
            return False, {}
        
        # Calculate average volume
        avg_volume = bars['volume'].tail(20).mean()
        current_volume = bars['volume'].iloc[-1]
        
        volume_spike_threshold = trigger_config.get('spike_threshold', 2.0)
        
        if current_volume >= avg_volume * volume_spike_threshold:
            return True, {
                'type': 'volume',
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'ratio': current_volume / avg_volume if avg_volume > 0 else 0
            }
        
        return False, {}
