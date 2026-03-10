"""
Feature engineering with strict as-of timestamps to prevent leakage.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from ..db import models


class FeatureBuilder:
    """Builds features for trading models with leakage prevention"""
    
    def __init__(self, config: dict):
        """
        Initialize feature builder.
        
        Args:
            config: Features configuration dictionary
        """
        self.config = config
        self.technical_config = config.get('technical', {})
        self.liquidity_config = config.get('liquidity', {})
        self.cross_sectional_config = config.get('cross_sectional', {})
        self.regime_config = config.get('regime', {})
    
    def build_features(
        self,
        db: Session,
        symbols: List[str],
        as_of_date: datetime
    ) -> pd.DataFrame:
        """
        Build features for all symbols as of a specific date.
        Ensures no data leakage by only using data available before as_of_date.
        
        Args:
            db: Database session
            symbols: List of symbols to build features for
            as_of_date: Date for which to build features (no future data used)
            
        Returns:
            DataFrame with features for each symbol
        """
        features_list = []
        
        for symbol in symbols:
            # Get historical EOD data
            eod_data = self._get_eod_data(db, symbol, as_of_date)
            
            if eod_data.empty or len(eod_data) < 20:
                continue
            
            # Build feature dict
            feature_dict = {
                'date': as_of_date,
                'symbol': symbol
            }
            
            # Technical features
            tech_features = self._build_technical_features(eod_data)
            feature_dict.update(tech_features)
            
            # Liquidity features
            liquidity_features = self._build_liquidity_features(eod_data)
            feature_dict.update(liquidity_features)
            
            features_list.append(feature_dict)
        
        if not features_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(features_list)
        
        # Cross-sectional features
        df = self._build_cross_sectional_features(df)
        
        # Regime features
        regime_features = self._build_regime_features(db, as_of_date)
        for key, value in regime_features.items():
            df[key] = value
        
        return df
    
    def _get_eod_data(
        self,
        db: Session,
        symbol: str,
        as_of_date: datetime,
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Get EOD data for a symbol up to (but not including) as_of_date.
        
        Args:
            db: Database session
            symbol: Symbol to fetch
            as_of_date: Maximum date (exclusive)
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with OHLCV data
        """
        start_date = as_of_date - timedelta(days=lookback_days)
        
        # Query database
        bars = db.query(models.EOD).filter(
            models.EOD.symbol == symbol,
            models.EOD.date < as_of_date,  # Strict: no data from as_of_date
            models.EOD.date >= start_date
        ).order_by(models.EOD.date).all()
        
        if not bars:
            return pd.DataFrame()
        
        data = []
        for bar in bars:
            data.append({
                'date': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'adj_close': bar.adj_close
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df
    
    def _build_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Build technical indicators"""
        features = {}
        
        if df.empty or len(df) < 20:
            return features
        
        # Returns
        for period in self.technical_config.get('returns', [1, 5, 20]):
            if len(df) >= period:
                features[f'return_{period}d'] = (
                    df['close'].iloc[-1] / df['close'].iloc[-period-1] - 1
                )
        
        # Volatility
        for window in self.technical_config.get('volatility', [5, 20, 60]):
            if len(df) >= window:
                returns = df['close'].pct_change()
                features[f'volatility_{window}d'] = returns.iloc[-window:].std()
        
        # Moving averages
        for window in self.technical_config.get('moving_averages', [10, 20, 50, 200]):
            if len(df) >= window:
                ma = df['close'].rolling(window).mean()
                features[f'ma_{window}'] = ma.iloc[-1]
                features[f'ma_{window}_slope'] = (ma.iloc[-1] / ma.iloc[-window] - 1) if ma.iloc[-window] > 0 else 0
        
        # RSI
        rsi_period = self.technical_config.get('rsi_period', 14)
        if len(df) >= rsi_period + 1:
            features['rsi'] = self._calculate_rsi(df['close'], rsi_period)
        
        # ATR
        atr_period = self.technical_config.get('atr_period', 14)
        if len(df) >= atr_period:
            features['atr'] = self._calculate_atr(df, atr_period)
        
        # Gap
        if len(df) >= 2:
            features['gap'] = (df['open'].iloc[-1] / df['close'].iloc[-2] - 1)
        
        return features
    
    def _build_liquidity_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Build liquidity and cost features"""
        features = {}
        
        if df.empty:
            return features
        
        # Volume features
        for window in self.liquidity_config.get('volume_windows', [5, 20]):
            if len(df) >= window:
                features[f'avg_volume_{window}d'] = df['volume'].iloc[-window:].mean()
        
        # Turnover
        for window in self.liquidity_config.get('turnover_windows', [5, 20]):
            if len(df) >= window:
                turnover = df['volume'].iloc[-window:] * df['close'].iloc[-window:]
                features[f'avg_turnover_{window}d'] = turnover.mean()
        
        # Amihud illiquidity
        amihud_window = self.liquidity_config.get('amihud_window', 20)
        if len(df) >= amihud_window:
            returns = np.abs(df['close'].pct_change())
            dollar_volume = df['volume'] * df['close']
            illiquidity = (returns / dollar_volume).replace([np.inf, -np.inf], np.nan)
            features['amihud_illiquidity'] = illiquidity.iloc[-amihud_window:].mean()
        
        return features
    
    def _build_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build cross-sectional features (ranks)"""
        if df.empty or len(df) < 2:
            return df
        
        # Sector-neutral ranks (simplified: use percentile ranks across all symbols)
        for col in df.columns:
            if col not in ['date', 'symbol'] and pd.api.types.is_numeric_dtype(df[col]):
                df[f'{col}_rank'] = df[col].rank(pct=True)
        
        return df
    
    def _build_regime_features(
        self,
        db: Session,
        as_of_date: datetime
    ) -> Dict[str, Any]:
        """Build market regime features"""
        features = {}
        
        # Get market index data (simplified: use average of all symbols)
        lookback = self.regime_config.get('volatility_regime_window', 60)
        start_date = as_of_date - timedelta(days=lookback + 20)
        
        # Query aggregate market data
        bars = db.query(models.EOD).filter(
            models.EOD.date < as_of_date,
            models.EOD.date >= start_date
        ).all()
        
        if not bars:
            return features
        
        # Group by date and compute average close
        from collections import defaultdict
        date_closes = defaultdict(list)
        
        for bar in bars:
            date_closes[bar.date].append(bar.close)
        
        # Compute market returns
        dates = sorted(date_closes.keys())
        market_closes = [np.mean(date_closes[d]) for d in dates]
        
        if len(market_closes) >= lookback:
            # Volatility regime
            returns = pd.Series(market_closes).pct_change()
            recent_vol = returns.iloc[-lookback:].std()
            historical_vol = returns.std()
            features['volatility_regime'] = 'high' if recent_vol > historical_vol * 1.5 else 'normal'
            
            # Trend regime
            trend_window = self.regime_config.get('trend_regime_window', 20)
            if len(market_closes) >= trend_window:
                trend = (market_closes[-1] / market_closes[-trend_window] - 1)
                features['trend_regime'] = 'up' if trend > 0.05 else ('down' if trend < -0.05 else 'neutral')
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
