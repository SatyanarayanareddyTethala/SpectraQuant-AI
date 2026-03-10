"""
Market data ingestion for EOD and intraday bars.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from ..db import crud


class MarketDataIngester:
    """Handles market data ingestion from various sources"""
    
    def __init__(self, config: dict):
        """
        Initialize market data ingester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.simulation_mode = config.get('simulation', {}).get('enabled', False)
    
    def fetch_eod_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch EOD (end-of-day) data for symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        if self.simulation_mode:
            return self._generate_mock_eod_data(symbols, start_date, end_date)
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if not df.empty:
                    # Normalize column names
                    df = df.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    
                    # Add adj_close (same as close if not available)
                    if 'adj_close' not in df.columns:
                        df['adj_close'] = df['close']
                    
                    data[symbol] = df
            except Exception as e:
                print(f"Error fetching EOD data for {symbol}: {e}")
        
        return data
    
    def fetch_intraday_data(
        self,
        symbols: List[str],
        interval: str = '5m',
        period: str = '5d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday data for symbols.
        
        Args:
            symbols: List of ticker symbols
            interval: Data interval (1m, 5m, 15m, 30m, 60m)
            period: Time period (1d, 5d, 1mo, etc.)
            
        Returns:
            Dictionary mapping symbols to DataFrames with intraday OHLCV data
        """
        if self.simulation_mode:
            return self._generate_mock_intraday_data(symbols, interval, period)
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if not df.empty:
                    # Normalize column names
                    df = df.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    
                    # Calculate VWAP
                    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
                    
                    data[symbol] = df
            except Exception as e:
                print(f"Error fetching intraday data for {symbol}: {e}")
        
        return data
    
    def save_eod_data(self, db: Session, data: Dict[str, pd.DataFrame]):
        """
        Save EOD data to database.
        
        Args:
            db: Database session
            data: Dictionary mapping symbols to DataFrames
        """
        for symbol, df in data.items():
            records = []
            for date, row in df.iterrows():
                records.append({
                    'date': date,
                    'symbol': symbol,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'adj_close': float(row.get('adj_close', row['close']))
                })
            
            if records:
                crud.bulk_insert_eod(db, records)
    
    def save_intraday_data(self, db: Session, data: Dict[str, pd.DataFrame]):
        """
        Save intraday data to database.
        
        Args:
            db: Database session
            data: Dictionary mapping symbols to DataFrames
        """
        for symbol, df in data.items():
            records = []
            for ts, row in df.iterrows():
                records.append({
                    'ts': ts,
                    'symbol': symbol,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'vwap': float(row.get('vwap', row['close']))
                })
            
            if records:
                crud.bulk_insert_bars_5m(db, records)
    
    def _generate_mock_eod_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Generate mock EOD data for simulation mode"""
        import numpy as np
        
        data = {}
        days = (end_date - start_date).days
        
        for symbol in symbols:
            # Generate random walk prices
            np.random.seed(hash(symbol) % (2**32))
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            base_price = 100 + np.random.rand() * 400
            returns = np.random.randn(len(dates)) * 0.02
            prices = base_price * np.cumprod(1 + returns)
            
            df = pd.DataFrame({
                'open': prices * (1 + np.random.randn(len(dates)) * 0.005),
                'high': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
                'low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
                'close': prices,
                'volume': np.random.randint(100000, 1000000, len(dates)),
                'adj_close': prices
            }, index=dates)
            
            data[symbol] = df
        
        return data
    
    def _generate_mock_intraday_data(
        self,
        symbols: List[str],
        interval: str,
        period: str
    ) -> Dict[str, pd.DataFrame]:
        """Generate mock intraday data for simulation mode"""
        import numpy as np
        
        data = {}
        
        # Parse period and interval
        if period.endswith('d'):
            days = int(period[:-1])
        else:
            days = 5
        
        if interval.endswith('m'):
            minutes = int(interval[:-1])
        else:
            minutes = 5
        
        for symbol in symbols:
            # Generate timestamps
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)
            timestamps = pd.date_range(
                start=start_dt,
                end=end_dt,
                freq=f'{minutes}min'
            )
            
            # Generate random walk prices
            np.random.seed(hash(symbol) % (2**32))
            base_price = 100 + np.random.rand() * 400
            returns = np.random.randn(len(timestamps)) * 0.001
            prices = base_price * np.cumprod(1 + returns)
            
            df = pd.DataFrame({
                'open': prices * (1 + np.random.randn(len(timestamps)) * 0.002),
                'high': prices * (1 + np.abs(np.random.randn(len(timestamps)) * 0.003)),
                'low': prices * (1 - np.abs(np.random.randn(len(timestamps)) * 0.003)),
                'close': prices,
                'volume': np.random.randint(10000, 100000, len(timestamps)),
                'vwap': prices
            }, index=timestamps)
            
            data[symbol] = df
        
        return data
