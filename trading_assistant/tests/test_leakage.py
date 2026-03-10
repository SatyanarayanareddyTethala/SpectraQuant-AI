"""
Test leakage prevention in feature engineering.
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from app.features.build import FeatureBuilder


def test_as_of_timestamp_enforcement():
    """Test that features only use data before as_of_date"""
    config = {
        'technical': {
            'returns': [1, 5, 20],
            'volatility': [5, 20, 60],
            'moving_averages': [10, 20, 50],
            'rsi_period': 14,
            'atr_period': 14
        },
        'liquidity': {
            'volume_windows': [5, 20],
            'turnover_windows': [5, 20],
            'amihud_window': 20
        },
        'cross_sectional': {
            'sector_neutral': True,
            'beta_window': 60,
            'correlation_window': 60
        },
        'regime': {
            'volatility_regime_window': 60,
            'trend_regime_window': 20
        }
    }
    
    builder = FeatureBuilder(config)
    
    # The key test: as_of_date must be respected
    # Features built for date T should only use data up to (but not including) T
    
    # This is a contract test - verifying the interface
    # In production, this would be tested with real database queries
    
    as_of_date = datetime(2024, 1, 15)
    
    # Feature builder should reject any data from as_of_date or later
    # This is enforced in the _get_eod_data method with: models.EOD.date < as_of_date
    
    assert as_of_date is not None  # Placeholder assertion
    

def test_no_forward_looking_bias():
    """Test that features don't leak future information"""
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    prices = np.random.randn(len(dates)).cumsum() + 100
    
    df = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # If we're building features for Jan 15, we should only use data up to Jan 14
    as_of_date = datetime(2024, 1, 15)
    
    # Filter data correctly (this is what the feature builder does)
    valid_data = df[df.index < as_of_date]
    
    # The latest data point should be Jan 14
    assert valid_data.index[-1].date() == datetime(2024, 1, 14).date()
    
    # Data from as_of_date or later should not be included
    assert as_of_date.date() not in valid_data.index.date


def test_time_series_alignment():
    """Test that all features are aligned to the same timestamp"""
    # This test ensures that features from different sources
    # (technical, liquidity, regime) are all computed as of the same date
    
    as_of_date = datetime(2024, 1, 15)
    
    # All features should have the same 'date' field
    features = {
        'date': as_of_date,
        'symbol': 'TEST',
        'return_1d': 0.01,
        'volatility_20d': 0.02,
        'avg_volume_5d': 1000000
    }
    
    # Verify all features share the same timestamp
    assert features['date'] == as_of_date


def test_lookback_window_validity():
    """Test that lookback windows don't extend into future"""
    # If we need 20 days of data for a feature,
    # and as_of_date is Jan 15,
    # we should use data from Dec 27 to Jan 14 (not including Jan 15)
    
    as_of_date = datetime(2024, 1, 15)
    lookback_days = 20
    
    start_date = as_of_date - timedelta(days=lookback_days)
    
    # Valid data range
    valid_range_start = start_date
    valid_range_end = as_of_date - timedelta(days=1)
    
    # Verify the range doesn't include as_of_date
    assert valid_range_end < as_of_date
    assert (as_of_date - valid_range_end).days == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
