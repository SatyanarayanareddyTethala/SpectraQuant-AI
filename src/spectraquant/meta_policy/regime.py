"""Market regime detection for meta-policy."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current market regime state."""
    volatility: str  # "low", "normal", "high"
    trend: str  # "up", "neutral", "down"
    timestamp: pd.Timestamp
    vol_value: float
    trend_value: float


def detect_regime(config: dict, prices_dir: str | Path) -> RegimeState:
    """Detect current market regime.
    
    Args:
        config: Full configuration dict with meta_policy.regime settings
        prices_dir: Path to price data directory
        
    Returns:
        RegimeState object describing current market conditions
    """
    meta_cfg = config.get("meta_policy", {})
    regime_cfg = meta_cfg.get("regime", {})
    
    index_ticker = regime_cfg.get("index_ticker", "^NSEI")
    vol_lookback = regime_cfg.get("vol_lookback", 20)
    trend_fast = regime_cfg.get("trend_fast", 20)
    trend_slow = regime_cfg.get("trend_slow", 50)
    high_vol_threshold = regime_cfg.get("high_vol_threshold", 0.25)
    
    # Load index price data
    prices_path = Path(prices_dir)
    index_file = prices_path / f"{index_ticker}.csv"
    
    if not index_file.exists():
        # Try parquet
        index_file = prices_path / f"{index_ticker}.parquet"
    
    if not index_file.exists():
        logger.warning("Index price data not found: %s; using default regime", index_ticker)
        return RegimeState(
            volatility="normal",
            trend="neutral",
            timestamp=pd.Timestamp.now(tz="UTC"),
            vol_value=0.02,
            trend_value=0.0,
        )
    
    try:
        # Load index data
        if index_file.suffix == ".parquet":
            df = pd.read_parquet(index_file)
        else:
            df = pd.read_csv(index_file)
        
        # Sort by date
        date_col = "Date" if "Date" in df.columns else "date"
        df = df.sort_values(date_col, ascending=False)
        
        close_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else "close")
        
        # Compute volatility
        df["returns"] = df[close_col].pct_change()
        recent_vol = df.head(vol_lookback)["returns"].std()
        
        # Classify volatility
        if recent_vol > high_vol_threshold:
            vol_state = "high"
        elif recent_vol < high_vol_threshold / 2:
            vol_state = "low"
        else:
            vol_state = "normal"
        
        # Compute trend using moving averages
        df["sma_fast"] = df[close_col].rolling(window=trend_fast).mean()
        df["sma_slow"] = df[close_col].rolling(window=trend_slow).mean()
        
        latest = df.iloc[0]
        if pd.notna(latest["sma_fast"]) and pd.notna(latest["sma_slow"]):
            trend_diff = (latest["sma_fast"] - latest["sma_slow"]) / latest["sma_slow"]
            
            if trend_diff > 0.02:  # 2% above
                trend_state = "up"
            elif trend_diff < -0.02:  # 2% below
                trend_state = "down"
            else:
                trend_state = "neutral"
        else:
            trend_diff = 0.0
            trend_state = "neutral"
        
        timestamp = pd.to_datetime(latest[date_col])
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize("UTC")
        
        regime = RegimeState(
            volatility=vol_state,
            trend=trend_state,
            timestamp=timestamp,
            vol_value=recent_vol,
            trend_value=trend_diff,
        )
        
        logger.info("Detected regime: vol=%s (%.2f%%), trend=%s (%.2f%%)",
                   vol_state, recent_vol * 100, trend_state, trend_diff * 100)
        
        return regime
    
    except Exception as e:
        logger.error("Failed to detect regime: %s", e)
        return RegimeState(
            volatility="normal",
            trend="neutral",
            timestamp=pd.Timestamp.now(tz="UTC"),
            vol_value=0.02,
            trend_value=0.0,
        )
