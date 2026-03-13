"""Meta-policy arbiter for expert selection and signal blending."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from spectraquant.meta_policy.regime import detect_regime, RegimeState
from spectraquant.meta_policy.performance_tracker import (
    load_historical_performance,
    compute_expert_weights,
)

logger = logging.getLogger(__name__)


def rule_based_selection(
    expert_signals: pd.DataFrame,
    regime: RegimeState,
    config: dict,
) -> pd.DataFrame:
    """Select expert signals using rule-based logic based on regime.
    
    Args:
        expert_signals: DataFrame with all expert signals
        regime: Current market regime
        config: Full configuration dict
        
    Returns:
        DataFrame with selected signals
    """
    if expert_signals.empty:
        return expert_signals
    
    # Define expert preferences by regime
    regime_preferences = {
        ("high", "down"): ["volatility", "mean_reversion"],  # Defensive in volatile downtrend
        ("high", "neutral"): ["volatility", "value"],  # Defensive in high vol
        ("high", "up"): ["momentum", "trend"],  # Ride momentum in volatile uptrend
        ("normal", "up"): ["trend", "momentum", "value"],  # Follow trend in normal vol
        ("normal", "down"): ["mean_reversion", "value"],  # Buy dips in normal vol
        ("normal", "neutral"): ["value", "momentum"],  # Balanced approach
        ("low", "up"): ["trend", "value"],  # Trend following in calm uptrend
        ("low", "down"): ["value", "mean_reversion"],  # Value hunting in calm downtrend
        ("low", "neutral"): ["value", "trend"],  # Long-term value in low vol
    }
    
    regime_key = (regime.volatility, regime.trend)
    preferred_experts = regime_preferences.get(regime_key, [])
    
    if not preferred_experts:
        logger.warning("No preference for regime %s; using all experts", regime_key)
        return expert_signals
    
    # Filter to preferred experts
    selected = expert_signals[expert_signals["expert"].isin(preferred_experts)]
    
    logger.info("Rule-based selection: regime=%s, preferred=%s, selected %d/%d signals",
               regime_key, preferred_experts, len(selected), len(expert_signals))
    
    return selected


def performance_weighted_blending(
    expert_signals: pd.DataFrame,
    expert_weights: dict[str, float],
    config: dict,
) -> pd.DataFrame:
    """Blend expert signals using performance-based weights.
    
    Args:
        expert_signals: DataFrame with all expert signals
        expert_weights: Dict mapping expert to weight
        config: Full configuration dict
        
    Returns:
        DataFrame with aggregated signals per ticker
    """
    if expert_signals.empty:
        return pd.DataFrame(columns=["ticker", "action", "score", "reason", "timestamp"])
    
    # Add weights to signals
    expert_signals["weight"] = expert_signals["expert"].map(expert_weights).fillna(0.05)
    
    # Aggregate by ticker
    aggregated = []
    
    for ticker in expert_signals["ticker"].unique():
        ticker_signals = expert_signals[expert_signals["ticker"] == ticker]
        
        # Compute weighted vote for each action
        buy_weight = ticker_signals[ticker_signals["action"] == "BUY"]["weight"].sum()
        sell_weight = ticker_signals[ticker_signals["action"] == "SELL"]["weight"].sum()
        hold_weight = ticker_signals[ticker_signals["action"] == "HOLD"]["weight"].sum()
        
        total_weight = buy_weight + sell_weight + hold_weight
        if total_weight == 0:
            continue
        
        # Determine final action
        if buy_weight > sell_weight and buy_weight > hold_weight:
            action = "BUY"
            score = min(100, (buy_weight / total_weight) * 100)
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            action = "SELL"
            score = min(100, (sell_weight / total_weight) * 100)
        else:
            action = "HOLD"
            score = 50
        
        # Combine reasons from top experts
        top_experts = ticker_signals.nlargest(3, "weight")["expert"].tolist()
        reason = f"Meta-policy consensus from {', '.join(top_experts)}"
        
        aggregated.append({
            "ticker": ticker,
            "action": action,
            "score": score,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc),
        })
    
    result = pd.DataFrame(aggregated)
    logger.info("Performance-weighted blending: %d tickers", len(result))
    
    return result


def apply_risk_guardrails(
    signals: pd.DataFrame,
    config: dict,
    prices_dir: str | Path,
) -> pd.DataFrame:
    """Apply risk guardrails to filter/modify signals.
    
    Args:
        signals: DataFrame with aggregated signals
        config: Full configuration dict
        prices_dir: Path to price data
        
    Returns:
        Filtered DataFrame
    """
    if signals.empty:
        return signals
    
    meta_cfg = config.get("meta_policy", {})
    guardrails = meta_cfg.get("risk_guardrails", {})
    
    disable_on_drawdown = guardrails.get("disable_on_drawdown")
    min_calibration = guardrails.get("min_calibration", 0.55)
    max_turnover = guardrails.get("max_turnover")
    
    # Check drawdown condition
    if disable_on_drawdown is not None:
        # Would need portfolio history to compute actual drawdown
        # For now, use index drawdown as proxy
        regime_cfg = meta_cfg.get("regime", {})
        index_ticker = regime_cfg.get("index_ticker", "^NSEI")
        
        prices_path = Path(prices_dir)
        index_file = prices_path / f"{index_ticker}.csv"
        
        if not index_file.exists():
            index_file = prices_path / f"{index_ticker}.parquet"
        
        if index_file.exists():
            try:
                if index_file.suffix == ".parquet":
                    df = pd.read_parquet(index_file)
                else:
                    df = pd.read_csv(index_file)
                
                close_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else "close")
                
                # Compute drawdown
                df = df.sort_values("Date" if "Date" in df.columns else "date")
                df["cummax"] = df[close_col].cummax()
                df["drawdown"] = (df[close_col] - df["cummax"]) / df["cummax"]
                
                latest_drawdown = abs(df.iloc[-1]["drawdown"])
                
                if latest_drawdown > disable_on_drawdown:
                    logger.warning("Drawdown %.2f%% exceeds threshold %.2f%%; disabling all signals",
                                 latest_drawdown * 100, disable_on_drawdown * 100)
                    return pd.DataFrame(columns=signals.columns)
            
            except Exception as e:
                logger.warning("Failed to check drawdown: %s", e)
    
    # Filter by minimum calibration (score threshold)
    if min_calibration:
        initial_count = len(signals)
        signals = signals[signals["score"] >= min_calibration * 100].copy()
        logger.info("Calibration filter: %d -> %d signals (min_score=%.1f)",
                   initial_count, len(signals), min_calibration * 100)
    
    # Turnover limit would require previous portfolio state
    # Skipping for now as it requires more context
    
    return signals


def run_meta_policy(
    expert_signals: pd.DataFrame,
    config: dict,
    prices_dir: str | Path,
) -> pd.DataFrame:
    """Main entry point: run meta-policy to select/blend expert signals.
    
    Args:
        expert_signals: DataFrame with all expert signals
        config: Full configuration dict
        prices_dir: Path to price data directory
        
    Returns:
        DataFrame with final signals after meta-policy
    """
    meta_cfg = config.get("meta_policy", {})
    
    if not meta_cfg.get("enabled", False):
        logger.info("Meta-policy disabled; returning all expert signals")
        return expert_signals
    
    if expert_signals.empty:
        logger.info("No expert signals to process")
        return expert_signals
    
    method = meta_cfg.get("method", "perf_weighted")
    lookback_days = meta_cfg.get("lookback_days", 90)
    
    # Step 1: Detect regime
    logger.info("Step 1: Detecting market regime...")
    regime = detect_regime(config, prices_dir)
    
    # Step 2: Load historical performance
    logger.info("Step 2: Loading historical performance...")
    performance = load_historical_performance(config, lookback_days)
    
    # Step 3: Apply method
    if method == "rule_based":
        logger.info("Step 3: Applying rule-based selection...")
        final_signals = rule_based_selection(expert_signals, regime, config)
    
    elif method == "perf_weighted":
        logger.info("Step 3: Applying performance-weighted blending...")
        expert_weights = compute_expert_weights(performance, config)
        final_signals = performance_weighted_blending(expert_signals, expert_weights, config)
    
    elif method == "contextual_bandit":
        # TODO: Contextual bandit not yet implemented
        logger.error("Contextual bandit method not yet implemented; falling back to perf_weighted")
        expert_weights = compute_expert_weights(performance, config)
        final_signals = performance_weighted_blending(expert_signals, expert_weights, config)
    
    else:
        logger.warning("Unknown method '%s'; using perf_weighted", method)
        expert_weights = compute_expert_weights(performance, config)
        final_signals = performance_weighted_blending(expert_signals, expert_weights, config)
    
    # Step 4: Apply risk guardrails
    logger.info("Step 4: Applying risk guardrails...")
    final_signals = apply_risk_guardrails(final_signals, config, prices_dir)
    
    # Step 5: Write output
    output_dir = Path("reports/meta_policy")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"meta_policy_signals_{timestamp}.csv"
    final_signals.to_csv(output_file, index=False)
    logger.info("Wrote %d meta-policy signals to %s", len(final_signals), output_file)
    
    # Also write regime state
    regime_file = output_dir / f"regime_state_{timestamp}.csv"
    regime_df = pd.DataFrame([{
        "timestamp": regime.timestamp,
        "volatility": regime.volatility,
        "trend": regime.trend,
        "vol_value": regime.vol_value,
        "trend_value": regime.trend_value,
    }])
    regime_df.to_csv(regime_file, index=False)
    logger.info("Wrote regime state to %s", regime_file)
    
    return final_signals
