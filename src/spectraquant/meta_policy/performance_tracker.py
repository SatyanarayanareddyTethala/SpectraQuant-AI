"""Performance tracking for expert signals."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_historical_performance(
    config: dict,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Load historical expert performance data.
    
    Args:
        config: Full configuration dict
        lookback_days: Days of history to load
        
    Returns:
        DataFrame with columns: expert, date, trades, win_rate, avg_return, sharpe
    """
    experts_cfg = config.get("experts", {})
    output_dir = Path(experts_cfg.get("output_dir", "reports/experts"))
    
    if not output_dir.exists():
        logger.info("No historical performance data found")
        return pd.DataFrame(columns=["expert", "date", "trades", "win_rate", "avg_return", "sharpe"])
    
    # Look for performance CSV files
    perf_files = list(output_dir.glob("expert_performance_*.csv"))
    
    if not perf_files:
        logger.info("No performance files found in %s", output_dir)
        return pd.DataFrame(columns=["expert", "date", "trades", "win_rate", "avg_return", "sharpe"])
    
    # Load and combine all performance files within lookback period
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    
    all_perf = []
    for perf_file in perf_files:
        try:
            df = pd.read_csv(perf_file)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] >= cutoff_date]
                all_perf.append(df)
        except Exception as e:
            logger.warning("Failed to load %s: %s", perf_file, e)
    
    if not all_perf:
        return pd.DataFrame(columns=["expert", "date", "trades", "win_rate", "avg_return", "sharpe"])
    
    combined = pd.concat(all_perf, ignore_index=True)
    combined = combined.sort_values("date", ascending=False)
    
    logger.info("Loaded %d performance records for %d experts",
               len(combined), combined["expert"].nunique())
    
    return combined


def compute_expert_weights(
    performance: pd.DataFrame,
    config: dict,
) -> dict[str, float]:
    """Compute expert weights based on historical performance.
    
    Args:
        performance: DataFrame from load_historical_performance()
        config: Full configuration dict
        
    Returns:
        Dict mapping expert name to weight (0-1)
    """
    meta_cfg = config.get("meta_policy", {})
    decay = meta_cfg.get("decay", 0.97)
    weight_floor = meta_cfg.get("weight_floor", 0.05)
    weight_cap = meta_cfg.get("weight_cap", 0.60)
    min_trades = meta_cfg.get("min_trades_for_trust", 20)
    
    if performance.empty:
        # Equal weights if no performance data
        experts_list = config.get("experts", {}).get("list", [])
        if not experts_list:
            return {}
        
        equal_weight = 1.0 / len(experts_list)
        return {expert: equal_weight for expert in experts_list}
    
    # Compute time-weighted performance scores
    expert_scores = {}
    
    for expert in performance["expert"].unique():
        expert_perf = performance[performance["expert"] == expert].sort_values("date")
        
        if len(expert_perf) == 0:
            continue
        
        # Check if expert has enough trades
        total_trades = expert_perf["trades"].sum()
        if total_trades < min_trades:
            # Assign floor weight for experts with insufficient history
            expert_scores[expert] = weight_floor
            continue
        
        # Compute time-weighted score using win rate and avg return
        weighted_score = 0.0
        total_weight = 0.0
        
        # Use itertuples for better performance
        for row in expert_perf.itertuples(index=False):
            days_ago = (datetime.now(timezone.utc) - row.date).days
            time_weight = decay ** days_ago
            
            # Combined score: win_rate * 0.5 + normalized_return * 0.5
            win_rate = row.win_rate if hasattr(row, 'win_rate') else 0.5
            avg_return = row.avg_return if hasattr(row, 'avg_return') else 0.0
            
            # Normalize return to 0-1 scale (assuming returns in -0.1 to 0.1 range)
            normalized_return = max(0, min(1, (avg_return + 0.1) / 0.2))
            
            score = win_rate * 0.5 + normalized_return * 0.5
            
            weighted_score += score * time_weight
            total_weight += time_weight
        
        if total_weight > 0:
            expert_scores[expert] = weighted_score / total_weight
        else:
            expert_scores[expert] = 0.5
    
    if not expert_scores:
        return {}
    
    # Normalize to sum to 1 and apply floor/cap
    total = sum(expert_scores.values())
    if total == 0:
        # Equal weights
        experts_list = list(expert_scores.keys())
        equal_weight = 1.0 / len(experts_list)
        return {expert: equal_weight for expert in experts_list}
    
    # Normalize
    weights = {expert: score / total for expert, score in expert_scores.items()}
    
    # Apply floor
    for expert in weights:
        weights[expert] = max(weight_floor, weights[expert])
    
    # Apply cap
    for expert in weights:
        weights[expert] = min(weight_cap, weights[expert])
    
    # Renormalize after floor/cap
    total = sum(weights.values())
    if total > 0:
        weights = {expert: w / total for expert, w in weights.items()}
    
    logger.info("Computed expert weights: %s", weights)
    
    return weights


def save_performance_snapshot(
    expert_signals: pd.DataFrame,
    actual_returns: pd.DataFrame,
    config: dict,
) -> None:
    """Save a snapshot of expert performance evaluation.
    
    This should be called periodically (e.g., daily) to track expert performance over time.
    
    Args:
        expert_signals: DataFrame with expert signals (from aggregator)
        actual_returns: DataFrame with actual ticker returns (ticker, return)
        config: Full configuration dict
    """
    if expert_signals.empty or actual_returns.empty:
        logger.info("No data to save performance snapshot")
        return
    
    # Merge signals with actual returns
    merged = expert_signals.merge(actual_returns, on="ticker", how="inner")
    
    if merged.empty:
        logger.info("No matching tickers between signals and returns")
        return
    
    # Compute performance metrics per expert
    performance = []
    
    for expert in merged["expert"].unique():
        expert_data = merged[merged["expert"] == expert]
        
        # Compute metrics
        trades = len(expert_data)
        
        # Win rate: signals matching return direction
        correct = 0
        for _, row in expert_data.iterrows():
            signal_direction = 1 if row["action"] == "BUY" else (-1 if row["action"] == "SELL" else 0)
            return_direction = 1 if row["return"] > 0 else (-1 if row["return"] < 0 else 0)
            
            if signal_direction * return_direction > 0:
                correct += 1
        
        win_rate = correct / trades if trades > 0 else 0.5
        avg_return = expert_data["return"].mean()
        
        # Simple Sharpe approximation
        sharpe = avg_return / expert_data["return"].std() if expert_data["return"].std() > 0 else 0.0
        
        performance.append({
            "expert": expert,
            "date": datetime.now(timezone.utc),
            "trades": trades,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "sharpe": sharpe,
        })
    
    # Save to file
    perf_df = pd.DataFrame(performance)
    
    experts_cfg = config.get("experts", {})
    output_dir = Path(experts_cfg.get("output_dir", "reports/experts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"expert_performance_{timestamp}.csv"
    perf_df.to_csv(output_file, index=False)
    
    logger.info("Saved performance snapshot to %s", output_file)
