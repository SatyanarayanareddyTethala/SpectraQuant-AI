# News-First Universe + Multi-Expert Meta-Policy

## Overview

This feature implements an optional news-driven trading system with multiple expert strategies and intelligent meta-policy blending.

## Architecture

### 1. News Universe Builder (`src/spectraquant/news/`)

**Purpose**: Build a focused ticker universe based on recent market-moving news.

**Key Functions**:
- `fetch_news_articles()` - Retrieves recent news using NewsAPI provider
- `dedupe_articles()` - Removes duplicate articles using stable hashing
- `load_universe_mapping()` - Maps tickers to company names and aliases
- `score_impact()` - Scores tickers based on news sentiment, recency, and mentions
- `apply_liquidity_filter()` - Filters candidates by minimum average volume
- `apply_price_confirmation()` - Confirms news impact with price/volume signals
- `build_news_universe()` - Main entry point for the full pipeline

**Outputs**: `reports/news/news_candidates_YYYYMMDD_HHMMSS.csv`

### 2. Expert System (`src/spectraquant/experts/`)

**Purpose**: Generate trading signals from multiple specialized strategies.

**Experts Implemented**:
1. **Trend Expert** - Moving average crossovers (20/50 SMA)
2. **Momentum Expert** - RSI and rate of change
3. **Mean Reversion Expert** - Bollinger Bands oversold/overbought
4. **Volatility Expert** - Favors low-volatility defensive positions
5. **Value Expert** - Historical price percentile valuation
6. **News Catalyst Expert** - News-driven signals

**Key Components**:
- `BaseExpert` - Abstract base class with common interface
- `aggregator.run_experts()` - Runs all enabled experts and aggregates signals

**Outputs**: `reports/experts/expert_signals_YYYYMMDD_HHMMSS.csv`

### 3. Meta-Policy System (`src/spectraquant/meta_policy/`)

**Purpose**: Select and blend expert signals based on regime and performance.

**Key Components**:
- **Regime Detection** (`regime.py`) - Detects market volatility and trend states
- **Performance Tracker** (`performance_tracker.py`) - Tracks expert historical performance
- **Arbiter** (`arbiter.py`) - Selects/blends expert signals

**Methods**:
1. **Performance-Weighted Blending** (default) - Weights experts by historical performance
2. **Rule-Based Selection** - Selects experts based on market regime
3. **Contextual Bandit** - Placeholder for future ML-based selection

**Risk Guardrails**:
- Drawdown threshold - Disables trading during large drawdowns
- Calibration threshold - Filters low-confidence signals
- Turnover limits - Controls portfolio churn

**Outputs**: 
- `reports/meta_policy/meta_policy_signals_YYYYMMDD_HHMMSS.csv`
- `reports/meta_policy/regime_state_YYYYMMDD_HHMMSS.csv`

## Configuration

All features are **disabled by default** for backward compatibility.

### Enable News Universe

```yaml
news_universe:
  enabled: true
  lookback_hours: 12
  max_candidates: 50
  min_liquidity_avg_volume: 200000
  sentiment_model: "finbert"  # "finbert", "vader", or "none"
  require_price_confirmation: true
  confirmation:
    method: "gap_or_volume"
    gap_abs_return_threshold: 0.015
    volume_z_threshold: 1.5
    lookback_days: 20
```

### Enable Expert System

```yaml
experts:
  enabled: true
  list: ["trend", "momentum", "mean_reversion", "volatility", "value", "news_catalyst"]
  min_coverage: 5  # Minimum experts required per ticker
  output_dir: "reports/experts"
```

### Enable Meta-Policy

```yaml
meta_policy:
  enabled: true
  method: "perf_weighted"  # "rule_based" | "perf_weighted"
  lookback_days: 90
  decay: 0.97
  weight_floor: 0.05
  weight_cap: 0.60
  min_trades_for_trust: 20
  regime:
    index_ticker: "^NSEI"
    vol_lookback: 20
    trend_fast: 20
    trend_slow: 50
    high_vol_threshold: 0.25
  risk_guardrails:
    disable_on_drawdown: 0.15
    min_calibration: 0.55
```

## Usage Example

```python
from spectraquant.news.universe_builder import build_news_universe
from spectraquant.experts.aggregator import run_experts
from spectraquant.meta_policy.arbiter import run_meta_policy
from spectraquant.config import get_config

# Load config
config = get_config()

# Step 1: Build news universe (optional)
if config.get("news_universe", {}).get("enabled"):
    candidates = build_news_universe(config)
    # Use candidates to filter prices DataFrame

# Step 2: Generate expert signals
prices = load_prices()  # Your price loading logic
expert_signals = run_experts(config, prices)

# Step 3: Apply meta-policy
prices_dir = config["data"]["prices_dir"]
final_signals = run_meta_policy(expert_signals, config, prices_dir)

# Step 4: Use final_signals for portfolio construction
```

## Data Requirements

### News Universe
- **NewsAPI Key**: Set `NEWSAPI_KEY` environment variable
- **Universe CSV**: `data/universe/universe_nse.csv` with ticker symbols
- **Company Aliases** (optional): `data/universe/company_aliases.csv`
- **Source Weights** (optional): `data/news/source_weights.csv`

### Expert System
- **Price Data**: OHLCV data in `data/prices/` (CSV or Parquet)
- **Minimum History**: 50-252 days depending on expert (value needs most)

### Meta-Policy
- **Index Data**: Price data for regime detection (e.g., ^NSEI)
- **Performance History** (optional): Previous expert performance CSVs

## Testing

Run tests:
```bash
# All new tests
pytest tests/test_news_universe.py tests/test_experts.py tests/test_meta_policy.py tests/test_integration_news_experts.py -v

# Integration tests only
pytest tests/test_integration_news_experts.py -v
```

**Test Coverage**:
- 30 unit tests
- 5 integration tests
- 100% pass rate
- Backward compatibility validated

## Performance Considerations

1. **News Fetching**: Rate-limited (1.1s between requests) to comply with NewsAPI limits
2. **Expert Signals**: Parallel-safe, can run concurrently for different tickers
3. **Meta-Policy**: Lightweight, adds minimal overhead (<1s typically)

## Future Enhancements

- [ ] Sentiment model integration (FinBERT, VADER)
- [ ] Contextual bandit meta-policy implementation
- [ ] Real-time news streaming
- [ ] Social media sentiment integration
- [ ] Additional experts (pairs trading, sector rotation, etc.)
- [ ] ML-based regime classification

## Schema Stability

All outputs follow stable schemas with explicit columns:

**News Candidates**: `ticker, score, mentions, top_headlines, asof_utc`

**Expert Signals**: `ticker, action, score, reason, expert, timestamp`

**Meta-Policy Signals**: `ticker, action, score, reason, timestamp`

**Regime State**: `timestamp, volatility, trend, vol_value, trend_value`

All date/time columns are UTC-aware for consistency.
