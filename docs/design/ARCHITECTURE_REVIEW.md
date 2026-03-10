# SpectraQuant-AI — Principal Architecture Review

**Date:** 2026-03-10  
**Scope:** Full repository audit — `src/spectraquant` (V2) vs `src/spectraquant_v3` (V3),
execution flows, contract boundaries, duplication, legacy overlap, and staged refactor plan.

---

## Table of Contents

1. [Top-Level Repository Structure](#1-top-level-repository-structure)
2. [Package-by-Package Analysis](#2-package-by-package-analysis)
   - 2.1 [src/spectraquant (V2)](#21-srcspectraquant-v2)
   - 2.2 [src/spectraquant_v3 (V3)](#22-srcspectraquant_v3-v3)
3. [Execution Flows](#3-execution-flows)
   - 3.1 [V2 Equity Pipeline](#31-v2-equity-pipeline)
   - 3.2 [V2 Crypto Pipeline](#32-v2-crypto-pipeline)
   - 3.3 [V3 Equity Pipeline](#33-v3-equity-pipeline)
   - 3.4 [V3 Crypto Pipeline](#34-v3-crypto-pipeline)
4. [Contract Boundaries](#4-contract-boundaries)
5. [Duplication and Legacy Overlap](#5-duplication-and-legacy-overlap)
6. [Safest Staged Refactor Plan](#6-safest-staged-refactor-plan)

---

## 1. Top-Level Repository Structure

```
SpectraQuant-AI/
│
├── src/
│   ├── spectraquant/         ← V2 package (247 .py files, ~34 700 LOC)
│   └── spectraquant_v3/      ← V3 package (118 .py files, ~17 200 LOC)
│
├── config/
│   ├── base.yaml             ← V2 global config (flat file, SPECTRAQUANT_CONFIG env-var)
│   ├── crypto.yaml
│   ├── equities.yaml
│   ├── logging.yaml
│   ├── nlp_training.yaml
│   └── v3/                   ← V3 config tree (SPECTRAQUANT_V3_CONFIG_DIR env-var)
│       ├── base.yaml
│       ├── crypto.yaml
│       ├── equities.yaml
│       ├── strategies.yaml
│       ├── providers.yaml
│       ├── risk.yaml
│       └── news.yaml
│
├── tests/                    ← Mixed V2 + V3 test suite (~90 files)
│   └── v3/                   ← V3-specific tests (~13 files)
│
├── dashboard/                ← Streamlit dashboard (uses spectraquant V2 helpers)
├── trading_assistant/        ← Archived standalone app (do not develop)
├── archive/                  ← Archived V2 code snapshots and old docs
├── alembic/                  ← Root Alembic migrations (event_store schema)
├── data/universe/            ← Static universe CSVs (NSE, LSE, etc.)
├── reports/                  ← Runtime pipeline output directory (gitignored)
├── docs/                     ← Design docs, how-tos, implementation reports
├── scripts/                  ← Utility scripts
│
├── pyproject.toml            ← Defines BOTH CLI entry points:
│                                 spectraquant → spectraquant.cli.main:main
│                                 sqv3         → spectraquant_v3.cli.main:main
├── requirements.txt          ← V2 dependencies
├── requirements-v3.txt       ← V3 dependencies
└── requirements.lock.txt     ← Locked dependencies
```

### Key tension points

| Item | Problem |
|---|---|
| `config/` vs `config/v3/` | Two parallel config trees; no shared loader |
| `tests/` vs `tests/v3/` | Tests are interleaved, not isolated per package |
| `trading_assistant/` still present at root | Was archived to `archive/` in READMEs but the live copy remains |
| `trading_assistant_runner.py` at root | Orphaned entry point for the archived app |
| `perplexity_research_agent.py` at root | Loose research script, not packaged |
| `test_perplexity.py` at root | Loose test file outside `tests/` |
| `alembic/` at root + `trading_assistant/alembic.ini` | Duplicate migration configurations |

---

## 2. Package-by-Package Analysis

### 2.1 `src/spectraquant` (V2)

**Role:** Production equities-focused research pipeline.  
**Version:** `0.5.0` (pyproject.toml)  
**CLI entry point:** `spectraquant` → `spectraquant.cli.main:main`  
**Status (per README):** *Production — equities-focused*  
**Actual state:** Monolithic, still-evolving; the `cli/main.py` alone is **4 249 lines**.

#### Directory map (2nd-level significant modules)

```
src/spectraquant/
├── __init__.py               # __version__ = "0.5.0"
├── config.py                 # Flat YAML loader; DEFAULT_CONFIG dict; 100+ key defaults
│
├── core/                     # Shared runtime contracts
│   ├── enums.py              # AssetClass, RunMode, SignalStatus, NoSignalReason
│   ├── errors.py             # SpectraQuantError hierarchy (9 exception types)
│   ├── schema.py             # SCHEMA_VERSIONS, validate_predictions/signals/portfolio
│   ├── providers/            # base.py, yfinance.py, mock.py — provider abstraction
│   ├── portfolio.py          # validate_weight_matrix
│   ├── diagnostics.py        # record_input/output, run_summary
│   ├── eval.py               # evaluate_feature_drift, evaluate_portfolio, …
│   ├── io.py                 # write_portfolio/predictions/signals
│   ├── model_registry.py     # register/load/promote model artifacts
│   ├── policy.py             # PolicyLimits, enforce_policy
│   ├── predictions.py        # ANNUAL_RETURN_* constants
│   ├── ranking.py            # add_rank, normalize_scores
│   ├── regime.py             # compute_regime (VIX + MA-based)
│   ├── run_manifest.py       # write_early_exit_manifest (standalone helper)
│   ├── time.py               # ensure_datetime_column, normalize_time_index
│   ├── trading_time.py       # market-hours utilities
│   └── universe.py           # load_universe, update_nse_universe
│
├── equities/                 # V2 equity sub-pipeline
│   ├── execution/paper_executor.py
│   ├── ingestion/price_downloader.py
│   ├── policy/allocator.py, meta_policy.py
│   ├── reporting/reporter.py
│   ├── signals/              # 7 agents: momentum, mean_reversion, breakout,
│   │                         #   volatility, regime, quality, news_sentiment
│   │   ├── _base_agent.py    # BaseEquityAgent / AgentOutput dataclass
│   │   └── ...
│   ├── symbols/equity_symbol_mapper.py, equity_symbol_registry.py
│   └── universe/equity_universe_builder.py
│
├── crypto/                   # V2 crypto sub-pipeline
│   ├── dataset/ingest.py
│   ├── exchange/ccxt_exec.py, coinbase_exec.py, coinbase_ws.py
│   ├── features/funding_oi.py, microstructure.py
│   ├── symbols/crypto_symbol_mapper.py, crypto_symbol_registry.py
│   └── universe/universe_builder.py, news_crypto_universe.py, quality_gate.py
│
├── pipeline/
│   ├── equity_run.py         # run_equity_pipeline() — V2 equity orchestrator
│   └── crypto_run.py         # V2 crypto orchestrator (inlines yfinance download)
│
├── alpha/
│   ├── experts/              # _base.py + 6 expert classes (momentum, mean_reversion,
│   │                         #   news_catalyst, trend, value, volatility)
│   ├── meta_policy.py        # detect_regime, compute_expert_weights, blend_signals
│   ├── factors.py            # compute_alpha_factors
│   ├── factor_registry.py    # register_default_factors
│   └── scorer.py             # compute_alpha_score, compute_factor_contributions
│
├── meta_policy/              # Duplicate meta-policy layer (separate from alpha/)
│   ├── arbiter.py            # rule_based_selection, performance_weighted_blending
│   ├── performance_tracker.py
│   └── regime.py             # detect_regime (second implementation)
│
├── agents/                   # Crypto-focused agent system
│   ├── arbiter.py
│   ├── regime.py
│   ├── registry.py
│   └── agents/               # carry_funding, mean_reversion, momentum,
│                             #   news_catalyst, onchain_flow, volatility
│
├── news/                     # News processing stack
│   ├── analog_memory.py
│   ├── collector.py
│   ├── dedupe.py, embeddings.py
│   ├── entity_linker.py, entity_map.py
│   ├── event_classifier.py, event_ontology.py
│   ├── impact_scoring.py
│   ├── schema.py
│   └── universe_builder.py
│
├── experts/                  # Standalone expert aggregator (different from alpha/experts)
│   ├── base.py
│   ├── aggregator.py
│   ├── momentum.py, mean_reversion.py, news_catalyst.py, trend.py, volatility.py
│
├── intelligence/             # Live-trading / scheduling layer
│   ├── config.py, state.py
│   ├── emailer.py, scheduler.py, trade_planner.py, trade_logger.py
│   ├── policy.py, risk.py, regime_engine.py
│   ├── learning.py, meta_learner.py, model_selector.py
│   ├── premarket.py, intraday.py, hourly_news.py
│   ├── analog_memory.py, failure_memory.py, capital_intelligence.py
│   ├── execution_intelligence.py, bootstrap.py
│   ├── cognition/            # Higher-order reasoning modules
│   └── db/                   # Database persistence (SQLAlchemy)
│   └── research_lab/         # Research management
│
├── dataset/                  # ML dataset assembly
│   ├── builder.py            # build_dataset
│   └── io.py                 # load_dataset, latest_dataset_path_from_manifest
│
├── features/                 # OHLCV feature computation
│   └── ohlcv_features.py
│
├── models/                   # ML model layer
│   ├── train.py, predict.py
│   ├── ensemble.py, impact_model.py
│
├── portfolio/
│   ├── allocator.py          # allocate_risk_parity, allocate_vol_target
│   ├── constraints.py, risk.py, simulator.py
│
├── qa/                       # Quality gate library (8 modules)
│   ├── quality_gates.py      # run_quality_gates_*
│   ├── output_check.py, pipeline_check.py
│   ├── model_check.py, filesystem_check.py, gitignore_check.py
│   ├── hash_utils.py, mode.py
│   └── research_isolation.py
│
├── sentiment/                # News sentiment provider
├── stress/                   # Param sensitivity, regime performance
├── analysis/                 # Feature pruning, model comparison, run comparison
├── explain/                  # Portfolio rationale builder
├── nlp/                      # Multi-task trainer, deduplication, weak supervision
├── onchain/                  # On-chain data collectors and features
├── pricing/                  # Price target / downside engine
├── intraday/                 # Intraday learner
├── mlops/                    # Auto-retrain
├── regime/                   # Simple regime classifier
├── research/                 # Alpha factors, factor registry
├── governance/               # Prediction log
├── providers/                # Router, yfinance adapter, newsapi adapter
└── utils/                    # News universe, optional deps
```

**Key V2 architectural problems:**

1. **Monolithic CLI (`cli/main.py` = 4 249 lines).**  
   All pipeline logic, configuration resolution, QA gates, and output writing are packed into a single file.  The file imports from ~25 different subpackages.  Any import error anywhere kills the entire CLI.

2. **Three competing meta-policy implementations:**  
   `alpha/meta_policy.py`, `meta_policy/arbiter.py`, and `agents/arbiter.py` each implement regime detection and expert blending with different APIs.

3. **Two expert/agent sub-systems:**  
   `alpha/experts/` (equity-oriented, DataFrame-based) and `agents/agents/` (crypto-oriented, dict-based) implement parallel sets of the same conceptual agents (momentum, mean_reversion, volatility, news_catalyst) with no shared base contract.  `experts/` is a third system.

4. **No `RunContext`.** State is passed through function arguments or module-level globals.  `config.py` uses a module-level `DEFAULT_CONFIG` dict that grows as new sub-systems are added.  There is no single object representing "this run."

5. **Mixed asset-class design.**  
   `pipeline/crypto_run.py` downloads crypto prices via yfinance (`-USD` suffix) while `pipeline/equity_run.py` uses a dedicated `EquityPriceDownloader`.  Both live in the same `pipeline/` module; the asset-class guard is a comment, not a type.

6. **Duplicate `regime` implementations.**  
   `core/regime.py` (`compute_regime`), `meta_policy/regime.py` (`detect_regime`, `RegimeState`), and `alpha/meta_policy.py::detect_regime` are three separate implementations with three different interfaces.

---

### 2.2 `src/spectraquant_v3` (V3)

**Role:** Production-grade, strictly-segregated, crypto + equity research platform.  
**Version:** `3.0.0`  
**CLI entry point:** `sqv3` → `spectraquant_v3.cli.main:main`  
**Status (per README):** *Scaffold — crypto + equities, strict segregation*

#### Directory map

```
src/spectraquant_v3/
├── __init__.py               # __version__ = "3.0.0"
│
├── core/                     # Shared abstractions (zero asset-class logic here)
│   ├── enums.py              # AssetClass, RunMode, RunStage, SignalStatus,
│   │                         #   RunStatus, ExecutionMode
│   ├── errors.py             # SpectraQuantError hierarchy (12 exception types)
│   ├── schema.py             # SymbolRecord, QARow, SignalRow, AllocationRow,
│   │                         #   validate_ohlcv_dataframe
│   ├── context.py            # RunContext — single thread-through object
│   ├── config.py             # load_config, get_crypto_config, get_equity_config
│   ├── paths.py              # ProjectPaths, RunPaths — canonical dir layout
│   ├── manifest.py           # RunManifest — JSON run manifest writer
│   ├── qa.py                 # QAMatrix — per-symbol audit rows
│   ├── cache.py              # CacheManager
│   ├── logging.py            # Structured logging helpers
│   ├── ingestion_result.py   # IngestionResult — typed ingestion outcome
│   └── async_engine.py       # Async execution utilities
│
├── crypto/                   # Crypto-only modules (must never import equities)
│   ├── ingestion/            # 7 ingestors: ohlcv_loader, price_downloader,
│   │                         #   news_ingestion, funding, onchain, orderbook,
│   │                         #   missing_bars, audit_log
│   │   └── providers/        # binance_futures, bybit, ccxt, coingecko,
│   │                         #   cryptocompare, cryptopanic, glassnode
│   ├── features/engine.py    # CryptoFeatureEngine
│   ├── signals/              # cross_sectional_momentum, hybrid, momentum
│   ├── news/                 # news_fetcher, normalizer, features, sentiment, store
│   ├── symbols/              # mapper.py, registry.py (CryptoSymbolRegistry)
│   └── universe/             # builder.py, universe_engine.py
│
├── equities/                 # Equity-only modules (must never import crypto)
│   ├── ingestion/            # ohlcv_loader, price_downloader, news_ingestion
│   │   └── providers/        # yfinance_provider.py, rss_provider.py
│   ├── features/engine.py    # EquityFeatureEngine
│   ├── signals/momentum.py   # EquityMomentumAgent
│   ├── symbols/              # mapper.py, registry.py
│   └── universe/builder.py
│
├── pipeline/                 # Orchestrators (shared, asset-class-parametrised)
│   ├── crypto_pipeline.py    # run_crypto_pipeline()
│   ├── equity_pipeline.py    # run_equity_pipeline()
│   ├── meta_policy.py        # MetaPolicy, MetaPolicyConfig, PolicyDecision
│   ├── allocator.py          # Allocator, AllocatorConfig
│   ├── reporter.py           # PipelineReporter
│   └── _strategy_runner.py   # run_strategy() — ties strategy → pipeline
│
├── strategies/               # Declarative strategy system
│   ├── strategy_definition.py # StrategyDefinition, RiskConfig dataclasses
│   ├── registry.py           # StrategyRegistry (4 built-in strategies)
│   ├── loader.py             # StrategyLoader
│   ├── agents/               # registry.py, runner.py
│   └── allocators/           # rank_vol_target_allocator.py, registry.py
│   └── policies/registry.py
│
├── backtest/
│   ├── engine.py             # BacktestEngine (walk-forward)
│   └── results.py            # BacktestResults, RebalanceSnapshot
│
├── execution/
│   ├── simulator.py          # ExecutionSimulator
│   └── result.py             # ExecutionResult
│
├── experiments/
│   ├── experiment_manager.py # ExperimentManager
│   ├── result_store.py       # ResultStore
│   └── run_tracker.py        # RunTracker
│
├── feature_store/
│   ├── store.py              # FeatureStore (Parquet + DuckDB)
│   └── metadata.py           # FeatureSetMetadata
│
├── strategy_portfolio/
│   ├── portfolio.py          # StrategyPortfolio (multi-strategy combiner)
│   └── result.py             # PortfolioResult
│
├── monitoring/
│   ├── monitor.py            # PipelineMonitor
│   └── health.py             # HealthReport
│
├── research/
│   └── dataset_builder.py    # DatasetBuilder
│
└── cli/
    ├── main.py               # Typer app (sqv3); ~120 lines
    └── commands/             # backtest, crypto, equities, experiment,
                              #   feature_store, research, strategy, strategy_portfolio
```

**V3 architectural strengths:**

1. **Strict runtime segregation enforced at import-time.**  Every `crypto/` module has a module-level guard: *"This module must never import from `spectraquant_v3.equities`."*  
   `MixedAssetClassRunError` / `AssetClassLeakError` are raised eagerly.

2. **`RunContext` as the single thread-through object.**  
   Every pipeline stage receives `ctx` containing `run_id`, `as_of`, `config`, `paths`, `cache`, `manifest`, and `qa_matrix`.  No module-level globals.

3. **Typed inter-stage contracts.**  
   `SymbolRecord`, `QARow`, `SignalRow`, `AllocationRow`, `IngestionResult` are proper dataclasses.  `validate_ohlcv_dataframe` is called at every ingestor boundary.

4. **Declarative strategy system.**  
   `StrategyDefinition` describes a strategy as data; `StrategyRegistry` is a process-wide catalog; `StrategyLoader` resolves definitions to components.  Strategies are composable without changing pipeline code.

5. **Clean, small CLI (120 lines vs 4 249).**  
   Sub-commands are in dedicated files; no business logic in the CLI layer.

**V3 current limitations:**

1. **Signal agents are sparse.**  Only `EquityMomentumAgent` and three crypto agents (`momentum`, `cross_sectional_momentum`, `hybrid`) are implemented.  V2 has 7 equity agents and 6 crypto agents.

2. **No news pipeline integration in V3 equities.**  
   `equities/ingestion/news_ingestion.py` and `equities/ingestion/providers/rss_provider.py` exist but are not wired into the equity pipeline.

3. **`meta_policy.py` uses equal-weight composite scoring.**  
   The V2 performance-weighted, regime-aware blending logic has not been ported.

4. **No ML model layer in V3.**  
   V2 has `models/train.py`, `models/predict.py`, `models/ensemble.py`, `dataset/builder.py`, and `mlops/auto_retrain.py`.  None of these exist in V3.

5. **`strategy_portfolio/portfolio.py` allows mixed asset classes.**  
   `StrategyPortfolio.strategy_ids` can combine crypto and equity strategies, which the v3 pipeline correctly supports; however, the combined portfolio execution path is not guarded.

---

## 3. Execution Flows

### 3.1 V2 Equity Pipeline

```
spectraquant equity-run [--from-news] [--dry-run]
     │
     ▼
cli/main.py::equity_run_cmd()
  ├─ config_module.get_config()           # flat YAML merge
  ├─ enforce_policy(tickers, config)      # core/policy.py
  ├─ (optional) run_news_universe_scan()  # news/universe_builder.py
  │
  ├─ pipeline/equity_run.py::run_equity_pipeline(cfg)
  │   ├─ EquityUniverseBuilder.build()    # equities/universe/equity_universe_builder.py
  │   ├─ EquityPriceDownloader.download() # equities/ingestion/price_downloader.py
  │   │     └─ ohlcv_result.assert_ohlcv_available()  # raises EmptyOHLCVError
  │   ├─ [MomentumAgent, MeanReversionAgent, BreakoutAgent,
  │   │    VolatilityAgent, RegimeAgent, QualityAgent,
  │   │    NewsSentimentAgent].run(df, sym) → AgentOutput
  │   ├─ EquityMetaPolicy.blend()        # equities/policy/meta_policy.py
  │   ├─ EquityAllocator.allocate()      # equities/policy/allocator.py
  │   ├─ EquityPaperExecutor.execute()   # equities/execution/paper_executor.py
  │   └─ EquityReporter.write()          # equities/reporting/reporter.py
  │
  ├─ run_quality_gates_predictions(df)   # qa/quality_gates.py
  ├─ evaluate_portfolio(results)         # core/eval.py
  └─ write_portfolio / write_signals     # core/io.py
```

**Key V2 equity contracts:**
- `AgentOutput` (dataclass): `signal_score ∈ [-1, 1]`, `confidence ∈ [0, 1]`, `status: SignalStatus`
- `EquityRunReport` (dataclass): `run_id`, `weights`, `signals`, `qa`, `blocked`, `report_path`
- OHLCV inputs must have columns `{open, high, low, close, volume}`, UTC DatetimeIndex

### 3.2 V2 Crypto Pipeline

```
spectraquant crypto-run
     │
     ▼
cli/main.py::crypto_run_cmd()
  ├─ config_module.get_config()
  │
  ├─ pipeline/crypto_run.py::run_crypto_pipeline(cfg)
  │   ├─ _download_prices_yfinance()     # inline yfinance wrapper
  │   ├─ _build_inputs_matrix()          # availability flags per symbol
  │   ├─ agents/registry.py → run agents # carry_funding, mean_reversion, momentum,
  │   │                                  #   news_catalyst, onchain_flow, volatility
  │   ├─ alpha/meta_policy.py::blend_signals()
  │   └─ portfolio/allocator.py::allocate_vol_target()
  │
  └─ (various qa and io calls from cli/main.py)
```

**Note:** The V2 crypto pipeline is less structured than the equity pipeline.  
The `crypto_run.py` orchestrator inlines price downloading, mixes agent running with
allocation, and does not use the `EquityPaperExecutor` or `EquityReporter` equivalents.

### 3.3 V3 Equity Pipeline

```
sqv3 equity run [--mode normal|test|refresh] [--dry-run]
     │
     ▼
cli/commands/equities.py::equity_run_cmd()
  └─ pipeline/equity_pipeline.py::run_equity_pipeline(cfg)
       │
       └─ with RunContext.create(EQUITY, run_mode, cfg) as ctx:
            │
            ├─ Stage 1: Symbol registry
            │   └─ equities/symbols/registry.py::build_registry_from_config()
            │
            ├─ Stage 2: Universe
            │   └─ equities/universe/builder.py::EquityUniverseBuilder.build()
            │       → UniverseArtifact(included_symbols, excluded_symbols, qa_rows)
            │
            ├─ Stage 3: Features
            │   └─ equities/features/engine.py::EquityFeatureEngine.compute()
            │       → {symbol: pd.DataFrame}  (OHLCV + derived features)
            │
            ├─ Stage 4: Signals
            │   └─ strategies/agents/runner.py::run_signal_agent(agent, feature_map)
            │       → [SignalRow]
            │
            ├─ Stage 5: Meta-policy
            │   └─ pipeline/meta_policy.py::MetaPolicy.run([SignalRow])
            │       → [PolicyDecision]
            │
            ├─ Stage 6: Allocation
            │   └─ pipeline/allocator.py::Allocator.allocate([PolicyDecision])
            │       → [AllocationRow]
            │
            ├─ Stage 7: Reporting
            │   └─ pipeline/reporter.py::PipelineReporter.write()
            │
            └─ ctx.write_qa_matrix()  ← always, even on failure
```

**Key V3 equity contracts:**
- `SignalRow` (dataclass): `signal_score ∈ [-1, 1]`, `confidence ∈ [0, 1]`, `status: SignalStatus`, `run_id`, `canonical_symbol`
- `PolicyDecision` (dataclass): `composite_score`, `composite_confidence`, `passed: bool`, `reason: str`
- `AllocationRow` (dataclass): `target_weight ∈ [0, 1]`, `blocked: bool`, `blocked_reason: str`

### 3.4 V3 Crypto Pipeline

Identical stage structure to §3.3 with `AssetClass.CRYPTO` and crypto-specific components:

```
sqv3 crypto run
  └─ pipeline/crypto_pipeline.py::run_crypto_pipeline(cfg)
       └─ with RunContext.create(CRYPTO, run_mode, cfg) as ctx:
            ├─ crypto/symbols/registry.py::build_registry_from_config()
            ├─ crypto/universe/builder.py::CryptoUniverseBuilder.build()
            ├─ crypto/features/engine.py::CryptoFeatureEngine.compute()
            ├─ strategies/agents/runner.py::run_signal_agent(agent, feature_map)
            │   [agents from strategies/agents/registry.py, e.g. crypto_momentum_v1]
            ├─ pipeline/meta_policy.py::MetaPolicy.run()
            ├─ pipeline/allocator.py::Allocator.allocate()
            └─ pipeline/reporter.py + ctx.write_qa_matrix()
```

**Hard segregation rule:** `pipeline/crypto_pipeline.py` has a module-level comment
and the `RunContext` is constructed with `AssetClass.CRYPTO`; any import of
`spectraquant_v3.equities` from within crypto modules raises `AssetClassLeakError`.

---

## 4. Contract Boundaries

### Core shared types (V3 `core/schema.py`)

| Type | Purpose | Direction |
|---|---|---|
| `SymbolRecord` | Canonical symbol + provider mapping | Universe → all downstream |
| `QARow` | One data-availability audit row per symbol | Ingestion → QAMatrix |
| `IngestionResult` | Typed outcome of one ingest attempt | Ingestors → Pipeline |
| `SignalRow` | One agent's signal for one symbol | Agents → MetaPolicy |
| `PolicyDecision` | Composite decision per symbol | MetaPolicy → Allocator |
| `AllocationRow` | Target weight + blocked flag | Allocator → Reporter/Execution |

### V2 shared types (`core/schema.py` + `equities/signals/_base_agent.py`)

| Type | Purpose | Notes |
|---|---|---|
| `AgentOutput` | V2 equity agent output dataclass | Equivalent to V3 `SignalRow` but different field names and ranges |
| SCHEMA_COLUMNS dict | Column lists for CSV validation | Dict-based, not typed |
| `PolicyLimits` (frozen dataclass) | max_positions, max_weight, max_turnover | Used by `enforce_policy` only |

### Boundary violations / gaps

1. **`AgentOutput` (V2) vs `SignalRow` (V3)** are semantically identical but structurally different:
   - V2: `signal_score ∈ [-1, 1]` — same as V3
   - V2: `status: SignalStatus` enum — same as V3
   - V2 adds: `metadata: dict`, `required_inputs: list`, `available_inputs: list`
   - V3 adds: `run_id`, `timestamp`, `horizon`, `agent_id` (all on `SignalRow`)
   - These diverge enough that a shared base type is non-trivial without V2 migration.

2. **Config boundary:**  
   - V2 uses `spectraquant.config.get_config()` returning a flat merged dict loaded from `config.yaml`.  
   - V3 uses `spectraquant_v3.core.config.get_crypto_config()` / `get_equity_config()` returning asset-class-specific dicts from `config/v3/`.  
   - No shared config loader; both are standalone.

3. **Manifest boundary:**  
   - V2 has `core/run_manifest.py::write_early_exit_manifest()` — a standalone function, no manifest object.  
   - V3 has `core/manifest.py::RunManifest` — a full lifecycle object with `mark_stage`, `add_error`, `write()`.

4. **QA boundary:**  
   - V2 has `qa/quality_gates.py` with 8 top-level functions called from `cli/main.py`.  
   - V3 has `core/qa.py::QAMatrix` collecting typed `QARow` objects, persisted as JSON by `ctx.write_qa_matrix()`.

---

## 5. Duplication and Legacy Overlap

### 5.1 Structural duplicates

| Concept | V2 path | V3 path | Notes |
|---|---|---|---|
| `AssetClass` enum | `spectraquant.core.enums.AssetClass` | `spectraquant_v3.core.enums.AssetClass` | Identical values; separate types |
| `RunMode` enum | `spectraquant.core.enums.RunMode` | `spectraquant_v3.core.enums.RunMode` | Identical values |
| `SignalStatus` enum | `spectraquant.core.enums.SignalStatus` | `spectraquant_v3.core.enums.SignalStatus` | V2 has `DEGRADED`; V3 does not |
| `SpectraQuantError` hierarchy | `spectraquant.core.errors` | `spectraquant_v3.core.errors` | V3 hierarchy is a superset |
| `AssetClassLeakError` | Both packages | Both packages | Different signatures |
| `SymbolResolutionError` | Both packages | Both packages | Different signatures |
| `EmptyUniverseError` | Both packages | Both packages | Identical intent |
| `CacheOnlyViolationError` | Both packages | Both packages | Identical intent |
| `ConfigValidationError` | Both packages | Both packages | Identical intent |
| `validate_ohlcv_dataframe` | V2: part of `core/schema.py` validate_* functions | V3: `core/schema.py::validate_ohlcv_dataframe` | V3 version is the canonical one |
| `CryptoSymbolRegistry` | `spectraquant.crypto.symbols.crypto_symbol_registry` | `spectraquant_v3.crypto.symbols.registry` | Same design, different class/attribute names |
| `EquitySymbolRegistry` | `spectraquant.equities.symbols.equity_symbol_registry` | `spectraquant_v3.equities.symbols.registry` | Same design, different class/attribute names |
| Equity momentum agent | `spectraquant.equities.signals.momentum_agent.MomentumAgent` | `spectraquant_v3.equities.signals.momentum.EquityMomentumAgent` | Slightly different scoring; same algorithm |
| Crypto symbol mapper | `spectraquant.crypto.symbols.crypto_symbol_mapper` | `spectraquant_v3.crypto.symbols.mapper` | Same design |
| Equity symbol mapper | `spectraquant.equities.symbols.equity_symbol_mapper` | `spectraquant_v3.equities.symbols.mapper` | Same design |

### 5.2 V2-only functionality (not yet ported to V3)

| V2 module | Function | Migration priority |
|---|---|---|
| `equities/signals/` (6 of 7 agents) | mean_reversion, breakout, volatility, regime, quality, news_sentiment | High |
| `models/train.py`, `models/predict.py` | ML model training and inference | High |
| `dataset/builder.py` | Feature dataset assembly for ML | High |
| `mlops/auto_retrain.py` | Walk-forward retrain scheduling | Medium |
| `alpha/meta_policy.py` | Performance-weighted expert blending | Medium |
| `meta_policy/arbiter.py` | Regime-aware expert selection | Medium |
| `qa/quality_gates.py` (8 functions) | Prediction / signal / portfolio QA | Medium |
| `news/` (9 modules) | News embedding, entity linking, impact scoring | Medium |
| `intelligence/` (20+ modules) | Scheduling, emailing, intraday, learning | Low (live-only) |
| `nlp/` | Multitask trainer, weak supervision | Low |
| `onchain/` | On-chain data collection and features | Low |
| `stress/` | Param sensitivity, regime performance | Low |
| `analysis/` | Feature pruning, model comparison | Low |
| `pricing/` | Price target / downside engine | Low |
| `intraday/` | Intraday learner | Low |
| `sentiment/newsapi_provider.py` | External news sentiment | Low |

### 5.3 V2-internal duplicates (within `spectraquant` alone)

| Concept | Location 1 | Location 2 | Location 3 |
|---|---|---|---|
| Regime detection | `core/regime.py::compute_regime` | `meta_policy/regime.py::detect_regime` | `alpha/meta_policy.py::detect_regime` |
| Expert weighting / blending | `alpha/meta_policy.py::compute_expert_weights` | `meta_policy/arbiter.py` (full class) | — |
| Agent concept | `alpha/experts/` (DataFrame-in) | `agents/agents/` (dict-in) | `experts/` (standalone aggregator) |
| Portfolio allocation | `portfolio/allocator.py::allocate_risk_parity` | `portfolio/allocator.py::allocate_vol_target` | `equities/policy/allocator.py::EquityAllocator` |
| Quality gates | `qa/quality_gates.py` | `qa/pipeline_check.py` | `qa/output_check.py` |

### 5.4 `trading_assistant/` overlap

The `trading_assistant/` directory (still present at repo root) replicates:
- `app/ingest/market.py` ← overlaps with `spectraquant/data/`
- `app/ingest/news.py` ← overlaps with `spectraquant/news/`
- `app/models/` ← overlaps with `spectraquant/models/`
- `app/risk/` ← overlaps with `spectraquant/portfolio/risk.py`
- `app/policy/` ← overlaps with `spectraquant/core/policy.py`
- `app/scheduler.py` ← overlaps with `spectraquant/intelligence/scheduler.py`

The archive README marks this module as inactive; however, the live copy
(`trading_assistant/`) is still tracked in git and not excluded from the active
package.  `trading_assistant_runner.py` still exists at the repo root.

---

## 6. Safest Staged Refactor Plan

**Guiding principles:**
- V2 tests must remain green at every stage.
- V3 tests must remain green at every stage.
- No cross-package imports (V2 → V3 or V3 → V2) are ever introduced.
- Each stage produces a shippable, green state.
- Staging is ordered from lowest risk (no behavioral change) to highest risk (behavioral migration).

---

### Stage 0 — Clean up the physical layout (no code changes)

**Risk: Zero.** These are file moves and `.gitignore` updates.

1. **Move `trading_assistant/` to `archive/trading_assistant/`** and delete `trading_assistant_runner.py` from root.  Update `archive/trading_assistant/README.md` accordingly.
2. **Move `perplexity_research_agent.py` and `test_perplexity.py`** to `scripts/` or `archive/` respectively — they are not part of either package and pollute the root.
3. **Verify `requirements-v3.txt`** is accurate and remove duplicates with `requirements.txt`.
4. **Add `reports/`, `logs/`, `models/` to `.gitignore`** if not already covered.

---

### Stage 1 — Stabilise V3 core contracts (no V2 changes)

**Risk: Low.** Additive changes to V3 only.

1. **Merge `RunStage` completeness check.**  
   `V3 RunStage` enum (UNIVERSE → REPORTING) should match actual pipeline stages.  Add `INGESTION` if missing.

2. **Add `DEGRADED` to `SignalStatus` in V3** (exists in V2; needed for graceful degradation).

3. **Add `NoSignalReason` enum to V3 `core/enums.py`** — mirrors V2 usage; allows structured reason codes.

4. **Document `validate_ohlcv_dataframe` as the canonical OHLCV guard** in V3 docs; ensure all V3 ingestors call it.

5. **Audit and fix the `strategy_portfolio/portfolio.py` mixed-asset-class gap:**  
   `StrategyPortfolio` should validate that all `strategy_ids` share the same `asset_class` or explicitly document that cross-asset portfolios are intentionally allowed.

---

### Stage 2 — Port missing V2 equity agents to V3 (V3-only additions)

**Risk: Low–Medium.** New V3 modules only; V2 untouched.

Port these six V2 equity agents to `src/spectraquant_v3/equities/signals/`:

| New V3 module | Source |
|---|---|
| `mean_reversion.py` | `spectraquant/equities/signals/mean_reversion_agent.py` |
| `breakout.py` | `spectraquant/equities/signals/breakout_agent.py` |
| `volatility.py` | `spectraquant/equities/signals/volatility_agent.py` |
| `regime.py` | `spectraquant/equities/signals/regime_agent.py` |
| `quality.py` | `spectraquant/equities/signals/quality_agent.py` |
| `news_sentiment.py` | `spectraquant/equities/signals/news_sentiment_agent.py` |

Each V3 agent must:
- Accept `(symbol: str, frame: pd.DataFrame, as_of: str)` and return `SignalRow`.
- Call `validate_ohlcv_dataframe` before processing.
- Emit `SignalStatus.NO_SIGNAL` (not raise) when data is insufficient.

Register new agents in `strategies/agents/registry.py`.  
Add matching `StrategyDefinition` entries in `strategies/registry.py`.

---

### Stage 3 — Consolidate regime detection in V2 (V2-only cleanup)

**Risk: Medium.** Touches multiple V2 modules; V3 unchanged.

1. **Designate `spectraquant.meta_policy.regime` as the canonical V2 regime module.**  
2. **Deprecate `spectraquant.core.regime.compute_regime`** — replace all callers with `detect_regime` from `meta_policy.regime`.  
3. **Deprecate `spectraquant.alpha.meta_policy.detect_regime`** — replace with a call to the canonical module.  
4. **Remove the now-dead `alpha/meta_policy.detect_regime` definition** (leave `blend_signals` and `compute_expert_weights`).

This reduces three regime implementations to one.  Each replacement is a one-line import change.

---

### Stage 4 — Port V2 ML model layer to V3 (V3-only additions)

**Risk: Medium.** New V3 modules; no V2 changes.

Create:
- `src/spectraquant_v3/research/dataset_builder.py` — already started; extend with feature assembly from OHLCV.
- `src/spectraquant_v3/models/` — port `train.py`, `predict.py`, `ensemble.py` from V2.
- `src/spectraquant_v3/mlops/auto_retrain.py` — port from V2.

These should consume `SignalRow` and `AllocationRow` as inputs, emit typed `ModelOutput` dataclasses, and integrate with `FeatureStore` for artifact persistence.

---

### Stage 5 — Port V2 QA gates to V3 (V3-only additions)

**Risk: Low–Medium.** New V3 modules only.

Create `src/spectraquant_v3/qa/` to host ported V2 quality gate functions.  Each function should:
- Accept typed V3 dataclasses rather than raw DataFrames.
- Integrate with `ctx.qa_matrix` and `ctx.manifest`.
- Replace the `cli/main.py` inline QA calls in V2.

This stage does not remove anything from V2; it only adds equivalent coverage to V3.

---

### Stage 6 — Consolidate enum/error types (breaking: requires coordination)

**Risk: High.** This stage changes the public API of both packages.

**Do not perform this stage until V3 is feature-complete enough to stand alone.**

1. Consider extracting `spectraquant_shared` (a new, minimal package) for:
   - `AssetClass`, `RunMode`, `SignalStatus` enums
   - `SpectraQuantError` base + `AssetClassLeakError`, `SymbolResolutionError`,
     `EmptyUniverseError`, `CacheOnlyViolationError`
   
2. Both `spectraquant` and `spectraquant_v3` import from `spectraquant_shared`.

3. **Prerequisites:** Full test coverage of both packages; explicit deprecation notices; semantic versioning bump.

**Alternative (lower risk):** Keep separate error hierarchies forever and accept the duplication as an isolation boundary between the two generations.

---

### Stage 7 — Deprecate V2 CLI and freeze V2 (long-term)

**Risk: High.** Changes user-facing behavior.

1. Mark `spectraquant` (V2) as `deprecated` in `pyproject.toml` description.
2. Add a `DeprecationWarning` in `spectraquant.cli.main:main` pointing to `sqv3`.
3. Freeze V2: no new features, security fixes only.
4. Once `sqv3` covers all V2 functionality, remove the `spectraquant` entry point.

---

### Summary table

| Stage | Scope | Risk | V2 affected | V3 affected |
|---|---|---|---|---|
| 0 | File/directory cleanup | Zero | No | No |
| 1 | V3 core contract hardening | Low | No | Yes (additive) |
| 2 | Port 6 equity agents to V3 | Low–Med | No | Yes (additive) |
| 3 | Consolidate regime in V2 | Medium | Yes (internal) | No |
| 4 | Port ML model layer to V3 | Medium | No | Yes (additive) |
| 5 | Port QA gates to V3 | Low–Med | No | Yes (additive) |
| 6 | Shared enum/error package | High | Yes (imports) | Yes (imports) |
| 7 | Deprecate V2 CLI | High | Yes (UX) | No |

---

*End of Architecture Review*
