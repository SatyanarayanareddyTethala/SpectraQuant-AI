# SpectraQuant-AI

**Systematic trading research platform** supporting equities and crypto with
event-driven + technical hybrid strategies.

| CLI | Package | Status |
|---|---|---|
| `spectraquant` | `spectraquant` (V2) | Stable — equities-focused |
| `sqv3` | `spectraquant_v3` (V3) | Active development — crypto + equities, modern typed architecture |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Capabilities](#2-key-capabilities)
3. [Architecture Overview](#3-architecture-overview)
4. [Repository Structure](#4-repository-structure)
5. [Development Setup](#5-development-setup)
6. [Example Workflow](#6-example-workflow)
7. [CLI Commands](#7-cli-commands)
8. [Testing](#8-testing)
9. [Current Research Areas](#9-current-research-areas)
10. [Roadmap](#10-roadmap)
11. [ML Predictive Analytics Module](#11-ml-predictive-analytics-module)
12. [Documentation](#12-documentation)
13. [Contributing](#13-contributing)

---

## 1. Project Overview

SpectraQuant-AI is a research platform for systematic trading strategy
development across **equities** and **crypto** markets.  The platform is built
around the principle that market opportunity is primarily driven by **events**
and **news catalysts**, layered on top of standard technical signals.

The codebase contains two generations:

- **V2** (`src/spectraquant/`) — production-quality equities pipeline, stable
  CLI, news-aware candidate selection, signal generation, and portfolio
  assembly.
- **V3** (`src/spectraquant_v3/`) — modern typed architecture, strict
  asset-class segregation, experiment infrastructure, hybrid strategy research,
  and a news intelligence layer with a deterministic market selector.

---

## 2. Key Capabilities

| Capability | Module |
|---|---|
| News intelligence abstraction | `spectraquant_v3/core/news_schema.py`, `news_intel_store.py` |
| Perplexity catalyst discovery | `spectraquant_v3/core/providers/perplexity_provider.py` |
| Deterministic market selector | `spectraquant_v3/intelligence/market_selector.py` |
| Hybrid strategy experiments | `spectraquant_v3/experiments/` |
| Experiment manager | `spectraquant_v3/experiments/experiment_manager.py` |
| Feature store | `spectraquant_v3/feature_store/` |
| Equity signal agents | `spectraquant_v3/equities/signals/` |
| Crypto pipeline | `spectraquant_v3/crypto/` |
| Backtesting framework | `spectraquant_v3/backtest/` |
| V2 equities pipeline (stable) | `spectraquant/` |

---

## 3. Architecture Overview

### V2 vs V3

**V2** (`src/spectraquant/`)

- Stable equities-only research pipeline
- NewsAPI-based candidate selection
- Feature building, model training, signal export
- Portfolio construction and governance
- Mature CLI (`spectraquant`)

**V3** (`src/spectraquant_v3/`)

- Modern typed architecture with strict runtime segregation
- Equities **and** crypto pipelines with no symbol cross-contamination
- News intelligence layer with provider abstraction (e.g. Perplexity)
- Deterministic market selector routing decisions to equity / crypto / mixed
- Hybrid strategy research combining technical + news signals
- Experiment manager for systematic strategy evaluation
- Feature store for reusable signal infrastructure
- Modern CLI (`sqv3`)

### V3 Key Rules

1. **Strict runtime segregation** — crypto and equity runs must never mix symbols.
2. **No silent failures** — empty DataFrames, unresolved symbols, and cache
   misses in test mode all raise explicit typed exceptions.
3. **Three run modes** — `normal` (cache-first), `test` (cache-only, CI-safe),
   `refresh` (force redownload).
4. **Every run writes a manifest** — including aborted runs.
5. **QA matrix** — one row per symbol per run.

### V3 Custom Exceptions

| Exception | When raised |
|---|---|
| `MixedAssetClassRunError` | Crypto + equity symbols in the same run |
| `AssetClassLeakError` | Wrong asset class detected in pipeline |
| `SymbolResolutionError` | Symbol not found in registry |
| `EmptyUniverseError` | Universe resolves to zero symbols |
| `EmptyPriceDataError` | Empty price series returned as result |
| `CacheOnlyViolationError` | Network call attempted in test mode |
| `CacheCorruptionError` | Cached parquet fails schema validation |

---

## 4. Repository Structure

```
SpectraQuant-AI/
├── src/
│   ├── spectraquant/          # V2: stable equities pipeline
│   │   ├── agents/            #   Signal agents
│   │   ├── alpha/             #   Alpha generation
│   │   ├── cli/               #   `spectraquant` CLI
│   │   ├── crypto/            #   V2 crypto modules
│   │   ├── equities/          #   V2 equity modules
│   │   ├── features/          #   Feature engineering
│   │   ├── intelligence/      #   V2 news intelligence
│   │   └── ...
│   └── spectraquant_v3/       # V3: modern typed architecture
│       ├── core/              #   Shared types, errors, schema, QA, cache
│       │   └── providers/     #   News provider abstraction (Perplexity, …)
│       ├── intelligence/      #   Market selector
│       ├── equities/          #   V3 equity pipeline
│       ├── crypto/            #   V3 crypto pipeline
│       ├── pipeline/          #   Orchestrators
│       ├── strategies/        #   Strategy definitions, agents, allocators
│       ├── backtest/          #   Backtesting engine
│       ├── experiments/       #   Experiment manager, hybrid params
│       ├── feature_store/     #   Feature store
│       ├── research/          #   Dataset builder
│       └── cli/               #   `sqv3` CLI
│           └── commands/
├── tests/                     # V2 test suite
│   └── v3/                    # V3 test suite
├── config/
│   ├── base.yaml              # V2 shared config
│   ├── equities.yaml
│   ├── crypto.yaml
│   └── v3/                    # V3 config files
│       ├── base.yaml
│       ├── equities.yaml
│       ├── crypto.yaml
│       ├── news.yaml
│       ├── providers.yaml
│       ├── risk.yaml
│       └── strategies.yaml
├── docs/
│   ├── architecture/          # Architecture docs (market selector, …)
│   ├── design/                # Design docs and ADRs
│   ├── howto/                 # Getting started, installation, quick reference
│   └── implementation/        # Implementation reports
├── scripts/                   # Utility scripts (download, research, diagnostics)
├── dashboard/                 # Streamlit dashboard
├── data/
│   └── universe/              # Universe CSV files (NSE, LSE, Nifty 50, …)
├── archive/                   # Archived / legacy code (not imported)
├── alembic/                   # Database migrations
├── pyproject.toml
├── requirements.txt
├── requirements-v3.txt
└── CONTRIBUTING.md
```

---

## 5. Development Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/SatyanarayanareddyTethala/SpectraQuant-AI.git
cd SpectraQuant-AI

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt    # V2 core deps
pip install -r requirements-v3.txt # V3 additional deps

# Editable install (exposes both CLI entry points)
pip install -e .
```

Windows users may alternatively run `install.bat`; macOS/Linux can use
`install.sh`.

**Minimum configuration:**

1. Copy `.env.example` to `.env` and fill in secrets:
   - `NEWSAPI_KEY` — required for V2 `news-scan`
   - `PERPLEXITY_API_KEY` — required for V3 news intelligence
   - `SPECTRAQUANT_UNIVERSE` — optional override for default universe CSV
2. Adjust `config.yaml` (V2) or `config/v3/` (V3) as needed.

**V3 config override at runtime:**

```bash
SPECTRAQUANT_V3_CONFIG_DIR=/path/to/my/config sqv3 crypto run
# or
sqv3 crypto run --config-dir /path/to/my/config
```

---

## 6. Example Workflow

A typical research loop using V3:

```
1. Ingest prices
   sqv3 equity run --mode normal
   sqv3 crypto run --mode normal

2. Fetch news intelligence
   sqv3 research run --news-scan

3. Generate signals
   sqv3 equity signals
   sqv3 crypto signals

4. Run hybrid strategies
   sqv3 strategy run --strategy momentum_news_hybrid

5. Run experiments
   sqv3 experiment run --config config/v3/strategies.yaml

6. Evaluate results
   sqv3 experiment results --latest

7. Deploy or paper-trade strategy
   sqv3 strategy portfolio --strategy momentum_news_hybrid
```

---

## 7. CLI Commands

### V2 (`spectraquant`)

```bash
spectraquant --help

# Health check — validates config, data folders, and dependencies
spectraquant doctor

# Download price data
spectraquant download

# News candidate scan
spectraquant news-scan

# End-to-end refresh (download → features → train → predict)
spectraquant refresh

# Signal export
spectraquant signals

# Portfolio build
spectraquant portfolio

# --- ML Predictive Analytics (new) ---
# Train Random Forest + XGBoost classifiers with walk-forward validation
spectraquant train-ml --ticker AAPL

# Generate ensemble ML signals and print last 10 rows
spectraquant predict-ml --ticker AAPL --rows 10

# Additional: build-dataset, train, predict, score, eval, universe utilities
```

### V3 (`sqv3`)

```bash
sqv3 --help

# Environment and config check
sqv3 doctor

# Equity pipeline
sqv3 equity run --mode normal      # normal | test | refresh
sqv3 equity signals

# Crypto pipeline
sqv3 crypto run --mode normal
sqv3 crypto signals

# Research (news intelligence + dataset building)
sqv3 research run

# Strategy and experiments
sqv3 strategy run --strategy <name>
sqv3 strategy portfolio --strategy <name>
sqv3 experiment run
sqv3 experiment results

# Backtesting
sqv3 backtest run --strategy <name>

# Feature store
sqv3 feature-store list
sqv3 feature-store build
```

### V3 Run Modes

| Mode | Behaviour | Network | Use case |
|---|---|---|---|
| `normal` | Cache-first, download missing | Yes | Day-to-day research |
| `test` | Cache-only | **No** (raises `CacheOnlyViolationError`) | CI, reproducibility |
| `refresh` | Force redownload | Yes | Stale-data recovery |

---

## 8. Testing

```bash
# Compile check
python -m compileall src

# Full V2 test suite
pytest -q

# V3 tests only
pytest tests/v3/ -q

# Single test file
pytest tests/v3/test_v3_market_selector.py -v

# Health check
spectraquant --help
sqv3 --help
```

Tests are located in:

- `tests/` — V2 tests (universe, signals, pipeline, governance, …)
- `tests/v3/` — V3 tests (market selector, hybrid strategies, backtest,
  experiments, ingestion, …)

CI runs on every push via `.github/workflows/tests.yml`.

---

## 9. Current Research Areas

- **News-driven strategies** — using event catalysts to time and size positions
  in equities and crypto
- **Cross-asset opportunity selection** — deterministic routing between equity
  and crypto based on news intelligence
- **Hybrid signals** — combining technical momentum with news sentiment and
  impact scores
- **Event intelligence** — improving the Perplexity catalyst provider with
  structured event ontologies

---

## 10. Roadmap

- News-first routing integrated into live strategy runner
- Improved catalyst scoring with structured ontology
- Automated experiment scheduling and result ranking
- Feature store expansion (on-chain metrics, macro factors)
- Provider abstraction extended to additional news vendors

See [CHANGELOG.md](CHANGELOG.md) for completed work.

---

## 11. ML Predictive Analytics Module

### Overview

`src/spectraquant/ml/` is a production-quality supervised classification layer
that sits cleanly on top of the existing SpectraQuant OHLCV and signal
infrastructure.

| Module | Purpose |
|---|---|
| `features.py` | 13-column ML feature set (returns, volatility, SMA ratios, RSI-14, MACD, sentiment) |
| `targets.py` | Configurable forward-return binary classification targets |
| `models.py` | Random Forest + optional XGBoost factories with fixed seeds |
| `walk_forward.py` | Leakage-free expanding-window walk-forward validation |
| `importance.py` | Feature importance extraction → `reports/ml/importance/` |
| `forecast.py` | Optional SARIMAX directional probability signal |
| `ensemble.py` | Weighted ensemble of RF + XGB + SARIMAX → signal score in {-1, 0, 1} |
| `pipeline.py` | End-to-end orchestrator, writes signals to `reports/ml/signals/` |

### Quick start

```bash
# Download data first
spectraquant download --ticker AAPL

# Train models and run walk-forward validation
spectraquant train-ml --ticker AAPL

# Generate and display the latest ensemble signals
spectraquant predict-ml --ticker AAPL --rows 10
```

Output files are written to:

```
reports/ml/
├── signals/       ml_signals_<timestamp>.csv
├── importance/    feature_importance_rf_<timestamp>.csv
│                  feature_importance_xgb_<timestamp>.csv
└── eval/          rf_folds_<timestamp>.csv
                   xgb_folds_<timestamp>.csv
```

### Configuration (`config/base.yaml`)

```yaml
ml:
  enabled: true
  horizon: 1          # forward-return look-ahead in bars
  train_size: 252     # rows per training window (≈ 1 year of daily bars)
  test_size: 21       # rows per test window (≈ 1 month)
  step_size: 21       # walk-forward step
  threshold: 0.55     # buy/sell probability threshold
  use_xgboost: true   # requires: pip install xgboost
  use_sarimax: false  # requires: pip install statsmodels
  weights:
    rf: 0.4           # Random Forest weight in ensemble
    xgb: 0.4          # XGBoost weight in ensemble
    ts: 0.2           # SARIMAX weight in ensemble
```

### Dependencies

| Dependency | Required | Install |
|---|---|---|
| `scikit-learn` | Yes (core dep) | included |
| `xgboost` | Optional | `pip install xgboost` or `pip install 'spectraquant[ml]'` |
| `statsmodels` | Optional | `pip install statsmodels` or `pip install 'spectraquant[opt]'` |

### Signal output format

`ensemble_to_signal` maps the blended probability to:

| probability | signal_score | meaning |
|---|---|---|
| > threshold | +1 | buy |
| < 1 − threshold | −1 | sell |
| otherwise | 0 | hold |

The `signal_score` column is compatible with the existing `AgentOutput.signal_score`
convention so ML signals can be merged with rule-based signals via
`spectraquant.models.ensemble.compute_ensemble_scores`.

### Anti-leakage principles

- Data is **never shuffled** — splits are purely positional/temporal.
- Each fold's training window ends strictly before its test window begins.
- `add_target` uses `shift(-horizon)` so the label always refers to a
  strictly future observation.
- The pipeline validates the DatetimeIndex and rejects pre-2000 timestamps
  (guards against the epoch-millisecond regression bug).

---

## 12. Documentation

| Location | Contents |
|---|---|
| `docs/architecture/` | Component architecture (market selector, …) |
| `docs/design/` | Design docs, ADRs, V3 audit notes |
| `docs/howto/` | Getting started, installation, quick reference |
| `docs/implementation/` | Implementation reports and deliverables |
| [DOCS_INDEX.md](DOCS_INDEX.md) | Full documentation index |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

---

## 13. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding guidelines, PR checklist,
and V3 development conventions.
