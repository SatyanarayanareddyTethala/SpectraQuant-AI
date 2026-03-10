# SpectraQuant-AI-V2 / V3

This repository contains two generations of the SpectraQuant systematic research platform:

| CLI entry point | Package | Status |
|---|---|---|
| `spectraquant` | `spectraquant` (V2) | Production — equities-focused |
| `sqv3` | `spectraquant_v3` (V3) | **Scaffold** — crypto + equities, strict segregation |

---

## SpectraQuant-AI-V3 (New)

SpectraQuant-AI-V3 is a production-grade systematic research and trading platform that supports **both crypto and equity markets with strict runtime segregation**.  The V3 architecture enforces that no crypto symbols appear in equity runs and no equity symbols appear in crypto runs.

### V3 Architecture

```
src/spectraquant_v3/
├── core/          # Shared abstractions only (enums, errors, config, cache, manifest, schema, QA)
├── crypto/        # Crypto pipeline (CCXT / Binance / CoinGecko / Glassnode)
│   ├── ingestion/
│   ├── signals/
│   ├── universe/
│   ├── symbols/
│   └── features/
├── equities/      # Equity pipeline (yfinance / provider abstraction)
│   ├── ingestion/
│   ├── signals/
│   ├── universe/
│   ├── symbols/
│   └── features/
├── pipeline/      # Orchestrators: crypto_pipeline.py, equity_pipeline.py
└── cli/           # Typer CLI (sqv3)
    └── commands/
        ├── crypto.py
        └── equities.py

config/v3/
├── base.yaml      # Shared defaults
├── crypto.yaml    # Crypto-specific config
└── equities.yaml  # Equity-specific config
```

### V3 Key Architecture Rules

1. **Strict runtime segregation**: `sqv3 crypto` and `sqv3 equity` must never be combined.
2. **No silent failures**: empty DataFrames, unresolved symbols, and cache misses in test-mode all raise explicit exceptions.
3. **Three run modes**: `normal` (cache-first), `test` (cache-only, CI-safe), `refresh` (force redownload).
4. **Every run writes a manifest** — including aborted runs.
5. **QA matrix** with one row per symbol per run.

### V3 Quick Start

```bash
# Install (editable)
pip install -e .

# Show help
sqv3 --help

# Check environment and config
sqv3 doctor

# Run crypto pipeline (scaffold stub — prints status, not yet implemented)
sqv3 crypto run --mode normal

# Run equity pipeline (scaffold stub)
sqv3 equity run --mode normal
```

### V3 Run Modes

| Mode | Cache behaviour | Network allowed | Use case |
|---|---|---|---|
| `normal` | Cache-first, download missing | Yes | Day-to-day research |
| `test` | Cache-only | **No** — raises `CacheOnlyViolationError` | CI, reproducibility |
| `refresh` | Force redownload | Yes | Stale-data recovery |

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

### V3 Config Files

```bash
config/v3/base.yaml      # Shared defaults (run mode, cache, QA thresholds, execution, portfolio)
config/v3/crypto.yaml    # Crypto universe, exchanges, quality gates, reports
config/v3/equities.yaml  # Equity universe, provider, quality gates, reports
```

Override the config directory at runtime:
```bash
SPECTRAQUANT_V3_CONFIG_DIR=/path/to/my/config sqv3 crypto run
# or
sqv3 crypto run --config-dir /path/to/my/config
```

---

## SpectraQuant-AI-V2 (Existing)

SpectraQuant-AI-V2 is a research-first equities pipeline for signal generation, portfolio simulation, and governance. The `spectraquant` CLI orchestrates data download, feature building, model training, news-aware candidate selection, signal generation, and portfolio assembly for supported universes (NSE/LSE by default).

## Quickstart (macOS/Linux/Windows)
```bash
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI-V2.git
cd SpectraQuant-AI-V2
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
# optional: editable install for CLI entrypoint
pip install -e .
```
Windows users may alternatively run `install.bat`; macOS/Linux can use `install.sh`.

## Minimum configuration
1. Copy `config.yaml` (already present) and adjust:
   - `universe.default`: choose a CSV under `data/universe/` (e.g., `nifty_50.csv`, `lse_all.csv`).
   - `news_api.key`: set your NewsAPI key.
2. Provide secrets via environment (or `.env.example` as a template):
   - `NEWSAPI_KEY` (required for `news-scan`)
   - optional: `SPECTRAQUANT_UNIVERSE` to override the default universe CSV name.
3. Ensure data directory access: `data/universe/` must stay tracked; runtime outputs land in `reports/`, `logs/`, `models/`.

## Primary workflows (CLI)
All commands run from the repo root after activating the venv:
```bash
spectraquant --help
```

Common flows:
- **Health check**: `spectraquant doctor` — validates config, data folders, and dependencies.
- **News candidate scan**: `spectraquant news-scan` — fetches news, writes candidates to `reports/news/`.
- **Data refresh**: `spectraquant refresh` — end-to-end download, feature build, dataset creation, training, and predictions.
- **Signal export**: `spectraquant signals` — emits signal CSVs under `reports/signals/`.
- **Portfolio build**: `spectraquant portfolio` — constructs point-in-time portfolios using latest signals/prices.

See `spectraquant --help` for additional commands such as `download`, `build-dataset`, `train`, `predict`, `score`, `eval`, and universe utilities.

## Troubleshooting
- **NewsAPI 401**: Confirm `NEWSAPI_KEY` is set in the environment and matches `config.yaml` if specified.
- **Empty universe**: Verify the chosen CSV exists under `data/universe/` and `SPECTRAQUANT_UNIVERSE` points to a valid file name without path changes.
- **Too few eligible tickers after filtering**: Relax QA thresholds in `config.yaml` (`qa.min_price_rows`, `qa.min_non_null_ratio`) or select a broader universe CSV.
- **Path import/shadowing issues**: Avoid local files named `spectraquant.py` or `path.py`; ensure you run inside the project venv and that `src/` is discoverable (editable install recommended).

## Testing and validation
Run a lightweight validation before changes:
```bash
python -m compileall src
pytest -q
spectraquant --help
```

## Documentation
- Quick reference and guides live under `docs/howto/`.
- Architecture and roadmap: `docs/design/`.
- Implementation reports and deliverables: `docs/implementation/`.
- Full index: [DOCS_INDEX.md](DOCS_INDEX.md).

## License and contributions
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [CHANGELOG.md](CHANGELOG.md) for release notes.
