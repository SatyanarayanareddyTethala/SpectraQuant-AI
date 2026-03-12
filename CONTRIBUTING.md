# Contributing to SpectraQuant-AI

Thank you for contributing to SpectraQuant-AI.  This document covers coding
conventions, where new work should go, and how to run tests.

---

## Where to Put New Work

| Work type | Location |
|---|---|
| V3 features, strategies, experiments | `src/spectraquant_v3/` |
| V2 bug fixes / maintenance | `src/spectraquant/` |
| New V3 tests | `tests/v3/` |
| New V2 tests | `tests/` |
| Architecture docs | `docs/architecture/` |
| Utility scripts | `scripts/` |

**Do not add new V2 modules** unless fixing an existing V2 bug.  New systematic
research work belongs in V3.

---

## How to Add a V3 Signal

1. Choose the correct asset-class directory:
   - Equity signals → `src/spectraquant_v3/equities/signals/`
   - Crypto signals → `src/spectraquant_v3/crypto/signals/`
2. Create a new module (e.g. `my_signal.py`) that returns a typed signal DataFrame.
3. Register the signal in the relevant `__init__.py` if exported.
4. Write tests in `tests/v3/` covering at least the happy path and edge cases.
5. Do not mix equity and crypto symbols in a single signal module.

---

## How to Add a V3 Experiment

1. Add experiment parameters to `src/spectraquant_v3/experiments/hybrid_params.py`
   or create a new params class there.
2. Register the experiment in `src/spectraquant_v3/experiments/experiment_manager.py`.
3. Wire up any required strategy agents in `src/spectraquant_v3/strategies/agents/`.
4. Add a test under `tests/v3/` that verifies the experiment runs end-to-end
   in `test` mode (no network calls).

---

## Coding Style

- **Python 3.10+** with type annotations on all public functions and methods.
- Use `dataclasses` for typed value objects; avoid plain dicts for structured data.
- ISO-8601 timestamps: normalize trailing `Z` to `+00:00` before calling
  `datetime.fromisoformat()` (see `market_selector.py` for the pattern).
- No silent fallbacks: raise a typed exception from `spectraquant_v3/core/errors.py`
  instead of returning empty data or `None`.
- Use `@property` accessors for backward-compatible renames, not in-place
  renaming of existing attributes.

---

## Hard Invariants (must always hold)

- Every persisted DataFrame contains an explicit `date` column.
- `date` is UTC tz-aware.
- No index-based time semantics.
- `1970-01-01` is a hard failure (epoch sentinel).
- Prediction dates equal latest price timestamps per ticker.
- Scores are continuous rankings in `[0, 100]`.
- Signals are snapshot-based.

---

## Prohibited Patterns

- Default or embedded tickers.
- Implicit use of DataFrame index as time.
- Silent fallbacks for missing dates or epoch values.
- Writing artifacts without schema validation.
- Mixing crypto and equity symbols in a single V3 run.

---

## Running Tests

```bash
# Full test suite
pytest -q

# V3 tests only
pytest tests/v3/ -q

# Single file
pytest tests/v3/test_v3_market_selector.py -v

# Compile check (no imports required)
python -m compileall src
```

CI runs on every push via `.github/workflows/tests.yml`.  All tests must pass
before a PR can be merged.

---

## PR Checklist

- [ ] New V3 code goes in `src/spectraquant_v3/`
- [ ] Schema validators updated if a new artifact type is added
- [ ] Explicit UTC `date` column enforced on all persisted DataFrames
- [ ] No default tickers or embedded universes
- [ ] Tests added/updated in `tests/v3/` (or `tests/` for V2)
- [ ] `python -m py_compile` passes for all touched files
- [ ] `pytest -q` passes locally

---

## Merge Conflict Resolution (providers and universes)

When resolving conflicts involving:
- `src/spectraquant/universe/`
- `scripts/download_universe.py`
- `src/spectraquant/core/providers/*`
- intraday fetch scripts under `scripts/`

Always prefer the version that:
- Uses provider abstractions (not direct `yfinance` calls).
- Preserves schema and invariant checks.
- Avoids hardcoded tickers or sample universes.
- Maintains deterministic behavior.

Reject any side that introduces:
- Embedded tickers or fallback universes.
- Implicit time index usage.
- Silent cleaning or silent data drops.

