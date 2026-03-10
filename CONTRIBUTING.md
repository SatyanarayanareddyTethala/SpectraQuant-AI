# Contributing to SpectraQuant-Next

## Hard invariants (must always hold)
- Every persisted DataFrame contains an explicit `date` column.
- `date` is UTC tz-aware.
- No index-based time semantics.
- 1970-01-01 is a hard failure.
- Prediction dates equal latest price timestamps per ticker.
- Scores are continuous rankings in [0, 100].
- Signals are snapshot-based.

## Prohibited patterns
- Default or embedded tickers.
- Implicit use of DataFrame indices as time.
- Silent fallbacks for missing dates or epoch values.
- Writing artifacts without schema validation.

## Merge conflict resolution (providers & universes)
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

## Release note expectations (v1.1.x)
Include a short note stating:
- Improved universe ticker cleaning (NaN/placeholder removal).
- Corrected universe metrics and duplicate reporting.
- No changes to CLI commands, schemas, or downstream artifacts.

## PR checklist
- [ ] Schema validators updated if new artifact is added.
- [ ] Explicit `date` column enforced with UTC tz.
- [ ] No default tickers or embedded universes.
- [ ] Tests added/updated for invariants (`pytest`).
- [ ] `python -m py_compile` passes for touched files.
