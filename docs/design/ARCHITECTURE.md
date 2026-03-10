# SpectraQuant-Next v2 Architecture

## Layer separation

```
src/spectraquant/
  core/          # invariant, production-safe logic
  research/      # experimental features, factors, models
  execution/     # paper trading, execution simulation
```

Rules:
- `core/` must not import from `research/`.
- `research/` may depend on `core/`.
- `execution/` consumes validated artifacts from `core/`.

## Data flow (daily + intraday)
1. **Ingestion**: normalize columns → `ensure_datetime_column` (UTC) → retention → persist.
2. **Features / Alpha**: consume explicit `date`; no NaNs post-cleaning.
3. **Scoring**: continuous 0–100 ranking scores; snapshot outputs only.
4. **Signals**: BUY/HOLD/SELL from ranked snapshot.
5. **Portfolio**: snapshot allocation (Top-K; equal or risk-weighted).
6. **Execution**: paper fills from signals only (no raw predictions).
7. **QA**: schema enforcement, epoch blocking, date alignment checks.

## Invariants
- Every persisted DataFrame includes explicit `date`.
- `date` is UTC tz-aware; epoch values are forbidden.
- No implicit index-based time usage.
- Prediction dates equal latest price timestamps per ticker.
- Scores are continuous rankings in `[0, 100]`.
