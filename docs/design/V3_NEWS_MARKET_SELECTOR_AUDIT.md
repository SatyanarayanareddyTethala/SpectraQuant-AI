# V3 News-First Market Selector — Architecture Audit (Phases 1–6)

**Audit date:** 2026-03-11  
**Scope:** Full repo-wide intelligence archaeology for a news-first market selector  
**Goal:** Determine the best migration path into V3 before writing any new code

---

## 1. Repo-Wide Intelligence Archaeology

This section maps every module that touches intelligence, planning, orchestration,
or news-first routing.  Each entry states what the module does, whether it is
active or legacy, and its relevance to the desired feature.

---

### 1.1  `src/spectraquant/intelligence/` — V2 Intelligence Layer (31 files)

| File | Type | Status | Relevance |
|---|---|---|---|
| `scheduler.py` | Orchestrator | Active | Runs `_run_hourly_news()` and `_run_premarket()` — the closest V2 approximation to a news-first trigger loop |
| `trade_planner.py` | Planner | Active | Generates ranked pre-market trade plans; candidates carry a `news_context` dict; regime-aware scoring with analog calibration |
| `execution_intelligence.py` | Execution | Active | State machine (WAIT→WATCH→EXECUTE→BLOCKED→DONE); blocks execution when `news_context.risk_score` is too high |
| `regime_engine.py` | Analyzer | Active | Classifies market regime (TRENDING/CHOPPY/RISK\_ON/RISK\_OFF/EVENT\_DRIVEN/PANIC) from OHLCV; regime affects trade plan weights |
| `capital_intelligence.py` | Risk | Active | Position/exposure checks; daily loss limit; used as gate before execution |
| `meta_learner.py` | Learner | Active | Auto-tunes signal thresholds using `FailureMemory` statistics |
| `failure_memory.py` | Memory | Active | Tracks failed trades for hypothesis generation |
| `analog_memory.py` | Memory | Active | Historical pattern matching for confidence calibration |
| `hypothesis_engine.py` | Research | Active | Detects market anomalies; generates structured hypotheses |
| `belief_engine.py` | Cognition | Active | Maintains probabilistic market beliefs |
| `explanation_engine.py` | Cognition | Active | Generates human-readable trade rationales |

**Key observations:**
- `scheduler.py` sequences `hourly_news → premarket_plan → intraday_monitor`. This is an implicit
  news-first loop at the *scheduling* level — news runs first, its output is picked up by the
  pre-market planner, and the planner's output drives execution. However, this loop is:
  - Equities-only in practice (India/NSE context)
  - Not typed (no explicit routing decision artifact)
  - Coupled to APScheduler, making it hard to test or replay
- `trade_planner.py` contains the only real cross-signal scoring logic. It scores candidates
  with regime filter + analog calibration + sector caps. The scoring kernel
  (`_score_candidate`) is reusable as a concept.
- **There is no cross-asset selector anywhere in this layer.** No module asks "should we run
  crypto or equities today?" The asset class is always implicit.

---

### 1.2  `src/spectraquant/news/` — V2 News Layer (11 files)

| File | Type | Status | Relevance |
|---|---|---|---|
| `impact_scoring.py` | Scorer | Active | `score_article()`: recency decay × source credibility × sentiment magnitude → `[0,1]` impact score. `build_news_features()`: aggregates to per-symbol DataFrame |
| `event_ontology.py` | Schema | Active | Typed event taxonomy: Earnings, Regulatory, Macro, CorporateAction, OperationsDisruption, Risk, M&A. Each has required/optional slots and auto-fill logic |
| `event_classifier.py` | Classifier | Active | Rule-based NLP → (event\_type, sentiment, magnitude, confidence) without LLM dependency |
| `entity_linker.py` | Linker | Active | Maps text mentions to market symbols |
| `schema.py` | Schema | Active | `normalize_article()` — canonical article fields; used by the news-first contract tests |
| `universe_builder.py` | Builder | Active | `build_news_universe()` for equities; `score_impact()`, `apply_liquidity_filter()` |
| `collector.py` | Transport | Active | RSS feed collection |
| `dedupe.py` | Utility | Active | Stable hash-based deduplication |
| `embeddings.py` | NLP | Active | Semantic embeddings (optional) |
| `analog_memory.py` | Memory | Active | Historical event pattern matching |

**Key observations:**
- `impact_scoring.py` contains the most reusable scoring primitive. Its three-factor formula
  (recency × credibility × sentiment magnitude) is clean, deterministic, and easy to adapt.
- `event_ontology.py` is the most architecturally valuable module in V2. Its typed event classes
  implicitly carry asset-class information: `EarningsEvent` → equities, `MacroEvent` → both,
  `RegulatoryEvent` (e.g. SEC, CFTC) → could be crypto-specific.
- **Neither module makes a cross-asset routing decision.** They produce per-symbol or
  per-article scores, but no caller aggregates these into "equities has the stronger set today."

---

### 1.3  `src/spectraquant/crypto/universe/news_crypto_universe.py` — Partial News-First

**`build_news_crypto_universe(cfg)`:**
- Collects RSS → dedupes → maps entities → applies exponential recency decay → ranks by
  `mention_count × recency_weight` → returns top-N as `["BTC-USD", "ETH-USD", ...]`
- This is the **only existing news-first selection function in the repo** — but it is
  crypto-only, symbol-selection-only, and returns a list of strings rather than a typed decision.
- Wired into `src/spectraquant/pipeline/crypto_run.py` under `universe_mode == "news_first"`.

There is **no equivalent function for equities.**

---

### 1.4  `src/spectraquant/meta_policy/` — V2 Meta-Policy Layer (4 files)

| File | Type | Status | Relevance |
|---|---|---|---|
| `arbiter.py` | Blender | Active | `run_meta_policy()`: regime detection → performance history → rule-based or perf-weighted expert blending → risk guardrails |
| `regime.py` | Analyzer | Active | `detect_regime()` → `RegimeState(volatility, trend)` using index price data |
| `performance_tracker.py` | Tracker | Active | Tracks expert win rates; computes performance-weighted expert blending weights |

**Key observations:**
- `arbiter.py` has a clean 4-step pipeline: detect regime → load performance → blend signals →
  apply guardrails. This structure is sound and adaptable.
- The regime detection here (`RegimeState` with `volatility` + `trend`) is different from the
  intelligence layer's `regime_engine.py` (which outputs `TRENDING/CHOPPY/RISK_ON/...`). There
  are two separate regime systems in V2.
- `apply_risk_guardrails()` uses drawdown as a circuit breaker. This concept is directly
  applicable to a market selector ("if drawdown > threshold, route to RUN\_NONE").

---

### 1.5  `src/spectraquant/agents/` — V2 Agent Layer (11 files)

| File | Type | Status | Relevance |
|---|---|---|---|
| `arbiter.py` | Blender | Active | Blends per-regime weighted agent signals for crypto. Has `news_catalyst` as a weighted agent |
| `registry.py` | Registry | Active | `AgentSignal(symbol, score, confidence, horizon, rationale_tags)` — clean signal contract |
| `agents/news_catalyst.py` | Agent | Active | Signals from `news_impact_mean × news_sentiment_mean`; confidence from article count |

**Key observations:**
- `AgentSignal` is a clean, typed data structure with `score [-1,1]` and `confidence [0,1]`.
  This pattern is the right model for a routing decision score.
- The V2 Arbiter's regime weights are crypto-only and hardcoded. Not reusable directly.
- `NewsCatalystAgent` computes `score = sentiment × impact` at the per-symbol level. This
  is exactly the right primitive for the market-level aggregation step.

---

### 1.6  `src/spectraquant_v3/core/` — V3 Core Contracts

| File | Type | Status | Relevance |
|---|---|---|---|
| `enums.py` | Enums | Active | `AssetClass(CRYPTO, EQUITY)`, `RunMode`, `RunStage`, `SignalStatus`, `NoSignalReason`, `RunStatus` |
| `news_schema.py` | Schema | Active | `NewsIntelligenceRecord`: provider-agnostic canonical news event (symbol, asset, timestamp, event\_type, sentiment\_score, impact\_score, confidence, rationale). `NewsIntelligenceProvider` protocol |
| `errors.py` | Errors | Active | `MixedAssetClassRunError`, `AssetClassLeakError` |
| `context.py` | Context | Active | `RunContext` — run lifecycle management |

**Key observations:**
- `NewsIntelligenceRecord` is **the best-designed contract in the entire codebase** for this
  feature. It is asset-class-agnostic, provider-agnostic, has clamping validation, and is
  already used in V3 tests. Its `asset` field (`"equity"` / `"crypto"`) is the natural
  split key for a cross-asset scorer.
- `AssetClass` enum is the correct typed token for routing output.
- **`enums.py` is missing a `MarketRoute` enum** — the explicit routing decision type
  (`RUN_EQUITIES`, `RUN_CRYPTO`, `RUN_BOTH`, `RUN_NONE`) that the market selector should produce.

---

### 1.7  `src/spectraquant_v3/pipeline/` — V3 Pipeline Layer (7 files)

| File | Type | Status | Relevance |
|---|---|---|---|
| `_strategy_runner.py` | Dispatcher | Active | Routes to `run_crypto_pipeline()` or `run_equity_pipeline()` based on strategy asset class. Raises `MixedAssetClassRunError` on mismatch |
| `equity_pipeline.py` | Pipeline | Active | Full equity run: universe → features → signals → meta\_policy → allocation → reporting |
| `crypto_pipeline.py` | Pipeline | Active | Full crypto run: same stages |

**Key observations:**
- `_strategy_runner.py` is the correct dispatch point for pipeline routing but currently
  requires a **pre-labeled strategy** (asset class is in the strategy definition).
- There is **no upstream selector** that decides which strategy to load based on current
  news intelligence. This is the exact gap the V3 market selector should fill.
- Both pipelines are well-structured and stable. The news-first market selector should
  sit **above** them, producing a routing decision that is then passed to `run_strategy()`
  or directly to `run_crypto_pipeline()` / `run_equity_pipeline()`.

---

### 1.8  `archive/trading_assistant/` — Legacy Autonomous Runner

`trading_assistant_runner.py` has a `build_plan()` function that sequences:
news-scan → download → predict → signals → portfolio. It explicitly labels this
"SpectraQuant Daily Plan (News-First)". This confirms the intent existed even
in the earliest version of the system — but the implementation was procedural
shell-command orchestration, not typed routing logic.

**Status:** Archive-only. The concept is confirmed; the implementation is not reusable.

---

## 2. Overlap With the Desired News-First Market Selector

The desired feature requires:

1. Consume news/catalyst intelligence
2. Score equities opportunity vs crypto opportunity
3. Produce a typed routing decision (RUN\_EQUITIES / RUN\_CRYPTO / RUN\_BOTH / RUN\_NONE)
4. Route into the correct pipeline(s)

**Closest existing components:**

| Component | Overlap | Gap |
|---|---|---|
| `build_news_crypto_universe()` | News → ranked crypto symbols (step 1+2 for crypto only) | No equity counterpart; returns symbols not a routing decision |
| `news/impact_scoring.py :: score_article()` | Recency × credibility × sentiment scoring (step 2 primitive) | Per-article, not per-asset-class aggregation |
| `news_schema.py :: NewsIntelligenceRecord` | Canonical, typed, asset-class-tagged news event (step 1 contract) | No aggregation/scoring layer above it |
| `_strategy_runner.py` | Routes between pipelines (step 4) | Requires pre-labeled strategy; no news-first input |
| `intelligence/scheduler.py` | Sequences news then plan (implicit step 1→2→3) | Not typed; equities-only; no cross-asset decision |
| `trade_planner.py :: _score_candidate()` | Regime-aware candidate scoring (step 2 for per-symbol) | Per-symbol, not per-asset-class |
| `meta_policy/arbiter.py :: run_meta_policy()` | Regime → blend → guardrails pipeline structure | Signal blending, not asset-class routing |
| `AgentSignal(score, confidence)` | Clean score contract pattern | Per-symbol agent signal, not market-level |

**What is genuinely missing:**
- A function that accepts a `list[NewsIntelligenceRecord]` and produces a
  `MarketSelectorDecision(route, equity_score, crypto_score, rationale, scored_at)`
- A `MarketRoute` enum in V3 enums
- An `intelligence/` namespace in V3

---

## 3. Reuse Classification

### ✅ Reusable now (as a direct import or thin wrapper)

| Module / Component | Why |
|---|---|
| `spectraquant_v3.core.news_schema.NewsIntelligenceRecord` | Already the right input contract; asset-class-agnostic; well-validated |
| `spectraquant_v3.core.enums.AssetClass` | The correct typed token for routing output (needs `MarketRoute` added) |
| `spectraquant_v3.core.errors.MixedAssetClassRunError` | Already the error type for routing violations |
| `spectraquant_v3.pipeline._strategy_runner.run_strategy()` | Already the dispatch point; market selector output feeds into this |

### 🔧 Reusable with adaptation (adapt the concept into V3, don't import directly)

| Module / Component | Adaptation needed |
|---|---|
| `spectraquant.news.impact_scoring.score_article()` | The three-factor formula (recency × credibility × sentiment) is sound. Adapt into a V3-native `_score_news_record()` that works on `NewsIntelligenceRecord` objects instead of raw dicts |
| `spectraquant.news.event_ontology` | The event taxonomy implies asset-class affinity (earnings→equity, listing→crypto, macro→both). Encode this as a small `EVENT_ASSET_AFFINITY` lookup in the new selector |
| `intelligence/regime_engine.py :: classify_regime()` | Regime state should modify routing confidence (e.g., PANIC → RUN\_NONE regardless of scores). Adapt the idea; do not import V2 directly into V3 |
| `agents/registry.py :: AgentSignal` | The `(score, confidence)` contract pattern is right. Use V3-native typed dataclasses instead |
| `news_crypto_universe.py :: build_news_crypto_universe()` | The recency decay + mention count formula is the right ranking primitive. Re-express it as asset-class-aware scoring |
| `meta_policy/arbiter.py :: run_meta_policy()` | The 4-step pipeline structure (detect → load → blend → gate) maps cleanly to the selector's (ingest → score\_equities → score\_crypto → decide) |
| `capital_intelligence.py :: guardrails` | Drawdown / daily-loss circuit-breaker idea: adapt as `regime_veto` in the routing logic |

### 💡 Conceptually useful only (idea is sound; code must not be ported)

| Module / Component | Reason |
|---|---|
| `intelligence/scheduler.py` | The implicit news-first scheduling loop is the right *concept*. V3 should make this explicit with a typed decision artifact instead of procedural job chaining |
| `intelligence/trade_planner.py` | NSE/India-specific field names (`equity_base`, `daily_loss_limit` in INR), no asset-class awareness. The idea of a pre-market plan with regime-aware scoring is right; the implementation is too tied to V2 context |
| `meta_policy/regime.py :: RegimeState` | The (volatility, trend) two-axis regime model is a sound concept. V3 should use its own regime signal (or the existing `regime_engine.py` output) rather than importing this |
| `alpha/scorer.py :: compute_alpha_score()` | Z-score normalization + weighted group aggregation is a good pattern. Can inform the selector's per-asset-class score normalization |
| `trading_assistant/trading_assistant_runner.py :: build_plan()` | Confirms the intended news-first sequence even in the earliest codebase |

### 🗄️ Archive-only / outdated

| Module / Component | Reason |
|---|---|
| `intelligence/hypothesis_engine.py` | Research loop abstraction; not relevant to real-time routing |
| `intelligence/meta_learner.py` | Threshold auto-tuning; useful long-term but not for the routing MVP |
| `intelligence/failure_memory.py` | Failure tracking for learning; not relevant for routing decision |
| `trading_assistant/` (everything except the `build_plan` concept) | Shell-command orchestration; not portable |

### ❌ Do not reuse

| Module / Component | Reason |
|---|---|
| `agents/arbiter.py` regime weights | Hardcoded crypto-only weights (BULL/BEAR/RANGE/HIGH\_VOL); not cross-asset |
| `meta_policy/arbiter.py :: rule_based_selection()` | Returns per-ticker signals via pandas DataFrames; the contract is wrong for a routing decision that has no individual symbols |
| `intelligence/execution_intelligence.py` | Per-trade state machine; no relevance to asset-class routing |
| `intelligence/analog_memory.py` | Historical pattern matching for per-trade confidence; adds complexity without routing value at this stage |

---

## 4. Best Old Intelligence Ideas Worth Preserving

### Idea 1: Recency-Decayed Impact Scoring
**Source:** `news/impact_scoring.py :: score_article()`

```
impact = exp(-λ·age_hours) × credibility × (0.3 + 0.7 × |sentiment|)
```

This formula is the right kernel. It naturally rewards:
- Fresh, breaking news (recency)
- Trusted sources (credibility)
- Strong directional signals (sentiment magnitude)

For the market selector, adapt this to work on `NewsIntelligenceRecord` objects, using
`confidence` as the credibility proxy and `impact_score` (already computed by the provider)
as the pre-scored input — avoiding re-computation and staying provider-agnostic.

### Idea 2: Event-Type Asset Affinity
**Source:** `news/event_ontology.py` — 7 event classes with financial context

Not every news event matters equally to both asset classes:
- Earnings, CorporateAction, M&A → primarily equities
- Macro (rate hike/cut, CPI) → both, but crypto reacts more violently
- Regulatory (SEC enforcement, CFTC) → crypto-specific risk
- Operational disruption → equity-specific

This implicit affinity should be made **explicit** in the selector as a lookup table:

```python
EVENT_ASSET_AFFINITY = {
    "earnings":               {"equity": 1.0, "crypto": 0.05},
    "m_and_a":                {"equity": 0.90, "crypto": 0.10},
    "corporate_action":       {"equity": 0.80, "crypto": 0.05},
    "macro":                  {"equity": 0.60, "crypto": 0.80},
    "regulatory":             {"equity": 0.50, "crypto": 0.85},
    "operations_disruption":  {"equity": 0.70, "crypto": 0.20},
    "risk":                   {"equity": 0.60, "crypto": 0.50},
    "listing":                {"equity": 0.10, "crypto": 0.90},
    "unknown":                {"equity": 0.50, "crypto": 0.50},
}
```

This table is the most valuable conceptual extraction from the V2 news layer.

### Idea 3: Regime as a Routing Veto
**Source:** `intelligence/regime_engine.py :: classify_regime()`

The regime engine already has `PANIC` and `RISK_OFF` states that suppress trading.
In the market selector, these should map directly to route vetoes:
- `PANIC` → `RUN_NONE` (override regardless of news scores)
- `RISK_OFF` → reduce routing confidence; bias toward `RUN_NONE`
- `EVENT_DRIVEN` → boost routing confidence (news is the primary driver)

This makes regime a **first-class input** to the routing decision, not an afterthought.

### Idea 4: The Pre-Market Plan Structure
**Source:** `intelligence/trade_planner.py :: generate_premarket_plan()`

The plan output format is right:
- `as_of` timestamp
- `regime` with label and confidence
- `status` (generated / failed / skipped)
- Written to a timestamped JSON artifact

The V3 routing decision should follow this same pattern — it is a persisted artifact
with a timestamp, a typed outcome, and full rationale for audit purposes.

### Idea 5: Typed Signal Contract
**Source:** `agents/registry.py :: AgentSignal(score, confidence, rationale_tags)`

Every score that flows through the system should carry `(score, confidence)` together.
The market selector should produce per-asset-class scores as `(score: float, confidence: float)`
pairs, not bare floats.

### Idea 6: Graceful Fallback with Fallback Symbols
**Source:** `news_crypto_universe.py :: build_news_crypto_universe()`

The function has a disciplined fallback pattern:
1. Try to collect news
2. If no articles → warn → fallback to configured symbols (no crash)
3. If no symbols extracted → warn → fallback
4. If `strict=True` → raise `RuntimeError`

The market selector should follow the same pattern:
- If no news records → log warning → return `RUN_NONE` with rationale "insufficient_news_data"
- If strict mode → raise error

---

## 5. Best V3 Design for a News-First Market Selector

### 5.1  Target Architecture

```
NewsIntelligenceRecord[]          (from V3 news_schema, any provider)
        │
        ▼
┌──────────────────────────────────────────┐
│  MarketSelector.score(records, regime)   │  ← new V3 module
│                                          │
│  1. Filter records by asset class        │
│  2. Score equity records → equity_score  │
│  3. Score crypto records → crypto_score  │
│  4. Apply regime veto                    │
│  5. Apply routing thresholds             │
│  6. Return MarketSelectorDecision        │
└──────────────────────────────────────────┘
        │
        ▼
MarketSelectorDecision(
    route=MarketRoute.RUN_CRYPTO,
    equity_score=0.21,
    crypto_score=0.74,
    equity_record_count=3,
    crypto_record_count=11,
    regime_label="RISK_ON",
    regime_vetoed=False,
    rationale="crypto: macro rate-cut + 3 regulatory catalysts (avg_impact=0.74); equity: 3 earnings events (avg_impact=0.21)",
    scored_at="2026-03-11T17:00:00+00:00",
    record_count=14,
)
        │
        ▼
run_strategy("crypto_momentum_v1", cfg=crypto_cfg)   (existing V3 dispatcher)
      or
run_strategy("equity_momentum_v1", cfg=equity_cfg)
      or
both / neither
```

### 5.2  New Enum: `MarketRoute`

Add to `src/spectraquant_v3/core/enums.py`:

```python
class MarketRoute(str, Enum):
    """Routing decision produced by the news-first market selector."""
    RUN_EQUITIES = "run_equities"
    RUN_CRYPTO   = "run_crypto"
    RUN_BOTH     = "run_both"
    RUN_NONE     = "run_none"
```

### 5.3  New Dataclass: `MarketSelectorDecision`

```python
@dataclass
class MarketSelectorDecision:
    route: MarketRoute
    equity_score: float          # [0.0, 1.0]
    crypto_score: float          # [0.0, 1.0]
    equity_record_count: int
    crypto_record_count: int
    regime_label: str            # from V2 regime_engine or "unknown"
    regime_vetoed: bool          # True if regime forced RUN_NONE
    rationale: str               # human-readable explanation
    scored_at: str               # ISO-8601 UTC timestamp
    record_count: int            # total NewsIntelligenceRecord inputs
```

### 5.4  New Module: `src/spectraquant_v3/intelligence/market_selector.py`

```
MarketSelector
├── __init__(config: dict)
│       Reads thresholds: min_score_to_run (default 0.35),
│       both_threshold (default 0.60), half_life_hours (default 6.0)
│
├── score(
│       records: list[NewsIntelligenceRecord],
│       regime_label: str = "UNKNOWN",
│   ) -> MarketSelectorDecision
│       1. Separate records by asset ("equity" vs "crypto")
│       2. _score_records(equity_records) → equity_score
│       3. _score_records(crypto_records) → crypto_score
│       4. _apply_regime_veto(regime_label, equity_score, crypto_score)
│       5. _decide(equity_score, crypto_score, vetoed)
│       6. Return MarketSelectorDecision
│
└── _score_records(records: list[NewsIntelligenceRecord]) → float
        Weighted average of:
            recency_weight(record.timestamp) ×
            record.impact_score ×             (already normalized 0-1)
            EVENT_ASSET_AFFINITY[record.event_type][asset_class] ×
            record.confidence
        Normalized to [0.0, 1.0]
```

### 5.5  Scoring Formula

For a set of `N` records belonging to one asset class:

```
w_i = recency(record_i.timestamp) × record_i.impact_score
           × EVENT_ASSET_AFFINITY[record_i.event_type][asset_class]
           × record_i.confidence

score = sum(w_i) / max(N, 1)
score = clip(score, 0.0, 1.0)
```

Where `recency(t) = exp(-λ · age_hours)` with `λ = ln(2) / half_life_hours`.

This formula directly adapts the V2 `impact_scoring.py` kernel, modified to:
- Use V3 `NewsIntelligenceRecord` fields (no raw dicts)
- Apply event-type affinity as a multiplier
- Remain fully deterministic (no randomness, no I/O)

### 5.6  Routing Decision Logic

```
if regime_vetoed:
    route = RUN_NONE

elif equity_score >= both_threshold and crypto_score >= both_threshold:
    route = RUN_BOTH

elif equity_score >= min_score and crypto_score >= min_score:
    if equity_score >= crypto_score * 1.5:
        route = RUN_EQUITIES
    elif crypto_score >= equity_score * 1.5:
        route = RUN_CRYPTO
    else:
        route = RUN_BOTH

elif equity_score >= min_score:
    route = RUN_EQUITIES

elif crypto_score >= min_score:
    route = RUN_CRYPTO

else:
    route = RUN_NONE
```

### 5.7  Regime Veto Rules

```
PANIC       → RUN_NONE (always)
RISK_OFF    → multiply both scores by 0.5 (effectively penalizes routing)
EVENT_DRIVEN → multiply both scores by 1.2 (boost: news is primary driver)
others      → no adjustment
```

### 5.8  Design Constraints

| Constraint | How Enforced |
|---|---|
| Provider-agnostic | Takes `list[NewsIntelligenceRecord]`, never calls any provider directly |
| Asset-class-safe | Only touches `record.asset` field; never imports crypto or equity modules |
| Testable | Pure function with no I/O; deterministic given same input + config |
| V2-safe | Lives entirely in `spectraquant_v3/`; zero imports from `spectraquant/` |
| Typed output | `MarketSelectorDecision` dataclass; `MarketRoute` string enum |
| Explainable | `rationale` field is always populated with score breakdown |

---

## 6. Smallest Safe Implementation Plan

### Step 1 — Add `MarketRoute` to V3 enums (surgical, zero risk)

**File:** `src/spectraquant_v3/core/enums.py`

Add `MarketRoute(str, Enum)` with four values:
`RUN_EQUITIES`, `RUN_CRYPTO`, `RUN_BOTH`, `RUN_NONE`.

No existing code is touched. This is a pure addition.

---

### Step 2 — Create the `intelligence/` namespace in V3

**Files to create:**
- `src/spectraquant_v3/intelligence/__init__.py`
- `src/spectraquant_v3/intelligence/market_selector.py`

`market_selector.py` should contain:
- `EVENT_ASSET_AFFINITY` lookup dict (the key idea from Phase 4, Idea 2)
- `MarketSelectorDecision` dataclass
- `MarketSelector` class with `score()` and `_score_records()` methods

This module imports only from `spectraquant_v3.core.*` — no V2 imports, no
asset-class-specific imports, no I/O.

---

### Step 3 — Add tests (no implementation is trusted without tests)

**File:** `tests/v3/test_v3_market_selector.py`

Tests must cover:
1. Empty records → `RUN_NONE`
2. Only equity records (strong) → `RUN_EQUITIES`
3. Only crypto records (strong) → `RUN_CRYPTO`
4. Both strong → `RUN_BOTH`
5. Both weak → `RUN_NONE`
6. Regime PANIC veto → `RUN_NONE` regardless of scores
7. Regime RISK\_OFF reduces scores
8. Regime EVENT\_DRIVEN boosts scores
9. Determinism: same input → same output (no randomness)
10. `MarketSelectorDecision.route` is a valid `MarketRoute` value
11. Equity-specific event type (earnings) → boosts equity score, not crypto
12. Crypto-specific event type (regulatory/CFTC) → boosts crypto score, not equity

---

### Step 4 — Wire into docs

Update `docs/design/V3_NEWS_MARKET_SELECTOR_AUDIT.md` with Phase 7 (Code Changes)
and Phase 8 (Tests) once implementation is confirmed.

---

### What is NOT in scope for this step

| Deferred item | Reason |
|---|---|
| Integrating with APScheduler or any runtime trigger | Scheduling is V2-specific and orthogonal to the routing logic |
| Connecting to live news providers | Provider integration already exists; this module is downstream of providers |
| Persisting routing decisions to disk | The architecture supports it but adds I/O complexity before the logic is proven |
| Migrating the V2 `trade_planner.py` into V3 | NSE/India-specific; out of scope for cross-asset selector |
| Regime auto-detection within the selector | The selector *accepts* a regime label; it does not detect regime itself |
| V2 scheduler changes | V2 must remain untouched |

---

## Summary

The repo already contains:
1. A canonical news event schema (`NewsIntelligenceRecord`) designed for exactly this purpose
2. A scoring formula that should be adapted (not copied) into the selector
3. An event taxonomy that implicitly carries asset-class affinity (must be made explicit)
4. A V3 dispatch layer (`_strategy_runner.py`) that is the right integration point
5. An `AssetClass` enum that is the right typed token for routing

The repo does **not** contain:
- A cross-asset opportunity scorer
- A `MarketRoute` enum or `MarketSelectorDecision` type
- Any module that asks "equities or crypto?" based on current news
- An `intelligence/` namespace in V3

The migration path is low-risk because:
- The new module has no V2 imports (zero risk of V2 corruption)
- It is a pure addition to V3 (zero risk to existing V3 pipelines)
- Its input contract (`NewsIntelligenceRecord`) already exists and is tested
- Its output contract (`MarketRoute`) is a simple enum addition

**Recommended next step:** Implement Steps 1–3 of Phase 6 as described above.
