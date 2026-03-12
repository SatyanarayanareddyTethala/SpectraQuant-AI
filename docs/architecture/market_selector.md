# V3 Market Selector

`src/spectraquant_v3/intelligence/market_selector.py`

---

## Overview

The **Market Selector** is a deterministic, news-first routing component that
decides whether current market intelligence points toward **equities**,
**crypto**, or a **mixed** allocation.  It consumes provider-agnostic
`NewsIntelligenceRecord` inputs and emits a typed `MarketSelectorDecision`
that can sit above the V3 strategy runner.

---

## Input Model

The selector accepts a list of `NewsIntelligenceRecord` objects.  Each record
carries the following key fields:

| Field | Type | Description |
|---|---|---|
| `event_type` | `str` | Canonical event category (e.g. `EARNINGS`, `MACRO`, `ONCHAIN`) |
| `sentiment_score` | `float` | Signed sentiment in `[-1.0, +1.0]` |
| `impact_score` | `float` | Magnitude of the event's expected market impact `[0.0, 1.0]` |
| `published_at` | `str` (ISO-8601) | Publication timestamp; supports trailing `Z` |
| `provider` | `str` | Source provider identifier |

The selector also accepts an optional `MarketSelectorConfig` (or plain `dict`)
to tune thresholds, weights, and risk-off behaviour at runtime.

---

## Event Type Canonicalization

Before any scoring, `event_type` values are:

1. Stripped of whitespace and converted to **UPPERCASE**.
2. Aliased through `_EVENT_TYPE_ALIASES` — e.g. `"M&A"` → `"M_AND_A"`.
3. Unknown values fall back to `"UNKNOWN"`.

The resulting key is looked up in `EVENT_ASSET_AFFINITY`, which maps each
event type to `{"equity": float, "crypto": float}` scores.

```python
EVENT_ASSET_AFFINITY = {
    "EARNINGS":             {"equity": 1.00, "crypto": 0.05},
    "MACRO":                {"equity": 0.60, "crypto": 0.80},
    "REGULATORY":           {"equity": 0.50, "crypto": 0.85},
    "PROTOCOL_UPGRADE":     {"equity": 0.05, "crypto": 0.95},
    "EXCHANGE_HACK":        {"equity": 0.00, "crypto": 1.00},
    "ONCHAIN":              {"equity": 0.00, "crypto": 0.95},
    "UNKNOWN":              {"equity": 0.50, "crypto": 0.50},
    # … (see source for full table)
}
```

---

## Scoring Logic

### Per-record score

Each record contributes a per-record weight:

```
sentiment_factor = 0.5 + 0.5 * abs(sentiment_score)
record_weight    = impact_score * sentiment_factor * affinity[asset_class]
```

This means:
- High-impact, strongly-opinionated news on a clearly asset-specific event
  dominates the signal.
- Neutral sentiment (`score ≈ 0`) halves the sentiment factor to `0.5`.

### Intensity score

The top-`K` (`K = 3`) records by combined weight are selected.  Their
contributions are blended with fixed weights:

```
intensity_score = 0.85 * top_1 + 0.10 * top_2 + 0.05 * top_3
```

### Breadth score

Breadth measures how many records exceed a dynamic threshold:

```
breadth_cutoff  = max(0.05, mean(weights) * 0.5)
breadth_score   = count(w > breadth_cutoff) / BREADTH_NORMALIZATION_FACTOR   # capped at 1.0
```

### Final per-asset score

```
raw_score  = 0.90 * intensity_score + 0.10 * breadth_score
noise_adj  = raw_score - noise_penalty   # small penalty for very low-quality records
```

---

## Risk-Off and Stress Adjustments

After computing base scores, global multipliers are applied when triggered:

| Condition | Multiplier | Applied to |
|---|---|---|
| Risk-off event detected | `0.75` | Both equity and crypto scores |
| Cross-asset stress detected | `0.90` | Both equity and crypto scores |
| Event-driven boost eligible | `1.10` | Winning asset class |

The `VetoFlags` dataclass records which adjustments fired:

- `risk_penalty_applied` — broad flag (always set when the risk-off multiplier fires)
- `risk_off_penalty_applied` — legacy alias kept for backward compatibility

---

## Routing Decision

The final `MarketSelectorDecision` contains:

| Field | Type | Description |
|---|---|---|
| `recommended_route` | `MarketRoute` | `EQUITY`, `CRYPTO`, or `MIXED` |
| `score_breakdown` | `ScoreBreakdown` | Per-asset final scores and confidence |
| `contributing_events` | `list[ContributingEventSummary]` | Top events that drove the decision |
| `veto_flags` | `VetoFlags` | Risk-off and stress flags that fired |
| `scored_at` | `str` (ISO-8601) | Wall-clock timestamp of the decision |
| `reference_time` | `str` (ISO-8601) | UTC recency anchor used during scoring |
| `rationale` | `str` | Human-readable explanation |

Backward-compatible `@property` accessors expose legacy names (`route`,
`equity_score`, `crypto_score`) on `MarketSelectorDecision`.

### Routing threshold

The route is decided by comparing final equity vs crypto scores:

```
if abs(equity_score - crypto_score) < threshold:
    route = MIXED
elif equity_score > crypto_score:
    route = EQUITY
else:
    route = CRYPTO
```

The `threshold` defaults to `0.05` and is configurable via
`MarketSelectorConfig.mixed_route_threshold`.

---

## Rationale Structure

The `rationale` field is a short natural-language summary produced
deterministically from the score breakdown and top contributing events.
It is intended for logging and human review, not for downstream parsing.

Example:

```
Route: EQUITY | equity=0.712 crypto=0.183 | Top event: EARNINGS (impact=0.90,
sentiment=0.75) | risk_off=False | cross_asset_stress=False
```

---

## Usage Example

```python
from spectraquant_v3.intelligence.market_selector import MarketSelector
from spectraquant_v3.core.news_schema import NewsIntelligenceRecord

records = [
    NewsIntelligenceRecord(
        event_type="EARNINGS",
        sentiment_score=0.8,
        impact_score=0.9,
        published_at="2024-01-15T09:30:00Z",
        provider="perplexity",
    ),
]

selector = MarketSelector()
decision = selector.select(records)

print(decision.recommended_route)   # MarketRoute.EQUITY
print(decision.rationale)
print(decision.to_dict())
```

---

## Configuration

Pass a `MarketSelectorConfig` or plain `dict` to `MarketSelector(config=...)`:

```python
from spectraquant_v3.intelligence.market_selector import MarketSelectorConfig, MarketSelector

config = MarketSelectorConfig(
    mixed_route_threshold=0.10,
    risk_off_multiplier=0.70,
    top_k_intensity=5,
)
selector = MarketSelector(config=config)
```

---

## Related Files

| Path | Role |
|---|---|
| `src/spectraquant_v3/intelligence/market_selector.py` | Implementation |
| `src/spectraquant_v3/core/news_schema.py` | `NewsIntelligenceRecord` definition |
| `src/spectraquant_v3/core/enums.py` | `MarketRoute` enum |
| `tests/v3/test_v3_market_selector.py` | Full test suite |
| `docs/design/V3_NEWS_MARKET_SELECTOR_AUDIT.md` | Design audit notes |

