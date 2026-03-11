"""Provider-agnostic news intelligence schema for SpectraQuant-AI-V3.

Defines the canonical :class:`NewsIntelligenceRecord` that every news
intelligence provider must produce, along with the
:class:`NewsIntelligenceProvider` protocol that adapters implement.

Design constraints
------------------
* The schema is asset-class-agnostic — it works for both equities and crypto.
* Providers handle *transport and extraction only*; they must never write to
  cache or make run-mode decisions.
* Deterministic backtest compatibility is preserved by keeping cached
  normalised records separate from live discovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical news-intelligence record
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    """Return *value* clamped to the closed interval [*lo*, *hi*]."""
    return max(lo, min(hi, value))


@dataclass
class NewsIntelligenceRecord:
    """A single provider-agnostic news intelligence event.

    Every news intelligence adapter must produce one or more of these records
    per query.  The schema is intentionally minimal and normalised so that
    downstream feature generation and backtest replay remain provider-agnostic.

    Attributes:
        canonical_symbol: Upper-case canonical ticker, e.g. ``"AAPL"`` or
            ``"BTC"``.
        asset:            Asset-class string, e.g. ``"equity"`` or ``"crypto"``.
        timestamp:        ISO-8601 UTC timestamp of the event / article cluster.
        event_type:       Machine-readable event category (e.g. ``"earnings"``,
            ``"listing"``, ``"security_incident"``).
        sentiment_score:  Normalised sentiment in ``[-1.0, +1.0]``.
        impact_score:     Estimated market impact in ``[0.0, 1.0]``.
        article_count:    Number of source articles that contributed to this
            record.  Must be ``>= 1``.
        source_urls:      List of source article URLs.
        confidence:       Provider confidence in ``[0.0, 1.0]``.
        rationale:        Free-text explanation from the provider.
        provider:         Name of the provider that produced this record.
        raw_response:     Optional dict holding the raw provider payload for
            debugging.  Never used in downstream feature generation.
    """

    canonical_symbol: str
    asset: str
    timestamp: str
    event_type: str
    sentiment_score: float = 0.0
    impact_score: float = 0.0
    article_count: int = 1
    source_urls: list[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    provider: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Clamp numeric fields to documented ranges and validate invariants."""
        if self.sentiment_score < -1.0 or self.sentiment_score > 1.0:
            logger.warning(
                "NewsIntelligenceRecord: sentiment_score=%.4f for %r is outside "
                "[-1.0, +1.0]; clamping.",
                self.sentiment_score,
                self.canonical_symbol,
            )
            self.sentiment_score = _clamp(self.sentiment_score, -1.0, 1.0)

        if self.impact_score < 0.0 or self.impact_score > 1.0:
            logger.warning(
                "NewsIntelligenceRecord: impact_score=%.4f for %r is outside "
                "[0.0, 1.0]; clamping.",
                self.impact_score,
                self.canonical_symbol,
            )
            self.impact_score = _clamp(self.impact_score, 0.0, 1.0)

        if self.confidence < 0.0 or self.confidence > 1.0:
            logger.warning(
                "NewsIntelligenceRecord: confidence=%.4f for %r is outside "
                "[0.0, 1.0]; clamping.",
                self.confidence,
                self.canonical_symbol,
            )
            self.confidence = _clamp(self.confidence, 0.0, 1.0)

        if self.article_count < 1:
            self.article_count = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for DataFrame or JSON export.

        The ``raw_response`` field is intentionally excluded to keep
        cached records deterministic and compact.
        """
        return {
            "canonical_symbol": self.canonical_symbol,
            "asset": self.asset,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "sentiment_score": self.sentiment_score,
            "impact_score": self.impact_score,
            "article_count": self.article_count,
            "source_urls": list(self.source_urls),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "provider": self.provider,
        }


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

#: Fields that must be present in every serialised record.
REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "canonical_symbol",
        "asset",
        "timestamp",
        "event_type",
        "sentiment_score",
        "impact_score",
        "article_count",
        "source_urls",
        "confidence",
        "rationale",
    }
)


def validate_news_intelligence_record(record: dict[str, Any]) -> None:
    """Validate that *record* contains all required fields with correct types.

    Raises:
        ValueError: When a required field is missing or has an invalid type.
    """
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        raise ValueError(f"NewsIntelligenceRecord missing required fields: {sorted(missing)}")

    if not isinstance(record.get("canonical_symbol"), str) or not record["canonical_symbol"].strip():
        raise ValueError("canonical_symbol must be a non-empty string")

    if not isinstance(record.get("asset"), str) or not record["asset"].strip():
        raise ValueError("asset must be a non-empty string")

    if not isinstance(record.get("timestamp"), str) or not record["timestamp"].strip():
        raise ValueError("timestamp must be a non-empty string")

    if not isinstance(record.get("source_urls"), list):
        raise ValueError("source_urls must be a list")

    if not isinstance(record.get("article_count"), int) or record["article_count"] < 1:
        raise ValueError("article_count must be an integer >= 1")


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class NewsIntelligenceProvider(Protocol):
    """Protocol that all news intelligence adapters must implement.

    Adapters are thin wrappers — they handle transport and extraction only.
    They must never write to cache, decide run-mode behaviour, or import from
    asset-class-specific sub-packages.
    """

    @property
    def provider_name(self) -> str:
        """Short machine-readable provider identifier, e.g. ``"perplexity"``."""
        ...

    def fetch_intelligence(
        self,
        symbols: list[str],
        *,
        asset_class: str = "",
        max_results: int = 10,
    ) -> list[NewsIntelligenceRecord]:
        """Fetch news intelligence for one or more canonical symbols.

        Args:
            symbols:     List of canonical tickers, e.g. ``["AAPL", "MSFT"]``
                         or ``["BTC", "ETH"]``.
            asset_class: Optional asset-class hint (``"equity"`` / ``"crypto"``).
            max_results: Soft limit on the number of records per symbol.

        Returns:
            List of :class:`NewsIntelligenceRecord` instances.  An empty list
            is valid when no events are found — it must not raise in that case.

        Raises:
            SpectraQuantError: On network or parsing failures.
        """
        ...
