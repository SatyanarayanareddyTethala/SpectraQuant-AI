"""News intelligence feature generation from cached store records.

Converts cached :class:`~spectraquant_v3.core.news_intel_store.NewsIntelligenceStore`
records into deterministic, point-in-time-safe daily feature DataFrames that
can be consumed by V3 equity and crypto strategies and backtests.

Design principles
-----------------
* **Point-in-time safety**: records whose ``timestamp`` is strictly after
  ``as_of_date`` are excluded before any aggregation, so no future data
  leaks into a backtest.
* **Deterministic output**: given the same set of cached records and the
  same ``as_of_date``, the output is always identical.
* **Provider-agnostic**: the builder works with any provider because it
  operates on the normalised :class:`NewsIntelligenceRecord` schema.
* **Asset-class-agnostic**: the same builder serves both equity and crypto
  feature pipelines; no asset-specific logic lives here.

Feature columns produced
------------------------
``news_sentiment_score``
    Mean ``sentiment_score`` across all records on each day.
``news_impact_score``
    Mean ``impact_score`` across all records on each day.
``article_count``
    Sum of ``article_count`` across all records on each day.
``confidence_score``
    Mean ``confidence`` across all records on each day.
``news_sentiment_rw``
    Exponentially-weighted (recency-weighted) sentiment.  Uses
    ``pandas.DataFrame.ewm(halflife=recency_halflife_days)`` applied to the
    per-day sentiment so that recent events weigh more than stale ones.
``news_impact_rw``
    Exponentially-weighted impact, same decay scheme as ``news_sentiment_rw``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_FEATURE_COLUMNS: list[str] = [
    "news_sentiment_score",
    "news_impact_score",
    "article_count",
    "confidence_score",
    "news_sentiment_rw",
    "news_impact_rw",
]

_EMPTY_SCHEMA: dict[str, Any] = {col: [] for col in _FEATURE_COLUMNS}


def _empty_features() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical feature schema."""
    return pd.DataFrame(_EMPTY_SCHEMA)


def _parse_timestamp(ts: Any) -> pd.Timestamp | None:
    """Parse a timestamp value to a tz-naive ``pd.Timestamp``, or ``None``."""
    if pd.isna(ts):
        return None
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return t
    except (ValueError, TypeError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Primary public function
# ---------------------------------------------------------------------------


def build_daily_features(
    records: list[dict[str, Any]],
    *,
    as_of_date: str | pd.Timestamp | None = None,
    recency_halflife_days: float = 3.0,
) -> pd.DataFrame:
    """Convert cached news intelligence records into daily feature rows.

    For each calendar date in *records*, aggregates all records on that date
    to produce a single feature row.  Only records with ``timestamp <=
    as_of_date`` are included, ensuring no future leakage.

    Args:
        records:               List of record dicts from
                               :class:`~spectraquant_v3.core.news_intel_store.NewsIntelligenceStore`.
        as_of_date:            Upper-bound timestamp for point-in-time safety.
                               Pass ``None`` to include all records (useful
                               for live runs).
        recency_halflife_days: Half-life (in calendar days) for the
                               exponential decay applied when computing
                               recency-weighted features
                               (``news_sentiment_rw``, ``news_impact_rw``).
                               Must be ``> 0``.

    Returns:
        :class:`pandas.DataFrame` with a tz-naive ``DatetimeIndex`` (date
        resolution) and columns defined by :data:`_FEATURE_COLUMNS`.
        Returns an empty DataFrame (with the same schema) when *records* is
        empty or all records are filtered out by *as_of_date*.

    Notes:
        The returned DataFrame is sorted ascending by date and has no
        duplicate index entries.  The caller must not sort or filter it after
        the fact; doing so would break the point-in-time contract.
    """
    if recency_halflife_days <= 0:
        raise ValueError(
            f"recency_halflife_days must be > 0, got {recency_halflife_days!r}"
        )

    if not records:
        return _empty_features()

    # Normalise as_of_date to a tz-naive Timestamp for comparison
    pit_bound: pd.Timestamp | None = None
    if as_of_date is not None:
        pit_bound = _parse_timestamp(as_of_date)
        if pit_bound is None:
            logger.warning(
                "build_daily_features: could not parse as_of_date=%r; "
                "no filtering applied.",
                as_of_date,
            )

    # Build a list of (date, sentiment, impact, article_count, confidence)
    rows: list[tuple[pd.Timestamp, float, float, int, float]] = []
    for rec in records:
        ts = _parse_timestamp(rec.get("timestamp"))
        if ts is None:
            continue

        # Point-in-time filter — exclude any record from the future
        if pit_bound is not None and ts > pit_bound:
            continue

        date = ts.normalize()  # truncate to midnight
        try:
            sentiment = float(rec.get("sentiment_score", 0.0))
            impact = float(rec.get("impact_score", 0.0))
            count = max(1, int(rec.get("article_count", 1)))
            confidence = float(rec.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue

        rows.append((date, sentiment, impact, count, confidence))

    if not rows:
        return _empty_features()

    df_raw = pd.DataFrame(
        rows,
        columns=["_date", "sentiment_score", "impact_score", "article_count", "confidence"],
    )

    # Daily aggregation: mean for scores, sum for counts
    agg = (
        df_raw.groupby("_date", sort=True)
        .agg(
            news_sentiment_score=("sentiment_score", "mean"),
            news_impact_score=("impact_score", "mean"),
            article_count=("article_count", "sum"),
            confidence_score=("confidence", "mean"),
        )
        .reset_index()
    )

    agg = agg.set_index("_date")
    agg.index.name = None
    agg = agg.sort_index()

    # Recency-weighted variants via EWM applied to the daily series
    halflife_td = pd.Timedelta(days=recency_halflife_days)
    agg["news_sentiment_rw"] = (
        agg["news_sentiment_score"]
        .ewm(halflife=halflife_td, times=agg.index, adjust=True)
        .mean()
    )
    agg["news_impact_rw"] = (
        agg["news_impact_score"]
        .ewm(halflife=halflife_td, times=agg.index, adjust=True)
        .mean()
    )

    return agg[_FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# High-level builder class
# ---------------------------------------------------------------------------


class NewsIntelligenceFeatureBuilder:
    """Builds daily news intelligence features from a cached store.

    This is the recommended entry-point for backtest feature pipelines.
    It wraps :func:`build_daily_features` and provides a multi-symbol
    convenience method that returns a dict suitable for the
    ``news_feature_map`` argument of
    :class:`~spectraquant_v3.backtest.engine.BacktestEngine`.

    Args:
        recency_halflife_days: Half-life for the EWM recency weighting.
                               Defaults to ``3.0`` (days).
    """

    def __init__(self, recency_halflife_days: float = 3.0) -> None:
        if recency_halflife_days <= 0:
            raise ValueError(
                f"recency_halflife_days must be > 0, got {recency_halflife_days!r}"
            )
        self.recency_halflife_days = recency_halflife_days

    # ------------------------------------------------------------------
    # Per-symbol API
    # ------------------------------------------------------------------

    def build(
        self,
        store: NewsIntelligenceStore,
        canonical_symbol: str,
        *,
        as_of_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Build daily features for one symbol.

        Reads all cached records for *canonical_symbol* from *store*,
        applies the point-in-time filter, and aggregates into daily rows.

        Args:
            store:            Initialised :class:`NewsIntelligenceStore`.
            canonical_symbol: Upper-case canonical ticker.
            as_of_date:       Upper-bound for point-in-time safety.

        Returns:
            Daily feature DataFrame (empty when no records exist).
        """
        records = store.read_records(canonical_symbol.upper())
        return build_daily_features(
            records,
            as_of_date=as_of_date,
            recency_halflife_days=self.recency_halflife_days,
        )

    # ------------------------------------------------------------------
    # Multi-symbol API (backtest convenience)
    # ------------------------------------------------------------------

    def build_many(
        self,
        store: NewsIntelligenceStore,
        symbols: list[str],
        *,
        as_of_date: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Build daily features for multiple symbols.

        Args:
            store:      Initialised :class:`NewsIntelligenceStore`.
            symbols:    List of canonical tickers.
            as_of_date: Upper-bound for point-in-time safety.

        Returns:
            Dict mapping canonical symbol → daily feature DataFrame.
            Symbols with no cached records are included as empty DataFrames.
        """
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            result[sym.upper()] = self.build(store, sym, as_of_date=as_of_date)
        return result
