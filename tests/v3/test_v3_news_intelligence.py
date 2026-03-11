"""Tests for the SpectraQuant-AI-V3 news intelligence adapter layer.

Covers:
* NewsIntelligenceRecord schema validation and clamping
* NewsIntelligenceProvider protocol compliance
* PerplexityNewsProvider with mocked HTTP responses
* NewsIntelligenceStore read/write/dedup/backtest determinism
* Provider wiring from both equity and crypto ingestion namespaces

All tests are self-contained: no network calls, no persistent file-system
side-effects outside of pytest's ``tmp_path`` fixture.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from spectraquant_v3.core.news_schema import (
    REQUIRED_FIELDS,
    NewsIntelligenceProvider,
    NewsIntelligenceRecord,
    validate_news_intelligence_record,
)


# ===========================================================================
# Fixtures
# ===========================================================================


def _sample_record(**overrides: Any) -> NewsIntelligenceRecord:
    """Return a valid sample record, with optional field overrides."""
    defaults: dict[str, Any] = {
        "canonical_symbol": "AAPL",
        "asset": "equity",
        "timestamp": "2025-01-15T12:00:00+00:00",
        "event_type": "earnings",
        "sentiment_score": 0.6,
        "impact_score": 0.8,
        "article_count": 3,
        "source_urls": ["https://example.com/1", "https://example.com/2"],
        "confidence": 0.9,
        "rationale": "Strong Q4 results beat expectations.",
        "provider": "perplexity",
    }
    defaults.update(overrides)
    return NewsIntelligenceRecord(**defaults)


def _perplexity_api_body(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a mock Perplexity chat-completion response body."""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(items),
                }
            }
        ]
    }


# ===========================================================================
# Schema tests
# ===========================================================================


class TestNewsIntelligenceRecord:
    """Tests for the canonical record dataclass."""

    def test_basic_construction(self) -> None:
        rec = _sample_record()
        assert rec.canonical_symbol == "AAPL"
        assert rec.asset == "equity"
        assert rec.event_type == "earnings"
        assert rec.article_count == 3

    def test_sentiment_clamped_high(self) -> None:
        rec = _sample_record(sentiment_score=1.5)
        assert rec.sentiment_score == 1.0

    def test_sentiment_clamped_low(self) -> None:
        rec = _sample_record(sentiment_score=-2.0)
        assert rec.sentiment_score == -1.0

    def test_impact_clamped(self) -> None:
        rec = _sample_record(impact_score=1.5)
        assert rec.impact_score == 1.0

    def test_confidence_clamped(self) -> None:
        rec = _sample_record(confidence=-0.1)
        assert rec.confidence == 0.0

    def test_article_count_minimum(self) -> None:
        rec = _sample_record(article_count=0)
        assert rec.article_count == 1

    def test_to_dict_excludes_raw_response(self) -> None:
        rec = _sample_record(raw_response={"debug": True})
        d = rec.to_dict()
        assert "raw_response" not in d
        assert d["canonical_symbol"] == "AAPL"

    def test_to_dict_has_all_required_fields(self) -> None:
        d = _sample_record().to_dict()
        assert REQUIRED_FIELDS <= set(d.keys())


class TestValidateNewsIntelligenceRecord:
    """Tests for the standalone validation helper."""

    def test_valid_record(self) -> None:
        validate_news_intelligence_record(_sample_record().to_dict())

    def test_missing_field_raises(self) -> None:
        d = _sample_record().to_dict()
        del d["event_type"]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_news_intelligence_record(d)

    def test_empty_symbol_raises(self) -> None:
        d = _sample_record().to_dict()
        d["canonical_symbol"] = ""
        with pytest.raises(ValueError, match="canonical_symbol"):
            validate_news_intelligence_record(d)

    def test_non_list_source_urls_raises(self) -> None:
        d = _sample_record().to_dict()
        d["source_urls"] = "not-a-list"
        with pytest.raises(ValueError, match="source_urls must be a list"):
            validate_news_intelligence_record(d)

    def test_article_count_zero_raises(self) -> None:
        d = _sample_record().to_dict()
        d["article_count"] = 0
        with pytest.raises(ValueError, match="article_count"):
            validate_news_intelligence_record(d)


# ===========================================================================
# Protocol tests
# ===========================================================================


class TestNewsIntelligenceProtocol:
    """Verify the protocol is runtime-checkable."""

    def test_perplexity_satisfies_protocol(self) -> None:
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        provider = PerplexityNewsProvider(api_key="test-key")
        assert isinstance(provider, NewsIntelligenceProvider)

    def test_protocol_rejects_plain_object(self) -> None:
        assert not isinstance(object(), NewsIntelligenceProvider)


# ===========================================================================
# Perplexity adapter tests
# ===========================================================================


class TestPerplexityNewsProvider:
    """Tests for the Perplexity adapter with mocked HTTP."""

    def _make_mock_session(self, body: dict[str, Any], status: int = 200) -> MagicMock:
        """Return a mock httpx-compatible client."""
        response = MagicMock()
        response.status_code = status
        response.json.return_value = body
        response.text = json.dumps(body)
        session = MagicMock()
        session.post.return_value = response
        return session

    def test_fetch_intelligence_success(self) -> None:
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        items = [
            {
                "canonical_symbol": "AAPL",
                "timestamp": "2025-01-15T12:00:00+00:00",
                "event_type": "earnings",
                "sentiment_score": 0.7,
                "impact_score": 0.8,
                "article_count": 5,
                "source_urls": ["https://example.com/a"],
                "confidence": 0.9,
                "rationale": "Beat estimates.",
            },
        ]
        body = _perplexity_api_body(items)
        session = self._make_mock_session(body)

        provider = PerplexityNewsProvider(api_key="test-key", _session=session)
        records = provider.fetch_intelligence(["AAPL"], asset_class="equity")

        assert len(records) == 1
        assert records[0].canonical_symbol == "AAPL"
        assert records[0].event_type == "earnings"
        assert records[0].provider == "perplexity"
        session.post.assert_called_once()

    def test_fetch_intelligence_multiple_symbols(self) -> None:
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        items = [
            {
                "canonical_symbol": "BTC",
                "timestamp": "2025-01-15T12:00:00+00:00",
                "event_type": "regulation",
                "sentiment_score": -0.3,
                "impact_score": 0.6,
                "article_count": 2,
                "source_urls": [],
                "confidence": 0.7,
                "rationale": "SEC crackdown.",
            },
            {
                "canonical_symbol": "ETH",
                "timestamp": "2025-01-15T13:00:00+00:00",
                "event_type": "protocol_upgrade",
                "sentiment_score": 0.5,
                "impact_score": 0.4,
                "article_count": 1,
                "source_urls": [],
                "confidence": 0.6,
                "rationale": "Dencun upgrade.",
            },
        ]
        body = _perplexity_api_body(items)
        session = self._make_mock_session(body)

        provider = PerplexityNewsProvider(api_key="test-key", _session=session)
        records = provider.fetch_intelligence(
            ["BTC", "ETH"], asset_class="crypto"
        )

        assert len(records) == 2
        symbols = {r.canonical_symbol for r in records}
        assert symbols == {"BTC", "ETH"}

    def test_fetch_intelligence_empty_response(self) -> None:
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        body = {"choices": [{"message": {"content": "[]"}}]}
        session = self._make_mock_session(body)

        provider = PerplexityNewsProvider(api_key="test-key", _session=session)
        records = provider.fetch_intelligence(["AAPL"])
        assert records == []

    def test_fetch_intelligence_no_choices(self) -> None:
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        body = {"choices": []}
        session = self._make_mock_session(body)

        provider = PerplexityNewsProvider(api_key="test-key", _session=session)
        records = provider.fetch_intelligence(["AAPL"])
        assert records == []

    def test_fetch_intelligence_malformed_json(self) -> None:
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        body = {"choices": [{"message": {"content": "NOT JSON"}}]}
        session = self._make_mock_session(body)

        provider = PerplexityNewsProvider(api_key="test-key", _session=session)
        records = provider.fetch_intelligence(["AAPL"])
        assert records == []

    def test_fetch_intelligence_strips_markdown_fences(self) -> None:
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        items = [
            {
                "canonical_symbol": "MSFT",
                "timestamp": "2025-02-01T09:00:00+00:00",
                "event_type": "partnership",
                "sentiment_score": 0.4,
                "impact_score": 0.5,
                "article_count": 2,
                "source_urls": [],
                "confidence": 0.8,
                "rationale": "New AI partnership.",
            },
        ]
        content = "```json\n" + json.dumps(items) + "\n```"
        body = {"choices": [{"message": {"content": content}}]}
        session = self._make_mock_session(body)

        provider = PerplexityNewsProvider(api_key="test-key", _session=session)
        records = provider.fetch_intelligence(["MSFT"], asset_class="equity")
        assert len(records) == 1
        assert records[0].canonical_symbol == "MSFT"

    def test_fetch_intelligence_empty_symbols_returns_empty(self) -> None:
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        provider = PerplexityNewsProvider(api_key="test-key")
        records = provider.fetch_intelligence([])
        assert records == []

    def test_no_api_key_raises(self) -> None:
        from spectraquant_v3.core.errors import SpectraQuantError
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        provider = PerplexityNewsProvider(api_key="")
        with pytest.raises(SpectraQuantError, match="no API key"):
            provider.fetch_intelligence(["AAPL"])

    def test_http_error_raises(self) -> None:
        from spectraquant_v3.core.errors import SpectraQuantError
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        session = self._make_mock_session({}, status=429)
        provider = PerplexityNewsProvider(api_key="test-key", _session=session)

        with pytest.raises(SpectraQuantError, match="HTTP 429"):
            provider.fetch_intelligence(["AAPL"])

    def test_provider_name_property(self) -> None:
        from spectraquant_v3.core.providers.perplexity_provider import (
            PerplexityNewsProvider,
        )

        provider = PerplexityNewsProvider(api_key="test-key")
        assert provider.provider_name == "perplexity"


# ===========================================================================
# News intelligence store tests
# ===========================================================================


class TestNewsIntelligenceStore:
    """Tests for the cached record store."""

    def test_write_and_read_roundtrip(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        records = [_sample_record(), _sample_record(event_type="partnership")]
        store.write_records("AAPL", records)

        loaded = store.read_records("AAPL")
        assert len(loaded) == 2
        assert loaded[0]["canonical_symbol"] == "AAPL"

    def test_deduplication_on_append(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        rec = _sample_record()
        store.write_records("AAPL", [rec])
        store.write_records("AAPL", [rec])  # duplicate

        loaded = store.read_records("AAPL")
        assert len(loaded) == 1

    def test_has_records(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        assert not store.has_records("AAPL")

        store.write_records("AAPL", [_sample_record()])
        assert store.has_records("AAPL")

    def test_read_records_cache_miss(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        assert store.read_records("UNKNOWN") == []

    def test_read_as_dataframe(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        store.write_records("BTC", [_sample_record(canonical_symbol="BTC", asset="crypto")])

        df = store.read_as_dataframe("BTC")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["canonical_symbol"] == "BTC"

    def test_read_as_dataframe_cache_miss(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        df = store.read_as_dataframe("MISSING")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_source_urls_serialization(self, tmp_path: Any) -> None:
        """Verify source_urls survive parquet roundtrip as lists."""
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        urls = ["https://a.com", "https://b.com"]
        store = NewsIntelligenceStore(tmp_path / "news_intel")
        store.write_records("AAPL", [_sample_record(source_urls=urls)])

        loaded = store.read_records("AAPL")
        assert loaded[0]["source_urls"] == urls


# ===========================================================================
# Provider wiring tests
# ===========================================================================


class TestProviderWiring:
    """Verify provider is importable from both asset-class namespaces."""

    def test_equity_provider_import(self) -> None:
        from spectraquant_v3.equities.ingestion.providers import PerplexityNewsProvider

        provider = PerplexityNewsProvider(api_key="test")
        assert provider.provider_name == "perplexity"

    def test_crypto_provider_import(self) -> None:
        from spectraquant_v3.crypto.ingestion.providers import PerplexityNewsProvider

        provider = PerplexityNewsProvider(api_key="test")
        assert provider.provider_name == "perplexity"

    def test_same_class_both_namespaces(self) -> None:
        from spectraquant_v3.crypto.ingestion.providers import (
            PerplexityNewsProvider as CryptoP,
        )
        from spectraquant_v3.equities.ingestion.providers import (
            PerplexityNewsProvider as EquityP,
        )

        assert CryptoP is EquityP


# ===========================================================================
# Backtest determinism tests
# ===========================================================================


class TestBacktestDeterminism:
    """Verify that the cached store path produces deterministic results."""

    def test_identical_reads_after_write(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        records = [
            _sample_record(event_type="earnings"),
            _sample_record(event_type="regulation", sentiment_score=-0.5),
        ]
        store.write_records("AAPL", records)

        read1 = store.read_records("AAPL")
        read2 = store.read_records("AAPL")
        assert read1 == read2

    def test_dataframe_deterministic(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        store.write_records("ETH", [
            _sample_record(canonical_symbol="ETH", asset="crypto"),
        ])

        df1 = store.read_as_dataframe("ETH")
        df2 = store.read_as_dataframe("ETH")
        pd.testing.assert_frame_equal(df1, df2)


# ===========================================================================
# News intelligence feature builder tests
# ===========================================================================


def _intel_records(
    symbol: str = "AAPL",
    asset: str = "equity",
    dates_sentiments: list[tuple[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Return a list of minimal record dicts for feature-builder tests."""
    if dates_sentiments is None:
        dates_sentiments = [("2025-01-10T12:00:00", 0.5)]
    records = []
    for ts, sent in dates_sentiments:
        records.append(
            {
                "canonical_symbol": symbol,
                "asset": asset,
                "timestamp": ts,
                "event_type": "earnings",
                "sentiment_score": sent,
                "impact_score": abs(sent),
                "article_count": 2,
                "source_urls": [],
                "confidence": 0.8,
                "rationale": "test",
                "provider": "test",
            }
        )
    return records


class TestBuildDailyFeatures:
    """Unit tests for the ``build_daily_features`` standalone function."""

    def test_empty_records_returns_empty_dataframe(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        result = build_daily_features([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_has_feature_columns(self) -> None:
        from spectraquant_v3.core.news_intel_features import (
            _FEATURE_COLUMNS,
            build_daily_features,
        )

        result = build_daily_features([])
        assert list(result.columns) == _FEATURE_COLUMNS

    def test_single_record_produces_one_row(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(dates_sentiments=[("2025-01-10T12:00:00", 0.6)])
        result = build_daily_features(records)
        assert len(result) == 1

    def test_two_records_same_day_aggregated_to_one_row(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(
            dates_sentiments=[
                ("2025-01-10T09:00:00", 0.4),
                ("2025-01-10T17:00:00", 0.8),
            ]
        )
        result = build_daily_features(records)
        assert len(result) == 1
        # Mean of 0.4 and 0.8
        assert abs(result["news_sentiment_score"].iloc[0] - 0.6) < 1e-9

    def test_two_records_different_days_produce_two_rows(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(
            dates_sentiments=[
                ("2025-01-10T09:00:00", 0.5),
                ("2025-01-15T09:00:00", -0.3),
            ]
        )
        result = build_daily_features(records)
        assert len(result) == 2

    def test_article_count_summed_per_day(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = [
            {**_intel_records(dates_sentiments=[("2025-01-10T09:00:00", 0.5)])[0], "article_count": 3},
            {**_intel_records(dates_sentiments=[("2025-01-10T15:00:00", 0.2)])[0], "article_count": 5},
        ]
        result = build_daily_features(records)
        assert result["article_count"].iloc[0] == 8

    def test_sorted_ascending_by_date(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(
            dates_sentiments=[
                ("2025-01-20T00:00:00", 0.1),
                ("2025-01-10T00:00:00", 0.9),
                ("2025-01-15T00:00:00", 0.5),
            ]
        )
        result = build_daily_features(records)
        assert list(result.index) == sorted(result.index)

    def test_recency_weighted_columns_present(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(dates_sentiments=[("2025-01-10T00:00:00", 0.5)])
        result = build_daily_features(records)
        assert "news_sentiment_rw" in result.columns
        assert "news_impact_rw" in result.columns

    def test_recency_weighted_single_row_equals_plain(self) -> None:
        """With a single record, EWM weight = 1, so rw == plain."""
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(dates_sentiments=[("2025-01-10T00:00:00", 0.7)])
        result = build_daily_features(records)
        assert abs(result["news_sentiment_rw"].iloc[0] - result["news_sentiment_score"].iloc[0]) < 1e-6

    def test_output_is_deterministic(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(
            dates_sentiments=[
                ("2025-01-10T00:00:00", 0.4),
                ("2025-01-15T00:00:00", -0.2),
            ]
        )
        df1 = build_daily_features(records)
        df2 = build_daily_features(records)
        pd.testing.assert_frame_equal(df1, df2)

    def test_invalid_halflife_raises(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        with pytest.raises(ValueError, match="recency_halflife_days"):
            build_daily_features([], recency_halflife_days=0.0)


class TestBuildDailyFeaturesPointInTime:
    """Tests proving no future leakage occurs."""

    def test_as_of_date_excludes_future_records(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(
            dates_sentiments=[
                ("2025-01-10T00:00:00", 0.5),  # past
                ("2025-01-20T00:00:00", 0.9),  # future
            ]
        )
        result = build_daily_features(records, as_of_date="2025-01-15T23:59:59")
        assert len(result) == 1
        assert result["news_sentiment_score"].iloc[0] == pytest.approx(0.5)

    def test_as_of_date_includes_record_on_boundary(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(dates_sentiments=[("2025-01-15T12:00:00", 0.5)])
        # Exact match: record at 12:00 on 2025-01-15; boundary at end of day
        result = build_daily_features(records, as_of_date="2025-01-15T23:59:59")
        assert len(result) == 1

    def test_all_records_future_returns_empty(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(
            dates_sentiments=[
                ("2025-02-01T00:00:00", 0.5),
                ("2025-03-01T00:00:00", 0.8),
            ]
        )
        result = build_daily_features(records, as_of_date="2025-01-31")
        assert result.empty

    def test_no_as_of_date_includes_all_records(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(
            dates_sentiments=[
                ("2020-01-01T00:00:00", 0.1),
                ("2030-12-31T00:00:00", 0.9),
            ]
        )
        result = build_daily_features(records, as_of_date=None)
        assert len(result) == 2

    def test_as_of_date_exactly_on_record_timestamp_is_included(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        ts = "2025-01-15T12:00:00"
        records = _intel_records(dates_sentiments=[(ts, 0.5)])
        result = build_daily_features(records, as_of_date=ts)
        assert len(result) == 1

    def test_records_one_second_after_boundary_excluded(self) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records(dates_sentiments=[("2025-01-15T12:00:01", 0.5)])
        result = build_daily_features(records, as_of_date="2025-01-15T12:00:00")
        assert result.empty


class TestBuildDailyFeaturesSymbolIsolation:
    """Tests verifying correct per-symbol isolation."""

    def test_different_symbols_isolated(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")

        aapl_records = [
            NewsIntelligenceRecord(**{
                "canonical_symbol": "AAPL",
                "asset": "equity",
                "timestamp": "2025-01-10T12:00:00",
                "event_type": "earnings",
                "sentiment_score": 0.8,
                "impact_score": 0.7,
                "article_count": 2,
                "source_urls": [],
                "confidence": 0.9,
                "rationale": "",
                "provider": "test",
            })
        ]
        msft_records = [
            NewsIntelligenceRecord(**{
                "canonical_symbol": "MSFT",
                "asset": "equity",
                "timestamp": "2025-01-10T12:00:00",
                "event_type": "earnings",
                "sentiment_score": -0.4,
                "impact_score": 0.5,
                "article_count": 1,
                "source_urls": [],
                "confidence": 0.7,
                "rationale": "",
                "provider": "test",
            })
        ]
        store.write_records("AAPL", aapl_records)
        store.write_records("MSFT", msft_records)

        aapl_df = build_daily_features(store.read_records("AAPL"))
        msft_df = build_daily_features(store.read_records("MSFT"))

        assert aapl_df["news_sentiment_score"].iloc[0] == pytest.approx(0.8)
        assert msft_df["news_sentiment_score"].iloc[0] == pytest.approx(-0.4)

    def test_unknown_symbol_returns_empty(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_features import build_daily_features
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        records = store.read_records("UNKNOWN_XYZ")
        df = build_daily_features(records)
        assert df.empty

    def test_mixed_symbols_in_records_all_aggregated(self) -> None:
        """Records from multiple symbols are all included when passed directly."""
        from spectraquant_v3.core.news_intel_features import build_daily_features

        records = _intel_records("AAPL", dates_sentiments=[("2025-01-10T00:00:00", 0.5)])
        records += _intel_records("MSFT", dates_sentiments=[("2025-01-10T00:00:00", -0.2)])
        # Both records are on the same day → one row after aggregation
        result = build_daily_features(records)
        assert len(result) == 1
        assert result["news_sentiment_score"].iloc[0] == pytest.approx(0.15)


class TestNewsIntelligenceFeatureBuilder:
    """Tests for the high-level ``NewsIntelligenceFeatureBuilder`` class."""

    def test_build_returns_dataframe(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_features import NewsIntelligenceFeatureBuilder
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        store.write_records("AAPL", [_sample_record()])

        builder = NewsIntelligenceFeatureBuilder()
        df = builder.build(store, "AAPL")

        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_build_empty_store_returns_empty(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_features import NewsIntelligenceFeatureBuilder
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        builder = NewsIntelligenceFeatureBuilder()
        df = builder.build(store, "MISSING_SYM")
        assert df.empty

    def test_build_many_returns_dict(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_features import NewsIntelligenceFeatureBuilder
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        store.write_records("AAPL", [_sample_record(canonical_symbol="AAPL")])
        store.write_records("BTC", [_sample_record(canonical_symbol="BTC", asset="crypto")])

        builder = NewsIntelligenceFeatureBuilder()
        result = builder.build_many(store, ["AAPL", "BTC"])

        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "BTC" in result
        assert not result["AAPL"].empty
        assert not result["BTC"].empty

    def test_build_many_symbol_isolation(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_features import NewsIntelligenceFeatureBuilder
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        store.write_records("AAPL", [_sample_record(sentiment_score=0.9)])
        store.write_records("ETH", [_sample_record(canonical_symbol="ETH", asset="crypto", sentiment_score=-0.6)])

        builder = NewsIntelligenceFeatureBuilder()
        result = builder.build_many(store, ["AAPL", "ETH"])

        assert result["AAPL"]["news_sentiment_score"].iloc[0] == pytest.approx(0.9)
        assert result["ETH"]["news_sentiment_score"].iloc[0] == pytest.approx(-0.6)

    def test_build_many_missing_symbol_returns_empty(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_features import NewsIntelligenceFeatureBuilder
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        builder = NewsIntelligenceFeatureBuilder()
        result = builder.build_many(store, ["GHOST_SYM"])
        assert result["GHOST_SYM"].empty

    def test_build_pit_filter_applied(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_features import NewsIntelligenceFeatureBuilder
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        records = [
            _sample_record(timestamp="2025-01-10T12:00:00+00:00", sentiment_score=0.5),
            _sample_record(timestamp="2025-02-10T12:00:00+00:00", sentiment_score=0.9),
        ]
        store.write_records("AAPL", records)

        builder = NewsIntelligenceFeatureBuilder()
        df = builder.build(store, "AAPL", as_of_date="2025-01-31")

        # Only the Jan record should be included
        assert len(df) == 1
        assert df["news_sentiment_score"].iloc[0] == pytest.approx(0.5)

    def test_invalid_halflife_raises(self) -> None:
        from spectraquant_v3.core.news_intel_features import NewsIntelligenceFeatureBuilder

        with pytest.raises(ValueError, match="recency_halflife_days"):
            NewsIntelligenceFeatureBuilder(recency_halflife_days=-1.0)

    def test_deterministic_repeated_calls(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_features import NewsIntelligenceFeatureBuilder
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        store.write_records("AAPL", [_sample_record(), _sample_record(event_type="partnership")])

        builder = NewsIntelligenceFeatureBuilder()
        df1 = builder.build(store, "AAPL")
        df2 = builder.build(store, "AAPL")
        pd.testing.assert_frame_equal(df1, df2)


class TestStoreConvenienceMethods:
    """Tests for NewsIntelligenceStore.build_news_feature_map()."""

    def test_build_news_feature_map_returns_dict(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        store.write_records("AAPL", [_sample_record()])

        feature_map = store.build_news_feature_map(["AAPL"])
        assert isinstance(feature_map, dict)
        assert "AAPL" in feature_map
        assert isinstance(feature_map["AAPL"], pd.DataFrame)

    def test_build_news_feature_map_multiple_symbols(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        store.write_records("AAPL", [_sample_record()])
        store.write_records("BTC", [_sample_record(canonical_symbol="BTC", asset="crypto")])

        feature_map = store.build_news_feature_map(["AAPL", "BTC"])
        assert len(feature_map) == 2

    def test_build_news_feature_map_pit_filter(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        records = [
            _sample_record(timestamp="2025-01-10T12:00:00+00:00", sentiment_score=0.5),
            _sample_record(timestamp="2025-03-10T12:00:00+00:00", sentiment_score=0.9),
        ]
        store.write_records("AAPL", records)

        feature_map = store.build_news_feature_map(["AAPL"], as_of_date="2025-02-01")
        df = feature_map["AAPL"]
        assert len(df) == 1

    def test_build_news_feature_map_missing_symbol_empty(self, tmp_path: Any) -> None:
        from spectraquant_v3.core.news_intel_store import NewsIntelligenceStore

        store = NewsIntelligenceStore(tmp_path / "news_intel")
        feature_map = store.build_news_feature_map(["NONEXISTENT"])
        assert feature_map["NONEXISTENT"].empty


class TestFeatureBuilderEquityImport:
    """Verify the builder is importable from the equity features sub-package."""

    def test_import_from_equities_features(self) -> None:
        from spectraquant_v3.equities.features import (
            NewsIntelligenceFeatureBuilder,
            build_daily_features,
        )

        assert callable(build_daily_features)
        builder = NewsIntelligenceFeatureBuilder()
        assert builder.recency_halflife_days == 3.0


class TestFeatureBuilderCryptoImport:
    """Verify the builder is importable from the crypto features sub-package."""

    def test_import_from_crypto_features(self) -> None:
        from spectraquant_v3.crypto.features import (
            NewsIntelligenceFeatureBuilder,
            build_daily_features,
        )

        assert callable(build_daily_features)
        builder = NewsIntelligenceFeatureBuilder()
        assert builder.recency_halflife_days == 3.0

    def test_same_class_both_asset_packages(self) -> None:
        from spectraquant_v3.crypto.features import (
            NewsIntelligenceFeatureBuilder as CryptoBuilder,
        )
        from spectraquant_v3.equities.features import (
            NewsIntelligenceFeatureBuilder as EquityBuilder,
        )

        assert CryptoBuilder is EquityBuilder
