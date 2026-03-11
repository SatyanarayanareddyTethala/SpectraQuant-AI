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
