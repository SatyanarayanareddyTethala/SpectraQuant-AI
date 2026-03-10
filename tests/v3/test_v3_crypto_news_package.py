from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spectraquant_v3.crypto.news.news_fetcher import CoinDeskRSSAdapter, CryptoPanicAdapter, fetch_articles
from spectraquant_v3.crypto.news.news_normalizer import CANONICAL_ARTICLE_FIELDS, normalize_article_payload, validate_article_schema
from spectraquant_v3.crypto.news.news_sentiment import DeterministicSentimentScorer
from spectraquant_v3.crypto.news.news_store import NewsStore


class _FakeCryptoPanicProvider:
    def __init__(self) -> None:
        self.calls = 0

    def _get(self, path: str, params: dict):
        assert path == "/posts/"
        self.calls += 1
        page = params["page"]
        if page == 1:
            return {
                "results": [
                    {"id": "2", "title": "BTC up after ETF", "url": "https://x/2", "published_at": "2024-01-01T02:00:00Z", "currencies": [{"code": "BTC"}]},
                    {"id": "1", "title": "ETH drop after hack", "url": "https://x/1", "published_at": "2024-01-01T01:00:00Z", "currencies": [{"code": "ETH"}]},
                ],
                "next": "page=2",
            }
        return {
            "results": [
                {"id": "3", "title": "SOL listing rumor", "url": "https://x/3", "published_at": "2024-01-01T03:00:00Z", "currencies": [{"code": "SOL"}]}
            ],
            "next": None,
        }


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


class _FakeRSSSession:
    def __init__(self, xml_text: str) -> None:
        self.xml_text = xml_text

    def get(self, url: str, timeout: int = 20):
        assert "coindesk" in url
        return _FakeResponse(self.xml_text)


def test_cryptopanic_adapter_contract_and_deterministic_ordering() -> None:
    provider = _FakeCryptoPanicProvider()
    adapter = CryptoPanicAdapter(provider=provider)
    rows = fetch_articles(adapter, max_pages=2)
    assert [r["id"] for r in rows] == ["3", "2", "1"]


def test_coindesk_rss_adapter_contract() -> None:
    xml = """
    <rss><channel>
      <item><guid>g1</guid><title>First</title><link>https://a/1</link><pubDate>2024-01-01T00:00:00Z</pubDate></item>
      <item><guid>g2</guid><title>Second</title><link>https://a/2</link><pubDate>2024-01-02T00:00:00Z</pubDate></item>
    </channel></rss>
    """
    adapter = CoinDeskRSSAdapter(session=_FakeRSSSession(xml))
    rows = fetch_articles(adapter)
    assert [r["guid"] for r in rows] == ["g2", "g1"]


def test_schema_validation_and_normalization() -> None:
    scorer = DeterministicSentimentScorer()
    payload = {
        "id": "42",
        "title": "BTC bullish rally after ETF approval",
        "url": "https://news/42",
        "published_at": "2024-05-01T10:00:00Z",
        "currencies": [{"code": "BTC"}],
    }
    article = normalize_article_payload(payload, source_name="cryptopanic", sentiment_score=scorer.score(payload["title"]))
    assert tuple(article.keys()) == CANONICAL_ARTICLE_FIELDS
    assert article["event_type"] == "etf"
    validate_article_schema(article)


def test_schema_validation_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        validate_article_schema({"article_id": "x"})


def test_news_store_idempotent_dedupe(tmp_path: Path) -> None:
    store = NewsStore(tmp_path)
    rows = [
        {
            "article_id": "a1",
            "published_at": "2024-01-01T00:00:00+00:00",
            "title": "t",
            "source": "s",
            "url": "https://u/1",
            "mentioned_symbols": ["BTC"],
            "sentiment_score": 0.1,
            "event_type": "general",
            "relevance_score": 0.5,
        },
        {
            "article_id": "a1",
            "published_at": "2024-01-01T00:00:00+00:00",
            "title": "t",
            "source": "s",
            "url": "https://u/1",
            "mentioned_symbols": ["BTC"],
            "sentiment_score": 0.1,
            "event_type": "general",
            "relevance_score": 0.5,
        },
    ]
    jsonl_path = store.write_jsonl("BTC", rows)
    store.write_jsonl("BTC", rows)
    assert len(jsonl_path.read_text().strip().splitlines()) == 1

    parquet_path = store.write_parquet("BTC", rows)
    store.write_parquet("BTC", rows)
    df = pd.read_parquet(parquet_path)
    assert len(df) == 1
