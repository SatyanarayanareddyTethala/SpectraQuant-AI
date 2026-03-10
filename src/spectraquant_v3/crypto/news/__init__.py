"""Crypto news toolkit for fetching, normalization, sentiment and persistence."""

from spectraquant_v3.crypto.news.news_fetcher import (
    CoinDeskRSSAdapter,
    CryptoPanicAdapter,
    fetch_articles,
)
from spectraquant_v3.crypto.news.news_features import build_news_features
from spectraquant_v3.crypto.news.news_normalizer import (
    CANONICAL_ARTICLE_FIELDS,
    validate_article_schema,
)
from spectraquant_v3.crypto.news.news_sentiment import (
    DeterministicSentimentScorer,
    LexiconSentimentModel,
)
from spectraquant_v3.crypto.news.news_store import NewsStore

__all__ = [
    "CANONICAL_ARTICLE_FIELDS",
    "CryptoPanicAdapter",
    "CoinDeskRSSAdapter",
    "DeterministicSentimentScorer",
    "LexiconSentimentModel",
    "NewsStore",
    "build_news_features",
    "fetch_articles",
    "validate_article_schema",
]
