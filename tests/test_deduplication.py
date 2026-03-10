"""Tests for article deduplication module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spectraquant.nlp.deduplication import ArticleDeduplicator, deduplicate_news_articles


def test_compute_content_hash() -> None:
    """Test content hash computation."""
    dedup = ArticleDeduplicator()
    
    text1 = "This is a test article"
    text2 = "This is a test article"
    text3 = "This is a different article"
    
    hash1 = dedup.compute_content_hash(text1)
    hash2 = dedup.compute_content_hash(text2)
    hash3 = dedup.compute_content_hash(text3)
    
    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64


def test_compute_content_hash_normalization() -> None:
    """Test that content hash normalizes whitespace and case."""
    dedup = ArticleDeduplicator()
    
    text1 = "Test Article"
    text2 = "  test article  "
    text3 = "TEST ARTICLE"
    
    hash1 = dedup.compute_content_hash(text1)
    hash2 = dedup.compute_content_hash(text2)
    hash3 = dedup.compute_content_hash(text3)
    
    assert hash1 == hash2 == hash3


@patch("spectraquant.nlp.deduplication.SentenceTransformer")
def test_compute_embedding(mock_transformer: MagicMock) -> None:
    """Test embedding computation."""
    mock_model = MagicMock()
    mock_embedding = np.array([0.1, 0.2, 0.3])
    mock_model.encode.return_value = mock_embedding
    mock_transformer.return_value = mock_model
    
    dedup = ArticleDeduplicator()
    embedding = dedup.compute_embedding("test text")
    
    assert np.array_equal(embedding, mock_embedding)
    mock_model.encode.assert_called_once()


@patch("spectraquant.nlp.deduplication.SentenceTransformer")
def test_compute_embedding_caching(mock_transformer: MagicMock) -> None:
    """Test that embeddings are cached."""
    mock_model = MagicMock()
    mock_embedding = np.array([0.1, 0.2, 0.3])
    mock_model.encode.return_value = mock_embedding
    mock_transformer.return_value = mock_model
    
    dedup = ArticleDeduplicator()
    
    embedding1 = dedup.compute_embedding("test text", use_cache=True)
    embedding2 = dedup.compute_embedding("test text", use_cache=True)
    
    assert np.array_equal(embedding1, embedding2)
    assert mock_model.encode.call_count == 1


@patch("spectraquant.nlp.deduplication.SentenceTransformer")
def test_compute_similarity(mock_transformer: MagicMock) -> None:
    """Test similarity computation."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    mock_transformer.return_value = mock_model
    
    dedup = ArticleDeduplicator()
    similarity = dedup.compute_similarity("text1", "text2")
    
    assert similarity == pytest.approx(1.0)


@patch("spectraquant.nlp.deduplication.SentenceTransformer")
def test_is_near_duplicate(mock_transformer: MagicMock) -> None:
    """Test near-duplicate detection."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.9, 0.1, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    mock_transformer.return_value = mock_model
    
    dedup = ArticleDeduplicator(similarity_threshold=0.85)
    
    is_dup1 = dedup.is_near_duplicate("text1", "text2")
    assert is_dup1 is True
    
    is_dup2 = dedup.is_near_duplicate("text1", "text3")
    assert is_dup2 is False


def test_find_duplicates_exact_hash() -> None:
    """Test finding exact hash duplicates."""
    articles = [
        {"text": "Article one"},
        {"text": "Article two"},
        {"text": "Article one"},
    ]
    
    dedup = ArticleDeduplicator()
    with patch.object(dedup, "is_near_duplicate", return_value=False):
        duplicates = dedup.find_duplicates(articles)
    
    assert duplicates[0]["is_duplicate"] is False
    assert duplicates[1]["is_duplicate"] is False
    assert duplicates[2]["is_duplicate"] is True
    assert duplicates[2]["dedupe_reason"] == "exact_hash"
    assert duplicates[2]["similar_to"] == 0


@patch("spectraquant.nlp.deduplication.SentenceTransformer")
def test_find_duplicates_near_duplicate(mock_transformer: MagicMock) -> None:
    """Test finding near-duplicates."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.9, 0.1, 0.0]),
    ]
    mock_transformer.return_value = mock_model
    
    articles = [
        {"text": "Apple announces new iPhone"},
        {"text": "Apple announces iPhone"},
    ]
    
    dedup = ArticleDeduplicator(similarity_threshold=0.85)
    duplicates = dedup.find_duplicates(articles)
    
    assert duplicates[0]["is_duplicate"] is False
    assert duplicates[1]["is_duplicate"] is True
    assert duplicates[1]["dedupe_reason"] == "near_duplicate"
    assert duplicates[1]["similar_to"] == 0


def test_find_duplicates_clustering() -> None:
    """Test that duplicates are assigned to same cluster."""
    articles = [
        {"text": "Article one"},
        {"text": "Article one"},
        {"text": "Article one"},
    ]
    
    dedup = ArticleDeduplicator()
    with patch.object(dedup, "is_near_duplicate", return_value=False):
        duplicates = dedup.find_duplicates(articles)
    
    cluster_ids = [duplicates[i]["cluster_id"] for i in range(3)]
    assert cluster_ids[0] == cluster_ids[1] == cluster_ids[2]


def test_deduplicate_articles() -> None:
    """Test article deduplication."""
    articles = [
        {"text": "Unique article 1"},
        {"text": "Unique article 2"},
        {"text": "Unique article 1"},
        {"text": "Unique article 3"},
    ]
    
    dedup = ArticleDeduplicator()
    with patch.object(dedup, "is_near_duplicate", return_value=False):
        unique = dedup.deduplicate_articles(articles)
    
    assert len(unique) == 3
    assert unique[0]["text"] == "Unique article 1"
    assert unique[1]["text"] == "Unique article 2"
    assert unique[2]["text"] == "Unique article 3"


def test_deduplicate_articles_with_metadata() -> None:
    """Test that deduplication adds metadata."""
    articles = [
        {"text": "Article one"},
        {"text": "Article two"},
    ]
    
    dedup = ArticleDeduplicator()
    with patch.object(dedup, "is_near_duplicate", return_value=False):
        unique = dedup.deduplicate_articles(articles, add_metadata=True)
    
    assert len(unique) == 2
    assert "dedupe_reason" in unique[0]
    assert "cluster_id" in unique[0]
    assert "content_hash" in unique[0]
    assert unique[0]["dedupe_reason"] is None
    assert isinstance(unique[0]["cluster_id"], int)


def test_deduplicate_articles_without_metadata() -> None:
    """Test deduplication without adding metadata."""
    articles = [
        {"text": "Article one"},
        {"text": "Article two"},
    ]
    
    dedup = ArticleDeduplicator()
    with patch.object(dedup, "is_near_duplicate", return_value=False):
        unique = dedup.deduplicate_articles(articles, add_metadata=False)
    
    assert len(unique) == 2
    assert "dedupe_reason" not in unique[0]
    assert "cluster_id" not in unique[0]


def test_clear_cache() -> None:
    """Test clearing embedding cache."""
    dedup = ArticleDeduplicator()
    dedup._embedding_cache["test"] = np.array([1.0, 2.0])
    
    assert len(dedup._embedding_cache) == 1
    dedup.clear_cache()
    assert len(dedup._embedding_cache) == 0


@patch("spectraquant.nlp.deduplication.ArticleDeduplicator")
def test_deduplicate_news_articles_convenience_function(mock_dedup_class: MagicMock) -> None:
    """Test convenience function for deduplication."""
    mock_dedup = MagicMock()
    mock_dedup.deduplicate_articles.return_value = [{"text": "Article"}]
    mock_dedup_class.return_value = mock_dedup
    
    articles = [{"text": "Article"}]
    result = deduplicate_news_articles(articles, similarity_threshold=0.9)
    
    mock_dedup_class.assert_called_once_with(similarity_threshold=0.9)
    mock_dedup.deduplicate_articles.assert_called_once_with(articles, text_field="text")
    assert result == [{"text": "Article"}]


def test_find_duplicates_empty_articles() -> None:
    """Test handling of empty article list."""
    dedup = ArticleDeduplicator()
    duplicates = dedup.find_duplicates([])
    
    assert duplicates == {}


def test_find_duplicates_with_empty_text() -> None:
    """Test handling of articles with empty text."""
    articles = [
        {"text": ""},
        {"text": "Article"},
    ]
    
    dedup = ArticleDeduplicator()
    with patch.object(dedup, "is_near_duplicate", return_value=False):
        duplicates = dedup.find_duplicates(articles)
    
    assert 0 not in duplicates
    assert duplicates[1]["is_duplicate"] is False
