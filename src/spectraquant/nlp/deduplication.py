"""Deduplication module for news articles using exact hashes and embeddings."""
from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class ArticleDeduplicator:
    """Deduplicates news articles using exact hash and near-duplicate detection."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the deduplicator.
        
        Args:
            similarity_threshold: Cosine similarity threshold for near-duplicates
            model_name: Name of the sentence transformer model to use
        """
        self.similarity_threshold = similarity_threshold
        self._model: Any | None = None
        self._model_name = model_name
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _get_model(self) -> "SentenceTransformer":
        """Lazy load the sentence transformer model."""
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is required for near-duplicate detection. "
                    "Install it with: pip install sentence-transformers"
                )
            logger.info("Loading sentence transformer model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def compute_content_hash(self, text: str) -> str:
        """Compute SHA256 hash of normalized text.
        
        Args:
            text: Input text to hash
            
        Returns:
            Hex string of SHA256 hash
        """
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def compute_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Compute sentence embedding for text.
        
        Args:
            text: Input text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector as numpy array
        """
        if use_cache and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        
        if use_cache:
            self._embedding_cache[text] = embedding
        
        return embedding

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        emb1 = self.compute_embedding(text1)
        emb2 = self.compute_embedding(text2)
        
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cosine_sim)

    def is_near_duplicate(self, text1: str, text2: str) -> bool:
        """Check if two texts are near-duplicates.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if texts are near-duplicates, False otherwise
        """
        similarity = self.compute_similarity(text1, text2)
        return similarity >= self.similarity_threshold

    def find_duplicates(
        self,
        articles: list[dict[str, Any]],
        text_field: str = "text",
    ) -> dict[int, dict[str, Any]]:
        """Find duplicates and near-duplicates in a list of articles.
        
        Args:
            articles: List of article dictionaries
            text_field: Name of the field containing article text
            
        Returns:
            Dictionary mapping article index to deduplication info:
            {
                'is_duplicate': bool,
                'dedupe_reason': str ('exact_hash' or 'near_duplicate'),
                'cluster_id': int,
                'similar_to': int (index of similar article),
            }
        """
        if not articles:
            return {}
        
        result: dict[int, dict[str, Any]] = {}
        hash_to_idx: dict[str, int] = {}
        cluster_counter = 0
        idx_to_cluster: dict[int, int] = {}
        
        for idx, article in enumerate(articles):
            text = article.get(text_field, "")
            if not text:
                continue
            
            content_hash = self.compute_content_hash(text)
            
            if content_hash in hash_to_idx:
                original_idx = hash_to_idx[content_hash]
                cluster_id = idx_to_cluster.get(original_idx, cluster_counter)
                if original_idx not in idx_to_cluster:
                    idx_to_cluster[original_idx] = cluster_id
                    cluster_counter += 1
                
                idx_to_cluster[idx] = cluster_id
                result[idx] = {
                    "is_duplicate": True,
                    "dedupe_reason": "exact_hash",
                    "cluster_id": cluster_id,
                    "similar_to": original_idx,
                }
                continue
            
            hash_to_idx[content_hash] = idx
            
            is_near_dup = False
            for prev_idx in range(idx):
                if prev_idx in result and result[prev_idx]["is_duplicate"]:
                    continue
                
                prev_text = articles[prev_idx].get(text_field, "")
                if not prev_text:
                    continue
                
                if self.is_near_duplicate(text, prev_text):
                    cluster_id = idx_to_cluster.get(prev_idx, cluster_counter)
                    if prev_idx not in idx_to_cluster:
                        idx_to_cluster[prev_idx] = cluster_id
                        cluster_counter += 1
                    
                    idx_to_cluster[idx] = cluster_id
                    result[idx] = {
                        "is_duplicate": True,
                        "dedupe_reason": "near_duplicate",
                        "cluster_id": cluster_id,
                        "similar_to": prev_idx,
                    }
                    is_near_dup = True
                    break
            
            if not is_near_dup:
                cluster_id = cluster_counter
                idx_to_cluster[idx] = cluster_id
                cluster_counter += 1
                result[idx] = {
                    "is_duplicate": False,
                    "dedupe_reason": None,
                    "cluster_id": cluster_id,
                    "similar_to": None,
                }
        
        return result

    def deduplicate_articles(
        self,
        articles: list[dict[str, Any]],
        text_field: str = "text",
        add_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Remove duplicates from article list and optionally add metadata.
        
        Args:
            articles: List of article dictionaries
            text_field: Name of the field containing article text
            add_metadata: Whether to add deduplication metadata to articles
            
        Returns:
            List of unique articles
        """
        duplicates_info = self.find_duplicates(articles, text_field)
        
        unique_articles = []
        for idx, article in enumerate(articles):
            info = duplicates_info.get(idx, {})
            
            if add_metadata:
                article = dict(article)
                article["dedupe_reason"] = info.get("dedupe_reason")
                article["cluster_id"] = info.get("cluster_id")
                article["content_hash"] = self.compute_content_hash(
                    article.get(text_field, "")
                )
            
            if not info.get("is_duplicate", False):
                unique_articles.append(article)
        
        logger.info(
            "Deduplication: %d unique articles from %d total (%.1f%% duplicates)",
            len(unique_articles),
            len(articles),
            100 * (len(articles) - len(unique_articles)) / max(len(articles), 1),
        )
        
        return unique_articles

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()


def deduplicate_news_articles(
    articles: list[dict[str, Any]],
    text_field: str = "text",
    similarity_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Convenience function to deduplicate a list of news articles.
    
    Args:
        articles: List of article dictionaries
        text_field: Name of the field containing article text
        similarity_threshold: Cosine similarity threshold for near-duplicates
        
    Returns:
        List of unique articles with deduplication metadata
    """
    deduplicator = ArticleDeduplicator(similarity_threshold=similarity_threshold)
    return deduplicator.deduplicate_articles(articles, text_field=text_field)
