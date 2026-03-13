"""Optional semantic embedding and clustering for news articles.

Uses sentence-transformers when available; falls back to TF-IDF.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Compute dense embeddings for a list of texts.

    Tries sentence-transformers first, falls back to TF-IDF.

    Parameters
    ----------
    texts : list of str
        Texts to embed.
    model_name : str
        Sentence-transformer model name (used only if library is available).

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(texts), dim)``.
    """
    if not texts:
        return np.empty((0, 0))

    try:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=False)
        logger.info("Computed %d embeddings via sentence-transformers", len(texts))
        return np.asarray(embeddings)
    except ImportError:
        logger.info("sentence-transformers not available; using TF-IDF fallback")
        return _tfidf_embeddings(texts)


def _tfidf_embeddings(texts: list[str]) -> np.ndarray:
    """Lightweight TF-IDF vectorization as embedding fallback."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray()


def cluster_articles(
    embeddings: np.ndarray,
    n_clusters: int = 5,
) -> np.ndarray:
    """Cluster embedding vectors via KMeans.

    Parameters
    ----------
    embeddings : np.ndarray
        2-D embeddings array.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Cluster labels of shape ``(n_samples,)``.
    """
    if embeddings.size == 0:
        return np.array([], dtype=int)
    from sklearn.cluster import KMeans

    n_clusters = min(n_clusters, len(embeddings))
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    return km.fit_predict(embeddings)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity."""
    if embeddings.size == 0:
        return np.empty((0, 0))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    return normed @ normed.T
