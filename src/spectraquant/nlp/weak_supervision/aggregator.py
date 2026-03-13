"""Probabilistic label aggregation for weak supervision."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class LabelAggregator:
    """Aggregates labels from multiple labeling functions."""

    def __init__(self, method: str = "majority_vote") -> None:
        """Initialize label aggregator."""
        self.method = method
    
    def aggregate(
        self,
        label_matrix: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate labels."""
        if self.method == "majority_vote":
            return self._majority_vote(label_matrix)
        raise ValueError(f"Unknown method: {self.method}")
    
    def _majority_vote(self, label_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simple majority vote."""
        n_samples = label_matrix.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        confidence = np.zeros(n_samples)
        
        for i in range(n_samples):
            votes = label_matrix[i][label_matrix[i] >= 0]
            if len(votes) > 0:
                unique, counts = np.unique(votes, return_counts=True)
                labels[i] = unique[np.argmax(counts)]
                confidence[i] = np.max(counts) / len(votes)
        
        return labels, confidence
