"""Feature store package for SpectraQuant-AI-V3.

Provides persistent, versioned feature storage backed by Parquet files with
an optional DuckDB index for fast querying.

Public API::

    from spectraquant_v3.feature_store import FeatureStore, FeatureSetMetadata
"""

from spectraquant_v3.feature_store.store import FeatureStore
from spectraquant_v3.feature_store.metadata import FeatureSetMetadata

__all__ = ["FeatureStore", "FeatureSetMetadata"]
