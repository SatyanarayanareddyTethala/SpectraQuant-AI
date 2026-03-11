"""Equity feature engineering sub-package for SpectraQuant-AI-V3.

Exports both the OHLCV-based technical feature engine and the
news-intelligence feature builder so that equity pipelines and backtests
can access everything from a single import path.
"""

from spectraquant_v3.equities.features.engine import EquityFeatureEngine, compute_features
from spectraquant_v3.core.news_intel_features import (
    NewsIntelligenceFeatureBuilder,
    build_daily_features,
)

__all__ = [
    "EquityFeatureEngine",
    "compute_features",
    "NewsIntelligenceFeatureBuilder",
    "build_daily_features",
]
