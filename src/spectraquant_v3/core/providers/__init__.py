"""Shared news intelligence providers for SpectraQuant-AI-V3.

This sub-package contains asset-class-agnostic news intelligence adapters
that populate :class:`~spectraquant_v3.core.news_schema.NewsIntelligenceRecord`
instances.  Individual adapters are thin wrappers — they never write to
cache or make run-mode decisions.
"""

from spectraquant_v3.core.providers.perplexity_provider import PerplexityNewsProvider

__all__ = ["PerplexityNewsProvider"]
