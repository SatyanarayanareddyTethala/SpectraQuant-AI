"""Perplexity news intelligence provider — equity ingestion adapter.

Thin re-export so that the equities ingestion layer can import the shared
:class:`~spectraquant_v3.core.providers.perplexity_provider.PerplexityNewsProvider`
through its own provider namespace.

This module must NEVER import from ``spectraquant_v3.crypto``.
"""

from spectraquant_v3.core.providers.perplexity_provider import PerplexityNewsProvider

__all__ = ["PerplexityNewsProvider"]
