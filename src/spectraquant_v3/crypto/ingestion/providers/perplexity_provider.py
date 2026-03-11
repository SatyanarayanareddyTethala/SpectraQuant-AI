"""Perplexity news intelligence provider — crypto ingestion adapter.

Thin re-export so that the crypto ingestion layer can import the shared
:class:`~spectraquant_v3.core.providers.perplexity_provider.PerplexityNewsProvider`
through its own provider namespace.

This module must NEVER import from ``spectraquant_v3.equities``.
"""

from spectraquant_v3.core.providers.perplexity_provider import PerplexityNewsProvider

__all__ = ["PerplexityNewsProvider"]
