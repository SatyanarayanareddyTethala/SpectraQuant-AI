"""Equity symbol mapper.

High-level façade over EquitySymbolRegistry for use in ingestion,
signal, and execution layers.

Rules:
- Only canonical symbols flow through the system.
- Provider symbols (yfinance) are derived here and nowhere else.
- Attempting to map a crypto symbol raises AssetClassLeakError.
"""
from __future__ import annotations

import logging
from typing import Sequence

from spectraquant.core.enums import AssetClass
from spectraquant.core.errors import AssetClassLeakError, SymbolResolutionError
from spectraquant.equities.symbols.equity_symbol_registry import (
    EquitySymbolRecord,
    EquitySymbolRegistry,
)

logger = logging.getLogger(__name__)

_KNOWN_CRYPTO_SYMBOLS = frozenset([
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
    "MATIC", "LTC", "BCH", "ATOM", "UNI", "XLM", "ALGO", "VET", "FIL", "TRX",
])


def _looks_like_crypto(symbol: str) -> bool:
    return symbol.strip().upper() in _KNOWN_CRYPTO_SYMBOLS


class EquitySymbolMapper:
    """Maps canonical equity symbols to provider-specific formats.

    This is the primary interface for all equity pipeline components
    that need provider-specific symbol strings.

    Example::

        mapper = EquitySymbolMapper()
        mapper.to_yfinance("INFY")      # "INFY.NS"
        mapper.to_yfinance("LLOY")      # "LLOY.L"

        # Raises AssetClassLeakError:
        mapper.to_yfinance("BTC")

    The mapper can also be bootstrapped from a list of yfinance-style
    symbols (e.g. ``["INFY.NS", "TCS.NS"]``) to auto-populate the registry.
    """

    def __init__(self, registry: EquitySymbolRegistry | None = None) -> None:
        self._registry = registry or EquitySymbolRegistry()

    # ------------------------------------------------------------------
    # Bootstrap helpers
    # ------------------------------------------------------------------

    def bootstrap_from_yfinance_symbols(self, yf_symbols: Sequence[str]) -> None:
        """Register canonical records derived from yfinance symbols.

        Symbols already present in the registry are left unchanged.

        Args:
            yf_symbols: List of yfinance-style symbols (e.g. ``["INFY.NS", "TCS.NS"]``).
        """
        for sym in yf_symbols:
            parts = sym.rsplit(".", 1)
            canonical = parts[0].upper()
            if canonical not in self._registry._records:
                try:
                    self._registry.register_from_yfinance_symbol(sym)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Could not register %r: %s", sym, exc)

    # ------------------------------------------------------------------
    # Per-provider helpers
    # ------------------------------------------------------------------

    def to_yfinance(self, canonical: str) -> str:
        """Return the yfinance symbol for *canonical* (e.g. 'INFY.NS')."""
        if _looks_like_crypto(canonical):
            raise AssetClassLeakError(
                contaminating_symbols=[canonical],
                pipeline_asset_class=AssetClass.EQUITY,
                contaminating_asset_class=AssetClass.CRYPTO,
            )
        return self._registry.get_yfinance_symbol(canonical)

    def get_record(self, canonical: str) -> EquitySymbolRecord:
        """Return the full symbol record for *canonical*."""
        return self._registry.get(canonical)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def validate_no_crypto_leak(self, symbols: Sequence[str]) -> None:
        """Raise AssetClassLeakError if any crypto symbols are in *symbols*.

        This is the hard guard called at the start of every equity pipeline
        invocation.
        """
        leaked = [s for s in symbols if _looks_like_crypto(s)]
        if leaked:
            raise AssetClassLeakError(
                contaminating_symbols=leaked,
                pipeline_asset_class=AssetClass.EQUITY,
                contaminating_asset_class=AssetClass.CRYPTO,
            )

    def map_to_yfinance(self, symbols: Sequence[str]) -> dict[str, str]:
        """Return {canonical → yfinance_symbol} for all *symbols*.

        Symbols that cannot be resolved are skipped with a warning.
        Crypto symbols raise AssetClassLeakError immediately.
        """
        self.validate_no_crypto_leak(symbols)
        result: dict[str, str] = {}
        for sym in symbols:
            try:
                result[sym] = self.to_yfinance(sym)
            except SymbolResolutionError as exc:
                logger.warning("No yfinance mapping for %r: %s – skipping", sym, exc)
        return result

    def list_canonical(self) -> list[str]:
        """Return all registered canonical equity symbols."""
        return self._registry.list_canonical()
