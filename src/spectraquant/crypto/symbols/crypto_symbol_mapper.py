"""Crypto symbol mapper.

Provides a high-level façade over CryptoSymbolRegistry for use in
ingestion, signal, and execution layers.

Rules:
- Only canonical symbols flow through the system.
- Provider-specific symbols are derived here and nowhere else.
- Attempting to map an equity symbol raises SymbolResolutionError.
"""
from __future__ import annotations

import logging
from typing import Sequence

from spectraquant.core.enums import AssetClass
from spectraquant.core.errors import AssetClassLeakError, SymbolResolutionError
from spectraquant.crypto.symbols.crypto_symbol_registry import (
    CryptoSymbolRecord,
    CryptoSymbolRegistry,
)

logger = logging.getLogger(__name__)

# Heuristics – suffixes that indicate equity tickers
_EQUITY_SUFFIXES = frozenset([
    ".NS", ".BO", ".L", ".TO", ".AX", ".DE", ".PA", ".AS",
    ".HK", ".TW", ".KS", ".SI", ".JK",
])


def _looks_like_equity(symbol: str) -> bool:
    upper = symbol.upper()
    return any(upper.endswith(sfx) for sfx in _EQUITY_SUFFIXES)


class CryptoSymbolMapper:
    """Maps canonical crypto symbols to provider-specific formats.

    This is the primary interface for all crypto pipeline components
    that need provider-specific symbol strings.

    Example::

        mapper = CryptoSymbolMapper()
        mapper.to_binance_spot("BTC")   # "BTC/USDT"
        mapper.to_coinbase("ETH")       # "ETH-USD"
        mapper.to_binance_perp("SOL")   # "SOLUSDT"

        # Raises SymbolResolutionError:
        mapper.to_binance_spot("INFY.NS")
    """

    def __init__(self, registry: CryptoSymbolRegistry | None = None) -> None:
        self._registry = registry or CryptoSymbolRegistry()

    # ------------------------------------------------------------------
    # Per-provider helpers
    # ------------------------------------------------------------------

    def to_binance_spot(self, canonical: str) -> str:
        """Convert canonical symbol to Binance spot format (e.g. BTC/USDT)."""
        return self._registry.get_provider_symbol(canonical, "binance_spot")

    def to_coinbase(self, canonical: str) -> str:
        """Convert canonical symbol to Coinbase format (e.g. BTC-USD)."""
        return self._registry.get_provider_symbol(canonical, "coinbase")

    def to_kraken(self, canonical: str) -> str:
        """Convert canonical symbol to Kraken format (e.g. XBT/USD)."""
        return self._registry.get_provider_symbol(canonical, "kraken")

    def to_binance_perp(self, canonical: str) -> str:
        """Convert canonical symbol to Binance perpetual format (e.g. BTCUSDT)."""
        return self._registry.get_provider_symbol(canonical, "binance_perp")

    def to_coingecko_id(self, canonical: str) -> str:
        """Return the CoinGecko ID for *canonical* (e.g. 'bitcoin')."""
        rec = self._registry.get(canonical)
        if not rec.coingecko_id:
            raise SymbolResolutionError(
                canonical,
                expected_asset_class=AssetClass.CRYPTO,
                reason="No CoinGecko ID mapping available",
            )
        return rec.coingecko_id

    def get_record(self, canonical: str) -> CryptoSymbolRecord:
        """Return the full symbol record."""
        return self._registry.get(canonical)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def validate_no_equity_leak(self, symbols: Sequence[str]) -> None:
        """Raise AssetClassLeakError if any equity symbols are in *symbols*.

        This is the hard guard called at the start of every crypto pipeline
        invocation to prevent equity symbols from leaking in.
        """
        leaked = [s for s in symbols if _looks_like_equity(s)]
        if leaked:
            raise AssetClassLeakError(
                contaminating_symbols=leaked,
                pipeline_asset_class=AssetClass.CRYPTO,
                contaminating_asset_class=AssetClass.EQUITY,
            )

    def map_to_binance_spot(self, symbols: Sequence[str]) -> dict[str, str]:
        """Return {canonical → binance_spot_symbol} for all *symbols*."""
        self.validate_no_equity_leak(symbols)
        result: dict[str, str] = {}
        for sym in symbols:
            try:
                result[sym] = self.to_binance_spot(sym)
            except SymbolResolutionError:
                logger.warning("No Binance spot mapping for %r – skipping", sym)
        return result

    def list_canonical(self) -> list[str]:
        """Return all registered canonical crypto symbols."""
        return self._registry.list_canonical()
