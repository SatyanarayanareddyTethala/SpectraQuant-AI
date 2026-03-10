"""Crypto symbol mapper for SpectraQuant-AI-V3.

The mapper translates between canonical symbols and provider-specific symbols.
It is the *only* place where symbol translation should occur.

Rules:
- Input and output canonical symbols must always be CRYPTO.
- Attempting to map an equity symbol raises :exc:`AssetClassLeakError`.
- A missing mapping raises :exc:`SymbolResolutionError` — never silently returns None.
"""

from __future__ import annotations

from spectraquant_v3.core.enums import AssetClass
from spectraquant_v3.core.errors import AssetClassLeakError, SymbolResolutionError
from spectraquant_v3.core.schema import SymbolRecord
from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry, _is_equity_symbol


class CryptoSymbolMapper:
    """Translates canonical crypto tickers to provider-specific symbols.

    Args:
        registry: The :class:`CryptoSymbolRegistry` to resolve symbols from.
    """

    def __init__(self, registry: CryptoSymbolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def _assert_not_equity(self, symbol: str) -> None:
        """Raise :exc:`AssetClassLeakError` if *symbol* looks like an equity."""
        if _is_equity_symbol(symbol):
            raise AssetClassLeakError(
                f"Symbol '{symbol}' looks like an equity ticker. "
                "The crypto mapper must not process equity symbols."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_record(self, canonical_symbol: str) -> SymbolRecord:
        """Return the full :class:`SymbolRecord` for *canonical_symbol*.

        Raises:
            AssetClassLeakError: If the symbol looks like an equity.
            SymbolResolutionError: If not registered.
        """
        self._assert_not_equity(canonical_symbol)
        record = self._registry.get(canonical_symbol)
        if record.asset_class != AssetClass.CRYPTO:
            raise AssetClassLeakError(
                f"Registry returned a non-crypto record for '{canonical_symbol}'. "
                "Asset-class integrity violation."
            )
        return record

    def to_provider_symbol(
        self,
        canonical_symbol: str,
        quote_currency: str = "USDT",
    ) -> str:
        """Translate a canonical ticker to the CCXT exchange pair format.

        Uses the stored ``provider_symbol`` from the registry if available.
        Falls back to constructing ``<TICKER>/<QUOTE>`` when the record has
        no explicit provider symbol.

        Args:
            canonical_symbol: Upper-case coin ticker, e.g. ``"BTC"``.
            quote_currency:   Quote currency used as fallback.

        Returns:
            Provider symbol string, e.g. ``"BTC/USDT"``.
        """
        self._assert_not_equity(canonical_symbol)
        record = self._registry.get(canonical_symbol)
        if record.provider_symbol:
            return record.provider_symbol
        return f"{canonical_symbol.upper()}/{quote_currency}"

    def to_exchange_symbol(self, canonical_symbol: str) -> str:
        """Return the exchange-specific symbol for *canonical_symbol*.

        Falls back to ``provider_symbol`` if ``exchange_symbol`` is empty.
        """
        self._assert_not_equity(canonical_symbol)
        record = self._registry.get(canonical_symbol)
        return record.exchange_symbol or record.provider_symbol or canonical_symbol.upper()

    def to_coingecko_id(self, canonical_symbol: str) -> str:
        """Return the CoinGecko ID for *canonical_symbol*.

        Falls back to lower-casing the ticker when ``coingecko_id`` is not set.
        """
        self._assert_not_equity(canonical_symbol)
        record = self._registry.get(canonical_symbol)
        return record.coingecko_id or canonical_symbol.lower()

    def from_provider_symbol(self, provider_symbol: str) -> str:
        """Reverse-map a provider symbol to its canonical ticker.

        Args:
            provider_symbol: CCXT-style pair, e.g. ``"BTC/USDT"``.

        Returns:
            Canonical ticker string, e.g. ``"BTC"``.

        Raises:
            SymbolResolutionError: If no record has this provider symbol.
        """
        for record in self._registry.all_records():
            if record.provider_symbol == provider_symbol:
                return record.canonical_symbol
        # Fallback: strip the quote currency
        if "/" in provider_symbol:
            base = provider_symbol.split("/")[0].upper()
            if self._registry.contains(base):
                return base
        raise SymbolResolutionError(
            f"Cannot reverse-map provider symbol '{provider_symbol}' to a "
            "canonical crypto ticker.  Ensure the symbol is registered."
        )

    def is_registered(self, canonical_symbol: str) -> bool:
        """Return True if *canonical_symbol* is in the registry."""
        return self._registry.contains(canonical_symbol)
