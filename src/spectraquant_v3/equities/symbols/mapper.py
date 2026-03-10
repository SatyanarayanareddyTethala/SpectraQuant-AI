"""Equity symbol mapper for SpectraQuant-AI-V3.

The mapper translates between canonical equity tickers and provider-specific
symbols.  It is the *only* place where symbol translation should occur.

Rules:
- Input and output canonical symbols must always be EQUITY.
- Attempting to map a crypto symbol raises :exc:`AssetClassLeakError`.
- A missing mapping raises :exc:`SymbolResolutionError`.
"""

from __future__ import annotations

from spectraquant_v3.core.enums import AssetClass
from spectraquant_v3.core.errors import AssetClassLeakError, SymbolResolutionError
from spectraquant_v3.core.schema import SymbolRecord
from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry, _is_crypto_symbol


class EquitySymbolMapper:
    """Translates canonical equity tickers to provider-specific symbols.

    Args:
        registry: The :class:`EquitySymbolRegistry` to resolve symbols from.
    """

    def __init__(self, registry: EquitySymbolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def _assert_not_crypto(self, symbol: str) -> None:
        """Raise :exc:`AssetClassLeakError` if *symbol* looks like a crypto pair."""
        if _is_crypto_symbol(symbol):
            raise AssetClassLeakError(
                f"Symbol '{symbol}' looks like a crypto pair. "
                "The equity mapper must not process crypto symbols."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_record(self, canonical_symbol: str) -> SymbolRecord:
        """Return the full :class:`SymbolRecord` for *canonical_symbol*.

        Raises:
            AssetClassLeakError: If the symbol looks like a crypto pair.
            SymbolResolutionError: If not registered.
        """
        self._assert_not_crypto(canonical_symbol)
        record = self._registry.get(canonical_symbol)
        if record.asset_class != AssetClass.EQUITY:
            raise AssetClassLeakError(
                f"Registry returned a non-equity record for '{canonical_symbol}'. "
                "Asset-class integrity violation."
            )
        return record

    def to_yfinance_symbol(self, canonical_symbol: str) -> str:
        """Return the yfinance-compatible ticker for *canonical_symbol*.

        For most equity symbols the canonical ticker *is* the yfinance ticker
        (e.g. ``"INFY.NS"``).  The record's ``yfinance_symbol`` attribute can
        override this when the canonical and provider forms differ.

        Args:
            canonical_symbol: Exchange-qualified ticker, e.g. ``"INFY.NS"``.

        Returns:
            yfinance-compatible ticker string.
        """
        self._assert_not_crypto(canonical_symbol)
        record = self._registry.get(canonical_symbol)
        return record.yfinance_symbol or canonical_symbol

    def to_provider_symbol(self, canonical_symbol: str) -> str:
        """Return the generic provider symbol for *canonical_symbol*."""
        self._assert_not_crypto(canonical_symbol)
        record = self._registry.get(canonical_symbol)
        return record.provider_symbol or canonical_symbol

    def from_yfinance_symbol(self, yf_symbol: str) -> str:
        """Reverse-map a yfinance ticker to its canonical ticker.

        Args:
            yf_symbol: yfinance-compatible ticker, e.g. ``"INFY.NS"``.

        Returns:
            Canonical ticker string.

        Raises:
            SymbolResolutionError: If no record has this yfinance symbol.
        """
        for record in self._registry.all_records():
            if record.yfinance_symbol == yf_symbol or record.canonical_symbol == yf_symbol:
                return record.canonical_symbol
        raise SymbolResolutionError(
            f"Cannot reverse-map yfinance symbol '{yf_symbol}' to a "
            "canonical equity ticker.  Ensure the symbol is registered."
        )

    def is_registered(self, canonical_symbol: str) -> bool:
        """Return True if *canonical_symbol* is in the registry."""
        return self._registry.contains(canonical_symbol)
