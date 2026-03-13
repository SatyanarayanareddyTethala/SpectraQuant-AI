"""Equity symbol registry.

Maps canonical equity symbols to provider-specific representations.
This is the ONLY place where provider symbol formatting should happen
for equity instruments.

Rules:
- Canonical symbols are ticker-only (e.g. INFY, TCS, RELIANCE).
- yfinance symbols carry the exchange suffix (e.g. INFY.NS, TCS.NS).
- Attempting to resolve a crypto symbol (e.g. BTC, ETH) raises
  SymbolResolutionError with asset class context.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

from spectraquant.core.enums import AssetClass
from spectraquant.core.errors import SymbolResolutionError

logger = logging.getLogger(__name__)

# Crypto asset names – used to detect crypto symbol leaks
_KNOWN_CRYPTO_SYMBOLS = frozenset([
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
    "MATIC", "LTC", "BCH", "ATOM", "UNI", "XLM", "ALGO", "VET", "FIL", "TRX",
    "SHIB", "NEAR", "ICP", "APT", "ARB", "OP", "IMX", "SAND", "AXS", "MANA",
])

# Provider symbol suffixes that indicate equities
_EQUITY_SUFFIXES = frozenset([
    ".NS", ".BO", ".L", ".TO", ".AX", ".DE", ".PA", ".AS",
    ".HK", ".TW", ".KS", ".SI", ".JK",
])


def _looks_like_crypto(symbol: str) -> bool:
    """Return True if the symbol looks like a crypto asset."""
    return symbol.strip().upper() in _KNOWN_CRYPTO_SYMBOLS


@dataclass
class EquitySymbolRecord:
    """Full symbol record for a single equity instrument."""

    canonical_symbol: str           # e.g. INFY
    asset_class: AssetClass = AssetClass.EQUITY
    yfinance_symbol: str = ""       # e.g. INFY.NS
    exchange_symbol: str = ""       # exchange-native symbol
    isin: str = ""
    primary_exchange_id: str = ""   # e.g. NSE, BSE, LSE, NYSE
    quote_currency: str = "INR"
    market_type: str = "equity"
    is_tradable: bool = True
    sector: str = ""
    industry: str = ""


class EquitySymbolRegistry:
    """Registry for equity canonical symbol → provider symbol mappings.

    Usage::

        registry = EquitySymbolRegistry()
        yf_sym = registry.get_yfinance_symbol("INFY")
        # → "INFY.NS"

        # Raises SymbolResolutionError:
        registry.get_yfinance_symbol("BTC")
    """

    # Built-in records for common NSE blue-chips.
    _DEFAULTS: ClassVar[list[dict]] = [
        {"canonical_symbol": "RELIANCE", "yfinance_symbol": "RELIANCE.NS",
         "primary_exchange_id": "NSE", "quote_currency": "INR", "sector": "Energy"},
        {"canonical_symbol": "TCS", "yfinance_symbol": "TCS.NS",
         "primary_exchange_id": "NSE", "quote_currency": "INR", "sector": "IT"},
        {"canonical_symbol": "INFY", "yfinance_symbol": "INFY.NS",
         "primary_exchange_id": "NSE", "quote_currency": "INR", "sector": "IT"},
        {"canonical_symbol": "HDFCBANK", "yfinance_symbol": "HDFCBANK.NS",
         "primary_exchange_id": "NSE", "quote_currency": "INR", "sector": "Finance"},
        {"canonical_symbol": "ICICIBANK", "yfinance_symbol": "ICICIBANK.NS",
         "primary_exchange_id": "NSE", "quote_currency": "INR", "sector": "Finance"},
        {"canonical_symbol": "HINDUNILVR", "yfinance_symbol": "HINDUNILVR.NS",
         "primary_exchange_id": "NSE", "quote_currency": "INR", "sector": "FMCG"},
        {"canonical_symbol": "WIPRO", "yfinance_symbol": "WIPRO.NS",
         "primary_exchange_id": "NSE", "quote_currency": "INR", "sector": "IT"},
        {"canonical_symbol": "BHARTIARTL", "yfinance_symbol": "BHARTIARTL.NS",
         "primary_exchange_id": "NSE", "quote_currency": "INR", "sector": "Telecom"},
        {"canonical_symbol": "SBIN", "yfinance_symbol": "SBIN.NS",
         "primary_exchange_id": "NSE", "quote_currency": "INR", "sector": "Finance"},
        {"canonical_symbol": "KOTAKBANK", "yfinance_symbol": "KOTAKBANK.NS",
         "primary_exchange_id": "NSE", "quote_currency": "INR", "sector": "Finance"},
        # FTSE examples
        {"canonical_symbol": "LLOY", "yfinance_symbol": "LLOY.L",
         "primary_exchange_id": "LSE", "quote_currency": "GBP", "sector": "Finance"},
        {"canonical_symbol": "HSBA", "yfinance_symbol": "HSBA.L",
         "primary_exchange_id": "LSE", "quote_currency": "GBP", "sector": "Finance"},
        {"canonical_symbol": "BP", "yfinance_symbol": "BP.L",
         "primary_exchange_id": "LSE", "quote_currency": "GBP", "sector": "Energy"},
        {"canonical_symbol": "VOD", "yfinance_symbol": "VOD.L",
         "primary_exchange_id": "LSE", "quote_currency": "GBP", "sector": "Telecom"},
        {"canonical_symbol": "GSK", "yfinance_symbol": "GSK.L",
         "primary_exchange_id": "LSE", "quote_currency": "GBP", "sector": "Healthcare"},
    ]

    def __init__(self) -> None:
        self._records: dict[str, EquitySymbolRecord] = {}
        for d in self._DEFAULTS:
            rec = EquitySymbolRecord(**d)
            self._records[rec.canonical_symbol.upper()] = rec

    def register(self, record: EquitySymbolRecord) -> None:
        """Add or overwrite a symbol record."""
        if _looks_like_crypto(record.canonical_symbol):
            raise SymbolResolutionError(
                record.canonical_symbol,
                expected_asset_class=AssetClass.EQUITY,
                reason="Crypto symbol cannot be registered in the equity registry",
            )
        self._records[record.canonical_symbol.upper()] = record

    def register_from_yfinance_symbol(self, yfinance_symbol: str) -> None:
        """Register a record derived from a yfinance symbol (e.g. 'INFY.NS').

        The canonical symbol is derived by stripping the exchange suffix.
        """
        parts = yfinance_symbol.rsplit(".", 1)
        canonical = parts[0].upper()
        exchange_suffix = f".{parts[1]}" if len(parts) > 1 else ""
        # Detect exchange
        exchange_map = {
            ".NS": "NSE", ".BO": "BSE", ".L": "LSE",
            ".TO": "TSX", ".AX": "ASX",
        }
        exchange = exchange_map.get(exchange_suffix.upper(), "UNKNOWN")
        currency_map = {
            ".NS": "INR", ".BO": "INR", ".L": "GBP",
            ".TO": "CAD", ".AX": "AUD",
        }
        currency = currency_map.get(exchange_suffix.upper(), "USD")
        rec = EquitySymbolRecord(
            canonical_symbol=canonical,
            yfinance_symbol=yfinance_symbol,
            primary_exchange_id=exchange,
            quote_currency=currency,
        )
        self._records[canonical] = rec

    def get(self, canonical_symbol: str) -> EquitySymbolRecord:
        """Return the symbol record for *canonical_symbol*.

        Raises:
            SymbolResolutionError: If the symbol is not found or looks like
                a crypto symbol.
        """
        key = canonical_symbol.strip().upper()
        if _looks_like_crypto(key):
            raise SymbolResolutionError(
                canonical_symbol,
                expected_asset_class=AssetClass.EQUITY,
                actual_asset_class=AssetClass.CRYPTO,
                reason="Crypto symbols cannot be resolved in the equity pipeline",
            )
        if key not in self._records:
            raise SymbolResolutionError(
                canonical_symbol,
                expected_asset_class=AssetClass.EQUITY,
                reason=f"Symbol not found in equity registry (known: {sorted(self._records)})",
            )
        return self._records[key]

    def get_yfinance_symbol(self, canonical_symbol: str) -> str:
        """Return the yfinance symbol for *canonical_symbol*.

        Raises:
            SymbolResolutionError: If the symbol is unknown or has no yfinance mapping.
        """
        rec = self.get(canonical_symbol)
        if not rec.yfinance_symbol:
            raise SymbolResolutionError(
                canonical_symbol,
                expected_asset_class=AssetClass.EQUITY,
                reason="No yfinance symbol mapping available",
            )
        return rec.yfinance_symbol

    def list_canonical(self) -> list[str]:
        """Return sorted list of all registered canonical symbols."""
        return sorted(self._records.keys())

    def validate_universe(self, symbols: list[str]) -> None:
        """Validate a list of canonical symbols.

        Raises:
            SymbolResolutionError: On the first unresolvable symbol.
        """
        for sym in symbols:
            self.get(sym)  # raises if invalid
