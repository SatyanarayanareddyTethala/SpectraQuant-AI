"""Crypto symbol registry.

Maps canonical crypto symbols to their provider-specific representations.
This is the ONLY place where provider symbol formatting should happen.

Rules:
- Every downstream component uses canonical_symbol (e.g. BTC, ETH).
- Provider symbol resolution happens only here.
- Attempting to resolve an equity symbol (e.g. INFY.NS) raises
  SymbolResolutionError.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar

from spectraquant.core.enums import AssetClass
from spectraquant.core.errors import SymbolResolutionError

logger = logging.getLogger(__name__)

# Heuristics to detect equity symbols leaking into the crypto registry.
_EQUITY_SUFFIXES = frozenset([
    ".NS", ".BO", ".L", ".TO", ".AX", ".DE", ".PA", ".AS",
    ".HK", ".TW", ".KS", ".SI", ".JK",
])


def _looks_like_equity(symbol: str) -> bool:
    """Return True if the symbol looks like an equity ticker."""
    upper = symbol.upper()
    return any(upper.endswith(sfx) for sfx in _EQUITY_SUFFIXES)


@dataclass
class CryptoSymbolRecord:
    """Full symbol record for a single crypto asset."""

    canonical_symbol: str
    asset_class: AssetClass = AssetClass.CRYPTO
    coingecko_id: str = ""
    ccxt_binance_spot: str = ""   # e.g. BTC/USDT
    ccxt_coinbase: str = ""       # e.g. BTC-USD
    ccxt_kraken: str = ""         # e.g. XBT/USD
    binance_perp: str = ""        # e.g. BTCUSDT
    quote_currency: str = "USDT"
    market_type: str = "spot"
    is_tradable: bool = True
    primary_exchange_id: str = "binance"


class CryptoSymbolRegistry:
    """Registry for crypto canonical symbol → provider symbol mappings.

    Usage::

        registry = CryptoSymbolRegistry()
        binance_sym = registry.get_provider_symbol("BTC", "binance_spot")
        # → "BTC/USDT"

        # Raises SymbolResolutionError:
        registry.get_provider_symbol("INFY.NS", "binance_spot")
    """

    # Default built-in records; can be extended via register().
    _DEFAULTS: ClassVar[list[dict]] = [
        {
            "canonical_symbol": "BTC",
            "coingecko_id": "bitcoin",
            "ccxt_binance_spot": "BTC/USDT",
            "ccxt_coinbase": "BTC-USD",
            "ccxt_kraken": "XBT/USD",
            "binance_perp": "BTCUSDT",
        },
        {
            "canonical_symbol": "ETH",
            "coingecko_id": "ethereum",
            "ccxt_binance_spot": "ETH/USDT",
            "ccxt_coinbase": "ETH-USD",
            "ccxt_kraken": "ETH/USD",
            "binance_perp": "ETHUSDT",
        },
        {
            "canonical_symbol": "SOL",
            "coingecko_id": "solana",
            "ccxt_binance_spot": "SOL/USDT",
            "ccxt_coinbase": "SOL-USD",
            "ccxt_kraken": "SOL/USD",
            "binance_perp": "SOLUSDT",
        },
        {
            "canonical_symbol": "BNB",
            "coingecko_id": "binancecoin",
            "ccxt_binance_spot": "BNB/USDT",
            "ccxt_coinbase": "",
            "ccxt_kraken": "",
            "binance_perp": "BNBUSDT",
        },
        {
            "canonical_symbol": "XRP",
            "coingecko_id": "ripple",
            "ccxt_binance_spot": "XRP/USDT",
            "ccxt_coinbase": "XRP-USD",
            "ccxt_kraken": "XRP/USD",
            "binance_perp": "XRPUSDT",
        },
        {
            "canonical_symbol": "ADA",
            "coingecko_id": "cardano",
            "ccxt_binance_spot": "ADA/USDT",
            "ccxt_coinbase": "ADA-USD",
            "ccxt_kraken": "ADA/USD",
            "binance_perp": "ADAUSDT",
        },
        {
            "canonical_symbol": "DOGE",
            "coingecko_id": "dogecoin",
            "ccxt_binance_spot": "DOGE/USDT",
            "ccxt_coinbase": "DOGE-USD",
            "ccxt_kraken": "DOGE/USD",
            "binance_perp": "DOGEUSDT",
        },
        {
            "canonical_symbol": "AVAX",
            "coingecko_id": "avalanche-2",
            "ccxt_binance_spot": "AVAX/USDT",
            "ccxt_coinbase": "AVAX-USD",
            "ccxt_kraken": "AVAX/USD",
            "binance_perp": "AVAXUSDT",
        },
        {
            "canonical_symbol": "LINK",
            "coingecko_id": "chainlink",
            "ccxt_binance_spot": "LINK/USDT",
            "ccxt_coinbase": "LINK-USD",
            "ccxt_kraken": "LINK/USD",
            "binance_perp": "LINKUSDT",
        },
        {
            "canonical_symbol": "DOT",
            "coingecko_id": "polkadot",
            "ccxt_binance_spot": "DOT/USDT",
            "ccxt_coinbase": "DOT-USD",
            "ccxt_kraken": "DOT/USD",
            "binance_perp": "DOTUSDT",
        },
    ]

    def __init__(self) -> None:
        self._records: dict[str, CryptoSymbolRecord] = {}
        for d in self._DEFAULTS:
            rec = CryptoSymbolRecord(**d)
            self._records[rec.canonical_symbol.upper()] = rec

    def register(self, record: CryptoSymbolRecord) -> None:
        """Add or overwrite a symbol record."""
        if _looks_like_equity(record.canonical_symbol):
            raise SymbolResolutionError(
                record.canonical_symbol,
                expected_asset_class=AssetClass.CRYPTO,
                reason="Equity symbol cannot be registered in the crypto registry",
            )
        self._records[record.canonical_symbol.upper()] = record

    def get(self, canonical_symbol: str) -> CryptoSymbolRecord:
        """Return the symbol record for *canonical_symbol*.

        Raises:
            SymbolResolutionError: If the symbol is not found or looks like
                an equity symbol.
        """
        key = canonical_symbol.strip().upper()
        if _looks_like_equity(key):
            raise SymbolResolutionError(
                canonical_symbol,
                expected_asset_class=AssetClass.CRYPTO,
                actual_asset_class=AssetClass.EQUITY,
                reason="Equity symbols cannot be resolved in the crypto pipeline",
            )
        if key not in self._records:
            raise SymbolResolutionError(
                canonical_symbol,
                expected_asset_class=AssetClass.CRYPTO,
                reason=f"Symbol not found in crypto registry (known: {sorted(self._records)})",
            )
        return self._records[key]

    def get_provider_symbol(self, canonical_symbol: str, provider: str) -> str:
        """Return the provider-specific symbol string.

        Args:
            canonical_symbol: Canonical symbol, e.g. ``"BTC"``.
            provider: One of ``"binance_spot"``, ``"coinbase"``,
                ``"kraken"``, ``"binance_perp"``.

        Returns:
            Provider symbol string, e.g. ``"BTC/USDT"``.

        Raises:
            SymbolResolutionError: If the symbol or provider is unknown.
        """
        rec = self.get(canonical_symbol)
        mapping = {
            "binance_spot": rec.ccxt_binance_spot,
            "coinbase": rec.ccxt_coinbase,
            "kraken": rec.ccxt_kraken,
            "binance_perp": rec.binance_perp,
        }
        if provider not in mapping:
            raise SymbolResolutionError(
                canonical_symbol,
                expected_asset_class=AssetClass.CRYPTO,
                reason=f"Unknown provider {provider!r}. Valid: {sorted(mapping)}",
            )
        sym = mapping[provider]
        if not sym:
            raise SymbolResolutionError(
                canonical_symbol,
                expected_asset_class=AssetClass.CRYPTO,
                reason=f"No {provider} symbol mapping for {canonical_symbol!r}",
            )
        return sym

    def list_canonical(self) -> list[str]:
        """Return sorted list of all registered canonical symbols."""
        return sorted(self._records.keys())

    def validate_universe(self, symbols: list[str]) -> None:
        """Validate a list of canonical symbols against the registry.

        Raises:
            SymbolResolutionError: On the first unresolvable symbol.
        """
        for sym in symbols:
            self.get(sym)  # raises if invalid
