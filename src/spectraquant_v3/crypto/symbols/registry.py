"""Crypto symbol registry for SpectraQuant-AI-V3.

The registry is the *single source of truth* for every crypto symbol processed
by the V3 pipeline.  All downstream modules receive a :class:`SymbolRecord`
from the registry and must never hard-code provider-specific symbol strings.

Design rules:
- Every symbol stored here must have ``asset_class == AssetClass.CRYPTO``.
- Attempting to register an equity symbol raises :exc:`AssetClassLeakError`.
- ``canonical_symbol`` is the coin ticker in upper-case (e.g. ``"BTC"``).
- Provider-specific symbols (exchange pairs, CoinGecko IDs) are attributes of
  the record, not additional keys.
"""

from __future__ import annotations

from typing import Any

from spectraquant_v3.core.enums import AssetClass
from spectraquant_v3.core.errors import AssetClassLeakError, SymbolResolutionError
from spectraquant_v3.core.schema import SymbolRecord

# ---------------------------------------------------------------------------
# Equity-pattern heuristic used for leak detection
# ---------------------------------------------------------------------------
_EQUITY_SUFFIXES = (
    ".NS", ".BO", ".L", ".TO", ".AX", ".HK", ".SI", ".PA",
    ".DE", ".F", ".MC", ".MI", ".AS", ".BR", ".CO", ".OL",
    ".ST", ".HE", ".IS", ".LS", ".VI", ".AT",
)
_CRYPTO_QUOTE_CURRENCIES = frozenset(
    ["USDT", "USDC", "BTC", "ETH", "BNB", "BUSD", "DAI", "USD"]
)


def _is_equity_symbol(symbol: str) -> bool:
    """Heuristic: return True when *symbol* looks like an equity ticker."""
    upper = symbol.upper()
    return any(upper.endswith(suffix.upper()) for suffix in _EQUITY_SUFFIXES)


class CryptoSymbolRegistry:
    """In-memory registry of canonical crypto symbols.

    Usage::

        registry = CryptoSymbolRegistry()
        registry.register(SymbolRecord(
            canonical_symbol="BTC",
            asset_class=AssetClass.CRYPTO,
            primary_provider="ccxt",
            primary_exchange_id="binance",
            provider_symbol="BTC/USDT",
            quote_currency="USDT",
        ))
        record = registry.get("BTC")
    """

    def __init__(self) -> None:
        self._records: dict[str, SymbolRecord] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, record: SymbolRecord) -> None:
        """Add *record* to the registry.

        Args:
            record: A fully-populated :class:`~spectraquant_v3.core.schema.SymbolRecord`.

        Raises:
            AssetClassLeakError: If ``record.asset_class != AssetClass.CRYPTO``.
        """
        if record.asset_class != AssetClass.CRYPTO:
            raise AssetClassLeakError(
                f"Cannot register non-crypto symbol '{record.canonical_symbol}' "
                f"(asset_class={record.asset_class.value!r}) in the crypto registry. "
                "This is an asset-class leak."
            )
        if _is_equity_symbol(record.canonical_symbol):
            raise AssetClassLeakError(
                f"Symbol '{record.canonical_symbol}' looks like an equity ticker "
                "(has a stock-exchange suffix).  Refuse to add to crypto registry."
            )
        self._records[record.canonical_symbol.upper()] = record

    def register_many(self, records: list[SymbolRecord]) -> None:
        """Register multiple records in one call."""
        for rec in records:
            self.register(rec)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, canonical_symbol: str) -> SymbolRecord:
        """Return the :class:`SymbolRecord` for *canonical_symbol*.

        Args:
            canonical_symbol: Upper-case coin ticker (e.g. ``"BTC"``).

        Raises:
            SymbolResolutionError: If the symbol is not in the registry.
        """
        key = canonical_symbol.upper()
        if key not in self._records:
            raise SymbolResolutionError(
                f"Crypto symbol '{canonical_symbol}' not found in the registry. "
                f"Available: {sorted(self._records)}"
            )
        return self._records[key]

    def contains(self, canonical_symbol: str) -> bool:
        """Return True if *canonical_symbol* is registered."""
        return canonical_symbol.upper() in self._records

    def all_symbols(self) -> list[str]:
        """Return sorted list of all registered canonical symbols."""
        return sorted(self._records.keys())

    def all_records(self) -> list[SymbolRecord]:
        """Return all registered :class:`SymbolRecord` objects."""
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"CryptoSymbolRegistry(n={len(self._records)})"


# ---------------------------------------------------------------------------
# Factory: build registry from config
# ---------------------------------------------------------------------------

def build_registry_from_config(
    cfg: dict[str, Any],
    exchange_id: str = "binance",
    quote_currency: str = "USDT",
) -> CryptoSymbolRegistry:
    """Construct a :class:`CryptoSymbolRegistry` from a V3 crypto config dict.

    Reads ``cfg["crypto"]["symbols"]`` and creates minimal :class:`SymbolRecord`
    entries.  The ``provider_symbol`` is constructed as ``<TICKER>/<QUOTE>``
    following CCXT pair notation.

    Args:
        cfg:            Merged crypto configuration dict.
        exchange_id:    Exchange to use as ``primary_exchange_id``.
        quote_currency: Quote currency for provider symbol construction.

    Returns:
        A populated :class:`CryptoSymbolRegistry`.

    Raises:
        AssetClassLeakError: If any symbol in the config looks like an equity.
    """
    registry = CryptoSymbolRegistry()
    symbols: list[str] = cfg.get("crypto", {}).get("symbols", [])
    provider: str = cfg.get("crypto", {}).get("primary_ohlcv_provider", "ccxt")

    for ticker in symbols:
        record = SymbolRecord(
            canonical_symbol=ticker.upper(),
            asset_class=AssetClass.CRYPTO,
            primary_provider=provider,
            primary_exchange_id=exchange_id,
            provider_symbol=f"{ticker.upper()}/{quote_currency}",
            exchange_symbol=f"{ticker.upper()}/{quote_currency}",
            quote_currency=quote_currency,
            market_type="spot",
            is_tradable=True,
            is_active=True,
        )
        registry.register(record)

    return registry
