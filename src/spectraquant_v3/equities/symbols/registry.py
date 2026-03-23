"""Equity symbol registry for SpectraQuant-AI-V3.

The registry is the *single source of truth* for every equity symbol processed
by the V3 pipeline.  All downstream modules receive a :class:`SymbolRecord`
from the registry and must never hard-code provider-specific symbol strings.

Design rules:
- Every symbol stored here must have ``asset_class == AssetClass.EQUITY``.
- Attempting to register a crypto symbol raises :exc:`AssetClassLeakError`.
- ``canonical_symbol`` is the yfinance-compatible ticker (e.g. ``"INFY.NS"``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectraquant_v3.core.enums import AssetClass
from spectraquant_v3.core.errors import (
    AssetClassLeakError,
    ConfigValidationError,
    EmptyUniverseError,
    SymbolResolutionError,
)
from spectraquant_v3.core.schema import SymbolRecord
from spectraquant.universe.loader import load_nse_universe

# ---------------------------------------------------------------------------
# Crypto-pattern heuristic used for leak detection
# ---------------------------------------------------------------------------
_CRYPTO_PATTERNS = ("/USDT", "/BTC", "/ETH", "/USDC", "/BNB", "/BUSD")


def _is_crypto_symbol(symbol: str) -> bool:
    """Heuristic: return True when *symbol* looks like a crypto pair."""
    upper = symbol.upper()
    return any(upper.endswith(p) for p in _CRYPTO_PATTERNS)


class EquitySymbolRegistry:
    """In-memory registry of canonical equity symbols.

    Usage::

        registry = EquitySymbolRegistry()
        registry.register(SymbolRecord(
            canonical_symbol="INFY.NS",
            asset_class=AssetClass.EQUITY,
            primary_provider="yfinance",
            yfinance_symbol="INFY.NS",
        ))
        record = registry.get("INFY.NS")
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
            AssetClassLeakError: If ``record.asset_class != AssetClass.EQUITY``.
        """
        if record.asset_class != AssetClass.EQUITY:
            raise AssetClassLeakError(
                f"Cannot register non-equity symbol '{record.canonical_symbol}' "
                f"(asset_class={record.asset_class.value!r}) in the equity registry. "
                "This is an asset-class leak."
            )
        if _is_crypto_symbol(record.canonical_symbol):
            raise AssetClassLeakError(
                f"Symbol '{record.canonical_symbol}' looks like a crypto pair. "
                "Refuse to add to equity registry."
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
            canonical_symbol: Exchange-qualified ticker (e.g. ``"INFY.NS"``).

        Raises:
            SymbolResolutionError: If the symbol is not in the registry.
        """
        key = canonical_symbol.upper()
        if key not in self._records:
            raise SymbolResolutionError(
                f"Equity symbol '{canonical_symbol}' not found in the registry. "
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
        return f"EquitySymbolRegistry(n={len(self._records)})"


# ---------------------------------------------------------------------------
# Factory: build registry from config
# ---------------------------------------------------------------------------

def build_registry_from_config(cfg: dict[str, Any]) -> EquitySymbolRegistry:
    """Construct an :class:`EquitySymbolRegistry` from a V3 equities config dict.

    Reads ``cfg["equities"]["universe"]["tickers"]`` and creates minimal
    :class:`SymbolRecord` entries using yfinance as primary provider.

    Args:
        cfg: Merged equities configuration dict.

    Returns:
        A populated :class:`EquitySymbolRegistry`.

    Raises:
        AssetClassLeakError: If any ticker in the config looks like a crypto pair.
    """
    registry = EquitySymbolRegistry()
    universe_cfg: dict[str, Any] = cfg.get("equities", {}).get("universe", {})
    tickers = resolve_equity_tickers_from_config(cfg)
    provider: str = cfg.get("equities", {}).get("primary_ohlcv_provider", "yfinance")

    for ticker in tickers:
        record = SymbolRecord(
            canonical_symbol=ticker,
            asset_class=AssetClass.EQUITY,
            primary_provider=provider,
            yfinance_symbol=ticker,
            provider_symbol=ticker,
            market_type="equity",
            is_tradable=True,
            is_active=True,
        )
        registry.register(record)

    return registry


def resolve_equity_tickers_from_config(cfg: dict[str, Any]) -> list[str]:
    """Resolve canonical equity tickers from config with NSE-first semantics.

    Resolution order:
    1. ``equities.universe.tickers`` if present.
    2. ``equities.universe.tickers_file`` using the stable V2 NSE loader.

    Bare NSE symbols are normalized to ``.NS`` suffixed canonical tickers.
    """
    universe_cfg: dict[str, Any] = cfg.get("equities", {}).get("universe", {})

    configured = universe_cfg.get("tickers", [])
    if configured:
        return _canonicalize_equity_tickers(configured)

    tickers_file = str(universe_cfg.get("tickers_file", "")).strip()
    if tickers_file:
        path = Path(tickers_file)
        suffix = str(universe_cfg.get("default_suffix", ".NS"))
        tickers, meta, diagnostics = load_nse_universe(path, suffix=suffix)
        if diagnostics:
            details = "; ".join(f"{diag.code}: {diag.message}" for diag in diagnostics)
            raise ConfigValidationError(
                f"Failed to resolve equity universe from tickers_file={path}: {details}"
            )
        if not tickers:
            raise EmptyUniverseError(
                f"Equity universe file '{path}' produced zero canonical tickers. meta={meta}"
            )
        return tickers

    return []


def _canonicalize_equity_tickers(tickers: list[str]) -> list[str]:
    """Normalize configured equity tickers and enforce deterministic ordering."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        symbol = str(raw).strip().upper()
        if not symbol:
            continue
        if _is_crypto_symbol(symbol):
            raise AssetClassLeakError(
                f"Symbol '{symbol}' looks like a crypto pair. Refuse to treat it as equity."
            )
        if "." not in symbol:
            symbol = f"{symbol}.NS"
        if symbol not in seen:
            cleaned.append(symbol)
            seen.add(symbol)
    return cleaned
