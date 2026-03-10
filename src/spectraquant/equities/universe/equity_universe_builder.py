"""Equity universe builder.

Builds the tradable equity universe from a static file or config list,
with optional filter rules (minimum price, minimum volume, history check).

Rules:
- The universe must NEVER contain crypto symbols.
- Any crypto symbol found in the input raises AssetClassLeakError.
- If the universe is empty after filtering, EmptyUniverseError is raised.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from spectraquant.core.enums import AssetClass
from spectraquant.core.errors import AssetClassLeakError, EmptyUniverseError

logger = logging.getLogger(__name__)

_KNOWN_CRYPTO_SYMBOLS = frozenset([
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
    "MATIC", "LTC", "BCH", "ATOM", "UNI", "XLM", "ALGO", "VET", "FIL", "TRX",
    "SHIB", "NEAR", "ICP", "APT", "ARB", "OP", "IMX", "SAND", "AXS", "MANA",
])

_KNOWN_CRYPTO_PAIR_PATTERNS = ("/USDT", "/USD", "-USD", "USDT", "PERP")


def _is_crypto_symbol(sym: str) -> bool:
    """Return True if the symbol looks like a crypto asset."""
    upper = sym.strip().upper()
    # Plain ticker like BTC, ETH
    base = upper.split("-")[0].split("/")[0]
    if base in _KNOWN_CRYPTO_SYMBOLS:
        return True
    # Pair-style like BTC/USDT or BTCUSDT
    return any(pat in upper for pat in _KNOWN_CRYPTO_PAIR_PATTERNS)


class EquityUniverseBuilder:
    """Builds the equity universe for a single pipeline run.

    Args:
        config: The ``equities`` section of the pipeline configuration.

    Example::

        builder = EquityUniverseBuilder(config={"tickers": ["INFY.NS", "TCS.NS"]})
        universe = builder.build()
        # → ["INFY.NS", "TCS.NS"]
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self) -> list[str]:
        """Build and return the list of yfinance-style equity symbols.

        The returned symbols are deduplicated, validated for crypto leaks,
        and filtered according to configured rules.

        Returns:
            List of yfinance-style symbols (e.g. ``["INFY.NS", "TCS.NS"]``).

        Raises:
            AssetClassLeakError: If crypto symbols are found in the input.
            EmptyUniverseError: If no symbols survive filtering.
        """
        raw = self._load_raw_symbols()
        self._check_no_crypto_leak(raw)
        filtered = self._apply_filters(raw)
        filtered = list(dict.fromkeys(filtered))  # deduplicate preserving order

        if not filtered:
            raise EmptyUniverseError(
                asset_class=AssetClass.EQUITY,
                reason="All symbols were filtered out. Check min_price, min_volume, and tickers_file.",
            )

        logger.info(
            "Equity universe built: %d symbols from %d candidates",
            len(filtered),
            len(raw),
        )
        return filtered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_raw_symbols(self) -> list[str]:
        """Load raw symbols from config and/or tickers file."""
        symbols: list[str] = []

        # From config list
        cfg_tickers: list[str] = self._config.get("tickers", [])
        symbols.extend(cfg_tickers)

        # From tickers file
        tickers_file: str | None = self._config.get("tickers_file")
        if tickers_file:
            path = Path(tickers_file)
            if path.exists():
                symbols.extend(self._read_tickers_file(path))
            else:
                logger.warning("Equity tickers_file not found: %s", tickers_file)

        return symbols

    def _read_tickers_file(self, path: Path) -> list[str]:
        """Read tickers from a CSV or text file."""
        tickers: list[str] = []
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                import csv
                with path.open(newline="") as fh:
                    reader = csv.DictReader(fh)
                    # Read fieldnames eagerly to determine column layout
                    fieldnames = reader.fieldnames or []
                    if "ticker" in [f.lower() for f in fieldnames if f]:
                        col = next(
                            f for f in fieldnames if f.lower() == "ticker"
                        )
                        tickers = [
                            row[col].strip()
                            for row in reader
                            if row.get(col, "").strip()
                        ]
                    else:
                        # Treat first column as ticker
                        for row in reader:
                            vals = list(row.values())
                            if vals:
                                tickers.append(str(vals[0]).strip())
            except Exception as exc:
                logger.warning("Could not read tickers file %s: %s", path, exc)
        else:
            # Plain text, one ticker per line
            tickers = [
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        return [t for t in tickers if t]

    def _check_no_crypto_leak(self, symbols: Sequence[str]) -> None:
        """Raise AssetClassLeakError if any crypto symbols are present."""
        leaked = [s for s in symbols if _is_crypto_symbol(s)]
        if leaked:
            raise AssetClassLeakError(
                contaminating_symbols=leaked,
                pipeline_asset_class=AssetClass.EQUITY,
                contaminating_asset_class=AssetClass.CRYPTO,
            )

    def _apply_filters(self, symbols: Sequence[str]) -> list[str]:
        """Apply configured filters; return surviving symbols."""
        # Currently just passes through; price/volume filters are applied
        # post-download by the quality gate.
        exclude: list[str] = self._config.get("exclude", [])
        exclude_set = {s.upper() for s in exclude}
        return [s for s in symbols if s.upper() not in exclude_set]
