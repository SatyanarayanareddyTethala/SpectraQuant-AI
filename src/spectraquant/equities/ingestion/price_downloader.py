"""Equity price downloader.

Orchestrates OHLCV ingestion for the equity pipeline.

Features:
- Resolves canonical symbols → yfinance symbols via EquitySymbolMapper.
- Enforces no-crypto-leak guard before any download.
- Supports test/normal/refresh run modes.
- Returns QA availability data per symbol.
- Raises EmptyOHLCVError if zero symbols have data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from spectraquant.core.enums import AssetClass, RunMode
from spectraquant.core.errors import EmptyOHLCVError
from spectraquant.equities.ingestion.providers.yfinance_provider import (
    YFinanceEquityProvider,
)
from spectraquant.equities.symbols.equity_symbol_mapper import EquitySymbolMapper

logger = logging.getLogger(__name__)


@dataclass
class EquityOHLCVResult:
    """Output of a single equity price download run."""

    prices: dict[str, pd.DataFrame] = field(default_factory=dict)
    """Mapping of yfinance symbol → OHLCV DataFrame."""

    qa: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-symbol QA metadata."""

    symbols_requested: list[str] = field(default_factory=list)
    symbols_loaded: list[str] = field(default_factory=list)
    symbols_failed: list[str] = field(default_factory=list)

    @property
    def load_rate(self) -> float:
        if not self.symbols_requested:
            return 0.0
        return len(self.symbols_loaded) / len(self.symbols_requested)

    def assert_ohlcv_available(self, run_mode: str = "") -> None:
        """Abort the pipeline if no symbol has usable OHLCV data.

        This is the hard guard documented by the QA module: a run with
        zero OHLCV rows must never proceed to signal generation.  Call
        this method immediately after QA population to enforce the
        contract.

        Args:
            run_mode: Optional run-mode label included in the error message
                for diagnostics (e.g. ``str(RunMode.TEST)``).

        Raises:
            EmptyOHLCVError: If the QA matrix shows no symbol with
                ``has_ohlcv=True``, or if no symbols were loaded at all.
        """
        if not any(entry.get("has_ohlcv", False) for entry in self.qa.values()):
            raise EmptyOHLCVError(
                asset_class=AssetClass.EQUITY,
                symbols=self.symbols_requested,
                run_mode=run_mode,
            )


class EquityPriceDownloader:
    """Download equity OHLCV and return structured results.

    Args:
        config: The equity/data section of the pipeline configuration.
        run_mode: Cache behaviour (NORMAL/TEST/REFRESH).
        cache_dir: Base directory for OHLCV parquet cache.
        mapper: Optional pre-built EquitySymbolMapper.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        run_mode: RunMode = RunMode.NORMAL,
        cache_dir: str | Path = "data/equities/prices",
        mapper: EquitySymbolMapper | None = None,
    ) -> None:
        self._config = config or {}
        self._run_mode = run_mode
        self._cache_dir = Path(cache_dir)
        self._mapper = mapper or EquitySymbolMapper()

    def download(self, yfinance_symbols: list[str]) -> EquityOHLCVResult:
        """Download OHLCV for *yfinance_symbols*.

        Args:
            yfinance_symbols: yfinance-style symbols (e.g. ``["INFY.NS", "TCS.NS"]``).

        Returns:
            EquityOHLCVResult with prices and QA data.

        Raises:
            EmptyOHLCVError: If zero symbols return data.
        """
        # Bootstrap mapper from provided symbols
        self._mapper.bootstrap_from_yfinance_symbols(yfinance_symbols)
        # Check for crypto leaks using yfinance symbols directly
        # (crypto symbols won't have exchange suffixes)
        provider = YFinanceEquityProvider(
            cache_dir=self._cache_dir,
            run_mode=self._run_mode,
        )
        raw = provider.fetch(yfinance_symbols)

        result = EquityOHLCVResult(
            prices=raw,
            symbols_requested=list(yfinance_symbols),
            symbols_loaded=list(raw.keys()),
            symbols_failed=[s for s in yfinance_symbols if s not in raw],
        )

        # Populate QA
        for sym in yfinance_symbols:
            df = raw.get(sym)
            if df is not None and not df.empty:
                result.qa[sym] = {
                    "has_ohlcv": True,
                    "rows_loaded": len(df),
                    "min_timestamp": str(df.index.min()),
                    "max_timestamp": str(df.index.max()),
                    "provider_used": "yfinance",
                    "asset_class": str(AssetClass.EQUITY),
                }
            else:
                result.qa[sym] = {
                    "has_ohlcv": False,
                    "rows_loaded": 0,
                    "provider_used": "yfinance",
                    "asset_class": str(AssetClass.EQUITY),
                    "error_codes": ["NO_DATA"],
                }

        # Hard guard: abort if no OHLCV data is available after QA population.
        result.assert_ohlcv_available(run_mode=str(self._run_mode))

        logger.info(
            "Equity OHLCV loaded: %d/%d symbols",
            len(result.symbols_loaded),
            len(result.symbols_requested),
        )
        return result
