"""Core error types for the SpectraQuant framework.

These errors enforce the non-negotiable architecture rules:
- Crypto and equity symbols must never be mixed in the same pipeline.
- Symbol resolution must always go through the symbol registry.
- Missing data must fail loudly, never silently produce empty signals.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectraquant.core.enums import AssetClass


class SpectraQuantError(Exception):
    """Base class for all SpectraQuant pipeline errors."""


# ---------------------------------------------------------------------------
# Symbol resolution errors
# ---------------------------------------------------------------------------


class SymbolResolutionError(SpectraQuantError):
    """Raised when a symbol cannot be resolved via the symbol registry.

    This is raised when:
    - The symbol does not exist in the registry.
    - The asset class of the symbol does not match the pipeline context.
    - A crypto symbol is used in an equity pipeline (or vice versa).
    """

    def __init__(
        self,
        symbol: str,
        expected_asset_class: "AssetClass | None" = None,
        actual_asset_class: "AssetClass | None" = None,
        reason: str = "",
    ) -> None:
        self.symbol = symbol
        self.expected_asset_class = expected_asset_class
        self.actual_asset_class = actual_asset_class
        self.reason = reason
        msg = f"Cannot resolve symbol {symbol!r}"
        if expected_asset_class:
            msg += f" in {expected_asset_class} pipeline"
        if actual_asset_class and actual_asset_class != expected_asset_class:
            msg += f" (symbol belongs to {actual_asset_class})"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class AssetClassLeakError(SpectraQuantError):
    """Raised when symbols from the wrong asset class leak into a pipeline.

    This is the hard guard against equity symbols appearing in the crypto
    pipeline (or crypto symbols appearing in the equity pipeline).
    """

    def __init__(
        self,
        contaminating_symbols: list[str],
        pipeline_asset_class: "AssetClass",
        contaminating_asset_class: "AssetClass",
    ) -> None:
        self.contaminating_symbols = contaminating_symbols
        self.pipeline_asset_class = pipeline_asset_class
        self.contaminating_asset_class = contaminating_asset_class
        super().__init__(
            f"ASSET CLASS LEAK DETECTED: {contaminating_asset_class} symbols "
            f"{contaminating_symbols!r} found in {pipeline_asset_class} pipeline. "
            "This is a hard architecture violation. "
            "Ensure crypto and equity universes are built separately."
        )


# ---------------------------------------------------------------------------
# Data availability errors
# ---------------------------------------------------------------------------


class EmptyOHLCVError(SpectraQuantError):
    """Raised when OHLCV load returns zero rows for all selected symbols.

    This prevents silent empty-signal runs. The pipeline must fail loudly
    and write a QA report + run manifest before exiting.
    """

    def __init__(
        self,
        asset_class: "AssetClass",
        symbols: list[str],
        run_mode: str = "",
    ) -> None:
        self.asset_class = asset_class
        self.symbols = symbols
        self.run_mode = run_mode
        ac_str = str(asset_class).upper()
        super().__init__(
            f"EMPTY_{ac_str}_OHLCV_UNIVERSE: Loaded prices for 0 symbols out of "
            f"{len(symbols)} requested {list(symbols)!r}. "
            f"run_mode={run_mode!r}. "
            "Check data cache, symbol map, and provider connectivity."
        )


class CacheOnlyViolationError(SpectraQuantError):
    """Raised when test-mode (cache-only) would require a network call.

    In test mode the pipeline must never make outbound network requests.
    Any cache miss must raise this error immediately.
    """

    def __init__(self, symbol: str, data_type: str = "OHLCV") -> None:
        self.symbol = symbol
        self.data_type = data_type
        super().__init__(
            f"TEST MODE VIOLATION: {data_type} cache miss for {symbol!r}. "
            "In test/CI mode no network calls are permitted. "
            "Populate the cache before running in test mode."
        )


# ---------------------------------------------------------------------------
# Universe errors
# ---------------------------------------------------------------------------


class EmptyUniverseError(SpectraQuantError):
    """Raised when the universe builder produces zero tradable symbols."""

    def __init__(self, asset_class: "AssetClass", reason: str = "") -> None:
        self.asset_class = asset_class
        self.reason = reason
        msg = f"Universe for {asset_class} produced 0 tradable symbols"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigValidationError(SpectraQuantError):
    """Raised when the pipeline configuration is invalid or inconsistent."""


class MixedAssetRunError(SpectraQuantError):
    """Raised when a run command attempts to mix crypto and equity in one invocation.

    The crypto and equity pipelines must never run in the same pipeline
    invocation. There must be no mixed run command, no shared runtime
    universe, and no combined ingest command.
    """

    def __init__(self) -> None:
        super().__init__(
            "MIXED ASSET RUN FORBIDDEN: Crypto and equity pipelines cannot run "
            "in the same invocation. Use 'equity-run' or 'crypto-run' separately."
        )
