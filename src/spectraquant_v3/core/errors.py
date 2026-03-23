"""Custom exceptions for SpectraQuant-AI-V3.

Every failure mode that must be detected explicitly has its own exception type.
Silent failures (empty frames returned as success, swallowed errors) are forbidden.
"""


class SpectraQuantError(Exception):
    """Base exception for all SpectraQuant-V3 errors."""


# ---------------------------------------------------------------------------
# Pipeline segregation
# ---------------------------------------------------------------------------


class MixedAssetClassRunError(SpectraQuantError):
    """Raised when crypto and equity symbols are detected in the same run.

    Example triggers:
    - An equity symbol (e.g. INFY.NS) appears in a crypto command.
    - A crypto pair (e.g. BTC/USDT) appears in an equity command.
    """


class AssetClassLeakError(MixedAssetClassRunError):
    """Raised when an equity symbol leaks into a crypto pipeline, or vice versa."""


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------


class SymbolResolutionError(SpectraQuantError):
    """Raised when a symbol cannot be resolved via the symbol registry/mapper.

    This must be raised loudly – never fall back to a default silently.
    """


# ---------------------------------------------------------------------------
# Data quality / availability
# ---------------------------------------------------------------------------


class EmptyUniverseError(SpectraQuantError):
    """Raised when the universe builder resolves to zero eligible symbols."""


class EmptyPriceDataError(SpectraQuantError):
    """Raised when a price series is empty or has insufficient rows.

    An empty DataFrame is never a valid success result.
    """


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class CacheOnlyViolationError(SpectraQuantError):
    """Raised in TEST run-mode when a network call would be required.

    This enforces reproducible, deterministic CI and research runs.
    """


class CacheCorruptionError(SpectraQuantError):
    """Raised when a cached parquet file fails schema validation."""


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


class InvalidRunModeError(SpectraQuantError):
    """Raised when an unrecognised run-mode string is supplied."""


class ManifestWriteError(SpectraQuantError):
    """Raised when the run manifest file cannot be written to disk."""


class ManifestValidationError(SpectraQuantError):
    """Raised when a manifest payload is missing required fields or is invalid."""


class StageAbortError(SpectraQuantError):
    """Raised when a pipeline stage must halt due to a fatal data condition."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigValidationError(SpectraQuantError):
    """Raised when a config dict is missing required keys or has wrong types.

    Example triggers:
    - A required top-level key (e.g. ``run``, ``cache``) is absent.
    - A value has an unexpected type (e.g. ``run.mode`` is not a string).
    """


# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------


class DataSchemaError(SpectraQuantError):
    """Raised when a DataFrame is missing required columns or has wrong dtypes.

    Every ingestor must return a DataFrame that passes schema validation
    before the data is written to cache or passed downstream.
    """


class ContractViolationError(SpectraQuantError):
    """Raised when a contract object is constructed with out-of-spec values.

    Example triggers:
    - ``SignalRow.signal_score`` outside the documented ``[-1.0, +1.0]`` range.
    - ``SignalRow.confidence`` outside the documented ``[0.0, 1.0]`` range.

    Note: the schema layer currently *clamps* values and logs a warning rather
    than raising, to keep execution paths robust.  This error is reserved for
    callers that need to enforce strict contract compliance.
    """


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


class UniverseValidationError(SpectraQuantError):
    """Raised when the hybrid universe CSV fails schema or content validation.

    Example triggers:
    - A required column (e.g. ``symbol``, ``asset_class``) is missing.
    - A row has an empty ``symbol`` field.
    - An unrecognised ``asset_class`` value is encountered.
    - The universe contains more than 150 assets.
    - Duplicate ``symbol`` entries are detected (after deduplication is disallowed).
    """
