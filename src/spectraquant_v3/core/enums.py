"""Core enumerations for SpectraQuant-AI-V3.

All pipeline-wide constant values are defined here so they can be imported
by core, crypto, and equities modules without circular dependencies.
"""

from enum import Enum


class AssetClass(str, Enum):
    """Asset-class identifier.  Used to enforce strict pipeline segregation."""

    CRYPTO = "crypto"
    EQUITY = "equity"


class RunMode(str, Enum):
    """Controls cache vs. network behaviour for every run.

    NORMAL  – cache-first; download only what is missing; patch cache.
    TEST    – cache-only; any network call raises CacheOnlyViolationError.
    REFRESH – force full re-download; overwrite stale cache.
    """

    NORMAL = "normal"
    TEST = "test"
    REFRESH = "refresh"


class RunStage(str, Enum):
    """Ordered pipeline stages.  Every stage must complete before the next begins."""

    UNIVERSE = "universe"
    INGESTION = "ingestion"
    FEATURES = "features"
    SIGNALS = "signals"
    META_POLICY = "meta_policy"
    ALLOCATION = "allocation"
    EXECUTION = "execution"
    REPORTING = "reporting"


class SignalStatus(str, Enum):
    """Outcome of a single signal-agent evaluation."""

    OK = "OK"
    NO_SIGNAL = "NO_SIGNAL"
    ERROR = "ERROR"


class RunStatus(str, Enum):
    """Final disposition of a complete pipeline run."""

    SUCCESS = "success"
    ABORTED = "aborted"
    PARTIAL = "partial"


class ExecutionMode(str, Enum):
    """Execution environment – paper is the default; live adapters are placeholders."""

    PAPER = "paper"
    LIVE = "live"
