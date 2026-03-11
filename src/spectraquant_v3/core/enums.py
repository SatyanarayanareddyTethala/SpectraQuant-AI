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
    """Outcome of a single signal-agent evaluation.

    OK          – Signal computed successfully; score and confidence are valid.
    NO_SIGNAL   – Agent ran but found no actionable signal (data present but
                  below threshold, or symbol not selected by a cross-sectional
                  filter).
    DEGRADED    – Agent produced a signal but with partial or lower-quality
                  data (e.g. fewer rows than ideal, missing one of several
                  input features).  Downstream consumers should treat this
                  with reduced confidence.
    ERROR       – Agent encountered an unrecoverable error; score and
                  confidence are always 0.0; ``error_reason`` is populated.
    """

    OK = "OK"
    NO_SIGNAL = "NO_SIGNAL"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class NoSignalReason(str, Enum):
    """Machine-readable reason why a signal agent returned NO_SIGNAL or DEGRADED.

    Stored in :attr:`~spectraquant_v3.core.schema.SignalRow.no_signal_reason`.
    Using controlled vocabulary here makes it possible to bucket diagnostics
    reliably in the backtest engine and reporter.

    These reasons are valid for both ``NO_SIGNAL`` (no actionable signal could
    be produced) and ``DEGRADED`` (a signal was produced but with reduced
    reliability due to data limitations).
    """

    MISSING_INPUTS = "missing_inputs"
    """Required input columns or data sources were absent."""

    INSUFFICIENT_ROWS = "insufficient_rows"
    """Not enough historical rows to compute the signal.
    Also the typical reason for a DEGRADED status when partial data is available."""

    TOP_N_CUTOFF = "top_n_cutoff"
    """Symbol was ranked but fell outside the top-N selection window."""

    BELOW_THRESHOLD = "below_threshold"
    """Computed score or confidence was below the configured minimum."""

    NO_NEWS_DATA = "no_news_data"
    """News data (catalyst column) was absent or contained only NaN values.
    The agent degrades safely to NO_SIGNAL rather than raising."""

    UNKNOWN = "unknown"
    """Catch-all for cases not covered by the above categories."""


class RunStatus(str, Enum):
    """Final disposition of a complete pipeline run."""

    SUCCESS = "success"
    ABORTED = "aborted"
    PARTIAL = "partial"


class ExecutionMode(str, Enum):
    """Execution environment – paper is the default; live adapters are placeholders."""

    PAPER = "paper"
    LIVE = "live"
