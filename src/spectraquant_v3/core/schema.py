"""Typed data schemas for SpectraQuant-AI-V3.

All inter-stage data structures are defined here as dataclasses so that
every pipeline stage produces strongly-typed, validated output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from spectraquant_v3.core.enums import AssetClass, NoSignalReason, SignalStatus  # noqa: F401
from spectraquant_v3.core.time import normalize_datetime_frame

_schema_logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float, hi: float) -> float:
    """Return *value* clamped to the closed interval [*lo*, *hi*]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Symbol registry record
# ---------------------------------------------------------------------------


@dataclass
class SymbolRecord:
    """Canonical symbol definition stored in the symbol registry.

    All downstream modules consume ``canonical_symbol`` only.
    Provider-specific resolution (e.g. yfinance_symbol, exchange_symbol)
    must happen exclusively through the symbol registry / symbol mapper.
    """

    canonical_symbol: str
    asset_class: AssetClass
    primary_provider: str = ""
    primary_exchange_id: str = ""
    provider_symbol: str = ""
    exchange_symbol: str = ""
    yfinance_symbol: str = ""
    coingecko_id: str = ""
    contract_address: str = ""
    quote_currency: str = ""
    market_type: str = "spot"
    is_tradable: bool = True
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# QA / data availability matrix row
# ---------------------------------------------------------------------------


@dataclass
class QARow:
    """One row in the per-run QA / data availability matrix.

    Every symbol processed in a run produces exactly one QARow so that
    data completeness can be audited at a glance.
    """

    run_id: str
    as_of: str
    canonical_symbol: str
    asset_class: str
    has_ohlcv: bool = False
    has_news: bool = False
    has_onchain: bool = False
    has_funding: bool = False
    has_oi: bool = False
    has_orderbook: bool = False
    provider_used: str = ""
    rows_loaded: int = 0
    min_timestamp: str = ""
    max_timestamp: str = ""
    missing_days: int = 0
    stage_status: str = "PENDING"
    error_codes: list[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Signal output row
# ---------------------------------------------------------------------------


@dataclass
class SignalRow:
    """Standardised signal output from any signal agent.

    Agents must emit a :class:`SignalRow` for every symbol they are asked
    to evaluate, even when data is unavailable (status=NO_SIGNAL).

    Required fields (must be non-empty strings):
        run_id, timestamp, canonical_symbol, asset_class, agent_id, horizon

    Optional fields (safe defaults provided):
        signal_score   – clamped to ``[-1.0, +1.0]`` on construction.
        confidence     – clamped to ``[0.0,  1.0]`` on construction.
        no_signal_reason – machine-readable reason key; use
                           :class:`~spectraquant_v3.core.enums.NoSignalReason`
                           values where possible.
        rationale      – free-text human-readable explanation.
        error_reason   – populated when status=ERROR.
        required_inputs / available_inputs – for QA audit only.
    """

    run_id: str
    timestamp: str
    canonical_symbol: str
    asset_class: str
    agent_id: str
    horizon: str
    signal_score: float = 0.0       # range -1.0 to +1.0; clamped at construction
    confidence: float = 0.0         # range  0.0 to  1.0; clamped at construction
    required_inputs: list[str] = field(default_factory=list)
    available_inputs: list[str] = field(default_factory=list)
    rationale: str = ""
    no_signal_reason: str = ""      # use NoSignalReason values where possible
    status: str = SignalStatus.NO_SIGNAL.value
    error_reason: str = ""

    def __post_init__(self) -> None:
        """Clamp signal_score and confidence to their documented ranges.

        Out-of-range values are clamped (not rejected) so that execution
        paths remain robust to minor floating-point overflows.  A warning
        is logged so that misbehaving agents can be identified and fixed.
        """
        if self.signal_score < -1.0 or self.signal_score > 1.0:
            _schema_logger.warning(
                "SignalRow: signal_score=%.6f for symbol=%r agent=%r is outside [-1.0, +1.0]; "
                "clamping to range.",
                self.signal_score,
                self.canonical_symbol,
                self.agent_id,
            )
            self.signal_score = _clamp(self.signal_score, -1.0, 1.0)

        if self.confidence < 0.0 or self.confidence > 1.0:
            _schema_logger.warning(
                "SignalRow: confidence=%.6f for symbol=%r agent=%r is outside [0.0, 1.0]; "
                "clamping to range.",
                self.confidence,
                self.canonical_symbol,
                self.agent_id,
            )
            self.confidence = _clamp(self.confidence, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Allocation output row
# ---------------------------------------------------------------------------


@dataclass
class AllocationRow:
    """Target portfolio weight for a single symbol after allocation.

    Required fields:
        run_id, canonical_symbol, asset_class

    Optional fields (safe defaults provided):
        target_weight     – decimal weight; positive = long, negative = short.
        blocked           – True when this symbol was excluded by a policy rule.
        blocked_reason    – human-readable reason for the block.
        expected_turnover – estimated one-way turnover for this position change.
        timestamp         – ISO-8601 UTC string recording when the allocation
                            was computed; leave empty when unavailable.
    """

    run_id: str
    canonical_symbol: str
    asset_class: str
    target_weight: float = 0.0
    blocked: bool = False
    blocked_reason: str = ""
    expected_turnover: float = 0.0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# DataFrame schema validators
# ---------------------------------------------------------------------------

# Minimum required columns for a valid OHLCV DataFrame
_OHLCV_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"open", "high", "low", "close", "volume"}
)


def validate_ohlcv_dataframe(
    df: Any,
    symbol: str = "",
    min_rows: int = 1,
) -> None:
    """Assert that *df* is a non-empty DataFrame with required OHLCV columns.

    This validation must be called by every ingestor before writing to cache
    or passing data to downstream stages.  Column names are case-insensitive.

    Args:
        df:        Object to validate (expected to be a ``pandas.DataFrame``).
        symbol:    Symbol name for error messages.
        min_rows:  Minimum number of rows required (default 1).

    Raises:
        DataSchemaError: If *df* is not a DataFrame, is empty, is missing
            required columns, or has fewer rows than *min_rows*.
        EmptyPriceDataError: If *df* has zero rows.
    """
    import pandas as pd

    from spectraquant_v3.core.errors import DataSchemaError, EmptyPriceDataError

    label = f" for '{symbol}'" if symbol else ""

    if not isinstance(df, pd.DataFrame):
        raise DataSchemaError(
            f"Expected a pandas DataFrame{label}, got {type(df).__name__}."
        )

    if df.empty:
        raise EmptyPriceDataError(
            f"OHLCV DataFrame{label} is empty. "
            "An empty DataFrame is never a valid ingestor result."
        )

    if len(df) < min_rows:
        raise EmptyPriceDataError(
            f"OHLCV DataFrame{label} has {len(df)} rows but {min_rows} are required."
        )

    actual_cols = {c.lower() for c in df.columns}
    missing = _OHLCV_REQUIRED_COLUMNS - actual_cols
    if missing:
        raise DataSchemaError(
            f"OHLCV DataFrame{label} is missing required columns: {sorted(missing)}. "
            f"Present columns: {sorted(actual_cols)}."
        )

    if "timestamp" in actual_cols or isinstance(df.index, pd.DatetimeIndex):
        normalize_datetime_frame(df, label=f"OHLCV DataFrame{label}")
