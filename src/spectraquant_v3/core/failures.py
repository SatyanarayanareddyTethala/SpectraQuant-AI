"""Deterministic failure taxonomy for orchestration and diagnostics.

These codes are stable contracts for machine operators (n8n, alerting, incident
triage). Do not repurpose existing codes; add new codes for new classes of
failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureCode(str, Enum):
    """Stable machine-readable failure codes."""

    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_MODE = "INVALID_MODE"
    RUN_MODE_GUARD = "RUN_MODE_GUARD"
    STALE_IDEMPOTENCY_KEY = "STALE_IDEMPOTENCY_KEY"
    DUPLICATE_SUBMISSION = "DUPLICATE_SUBMISSION"
    RUN_LOCKED = "RUN_LOCKED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"

    CONFIG_ERROR = "CONFIG_ERROR"
    EMPTY_UNIVERSE = "EMPTY_UNIVERSE"
    EMPTY_DATASET = "EMPTY_DATASET"
    DATE_ALIGNMENT = "DATE_ALIGNMENT"
    TIMESTAMP_RANGE = "TIMESTAMP_RANGE"
    ALLOCATION_ZERO = "ALLOCATION_ZERO"
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    MANIFEST_INVALID = "MANIFEST_INVALID"

    PIPELINE_FAILURE = "PIPELINE_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class FailureDetail:
    """Normalized failure payload emitted by service and diagnostics."""

    code: FailureCode
    message: str
    stage: str = ""
    retriable: bool = False
