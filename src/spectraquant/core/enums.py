"""Core enumerations shared across the SpectraQuant framework.

These enums define the canonical asset classes and other shared types
used throughout the pipeline to ensure type-safe segregation.
"""
from __future__ import annotations

from enum import Enum


class AssetClass(str, Enum):
    """Canonical asset class labels.

    Every symbol in the system must have an associated AssetClass.
    Crypto and equity pipelines must never be mixed within a single run.
    """

    CRYPTO = "crypto"
    EQUITY = "equity"

    def __str__(self) -> str:  # noqa: D105
        return self.value


class RunMode(str, Enum):
    """Pipeline execution modes.

    - NORMAL: cache-first, download missing data on cache miss.
    - TEST: cache-only, fail loudly on any cache miss; no network calls.
    - REFRESH: bypass cache, force full redownload/backfill.
    """

    NORMAL = "normal"
    TEST = "test"
    REFRESH = "refresh"

    def __str__(self) -> str:  # noqa: D105
        return self.value


class SignalStatus(str, Enum):
    """Status of a signal produced by an agent."""

    OK = "OK"
    NO_SIGNAL = "NO_SIGNAL"
    ERROR = "ERROR"
    DEGRADED = "DEGRADED"

    def __str__(self) -> str:  # noqa: D105
        return self.value


class NoSignalReason(str, Enum):
    """Reason why an agent produced NO_SIGNAL."""

    NO_PRICE_DATA = "NO_PRICE_DATA"
    NO_NEWS_DATA = "NO_NEWS_DATA"
    NO_ONCHAIN_DATA = "NO_ONCHAIN_DATA"
    NO_FUNDING_DATA = "NO_FUNDING_DATA"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    FILTER_REJECTED = "FILTER_REJECTED"
    DISABLED = "DISABLED"

    def __str__(self) -> str:  # noqa: D105
        return self.value
