"""Missing-bar diagnostics for SpectraQuant-AI-V3."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MissingBarReport:
    canonical_symbol: str
    expected_freq: str
    total_expected: int
    total_present: int
    missing_count: int
    missing_fraction: float
    first_missing: datetime.datetime | None
    last_missing: datetime.datetime | None
    gap_timestamps: list[datetime.datetime] = field(default_factory=list)
    gap_runs: list[tuple[datetime.datetime, datetime.datetime]] = field(default_factory=list)

    @property
    def has_gaps(self) -> bool:
        return self.missing_count > 0

    @property
    def coverage(self) -> float:
        return 0.0 if self.total_expected == 0 else self.total_present / self.total_expected

    @property
    def missing_dates(self) -> list[str]:
        return [str(ts.date()) if hasattr(ts, "date") else str(ts) for ts in self.gap_timestamps]

    @property
    def symbol(self) -> str:
        return self.canonical_symbol

    def summary(self) -> str:
        if not self.has_gaps:
            return f"{self.canonical_symbol}: no missing bars"
        return f"{self.canonical_symbol}: {self.missing_count} missing bars"

    def to_dict(self) -> dict:
        return {
            "symbol": self.canonical_symbol,
            "canonical_symbol": self.canonical_symbol,
            "expected_freq": self.expected_freq,
            "total_expected": self.total_expected,
            "total_present": self.total_present,
            "missing_count": self.missing_count,
            "missing_fraction": round(self.missing_fraction, 6),
            "coverage": round(self.coverage, 6),
            "first_missing": str(self.first_missing) if self.first_missing else None,
            "last_missing": str(self.last_missing) if self.last_missing else None,
            "missing_dates": self.missing_dates,
        }


_FREQ_ALIASES = {"1d": "D", "1h": "h", "4h": "4h", "1w": "W", "D": "D", "H": "h", "W": "W"}


def _to_pandas_freq(expected_freq: str) -> str:
    return _FREQ_ALIASES.get(expected_freq, expected_freq)


def diagnose_missing_bars(
    df: pd.DataFrame,
    symbol: str,
    expected_freq: str = "1d",
    end: str | datetime.datetime | None = None,
    max_gaps: int = 1000,
) -> MissingBarReport:
    """Supports both signatures:
    - diagnose_missing_bars(df, symbol, expected_freq='1d')
    - diagnose_missing_bars(df, symbol, start, end)
    """
    if df.empty:
        return MissingBarReport(
            canonical_symbol=symbol,
            expected_freq="1d" if end is not None else expected_freq,
            total_expected=0,
            total_present=0,
            missing_count=0,
            missing_fraction=0.0,
            first_missing=None,
            last_missing=None,
            gap_timestamps=[],
            gap_runs=[],
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    start_ts: pd.Timestamp
    end_ts: pd.Timestamp

    if end is not None:
        start_ts = pd.Timestamp(expected_freq)
        end_ts = pd.Timestamp(end)
        pandas_freq = "D"
        report_freq = "1d"
    else:
        pandas_freq = _to_pandas_freq(expected_freq)
        report_freq = expected_freq
        if df.empty:
            start_ts = end_ts = pd.Timestamp.now(tz="UTC")
        else:
            start_ts = df.index.min()
            end_ts = df.index.max()

    expected_index = pd.date_range(start=start_ts, end=end_ts, freq=pandas_freq, tz=df.index.tz)
    present_index = df.index
    if pandas_freq == "D":
        expected_set = set(expected_index.normalize())
        present_set = set(present_index.normalize())
    else:
        expected_set = set(expected_index)
        present_set = set(present_index)

    missing = sorted(expected_set - present_set)
    total_expected = len(expected_set)
    total_present = len(expected_set & present_set)
    missing_count = len(missing)

    gap_runs: list[tuple[datetime.datetime, datetime.datetime]] = []
    if missing:
        run_start = missing[0]
        run_end = missing[0]
        step = pd.Timedelta(days=1) if pandas_freq == "D" else pd.Timedelta(expected_index.freq)
        for ts in missing[1:]:
            if pd.Timestamp(ts) - pd.Timestamp(run_end) <= step * 1.5:
                run_end = ts
            else:
                gap_runs.append((run_start, run_end))
                run_start = ts
                run_end = ts
        gap_runs.append((run_start, run_end))

    return MissingBarReport(
        canonical_symbol=symbol,
        expected_freq=report_freq,
        total_expected=total_expected,
        total_present=total_present,
        missing_count=missing_count,
        missing_fraction=(missing_count / total_expected) if total_expected else 0.0,
        first_missing=missing[0] if missing else None,
        last_missing=missing[-1] if missing else None,
        gap_timestamps=missing[:max_gaps],
        gap_runs=gap_runs,
    )
