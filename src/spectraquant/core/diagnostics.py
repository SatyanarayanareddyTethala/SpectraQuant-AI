"""Operator diagnostics and run summary artifacts."""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List

import traceback

logger = logging.getLogger(__name__)

_ACTIVE_SUMMARY: ContextVar["RunSummary | None"] = ContextVar("active_run_summary", default=None)


@dataclass
class RunSummary:
    command: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: str | None = None
    duration_seconds: float | None = None
    status: str = "running"
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    quality_gates: List[Dict[str, Any]] = field(default_factory=list)
    universe_count: int | None = None
    tickers_used: List[str] = field(default_factory=list)

    def add_input(self, value: str) -> None:
        if value not in self.inputs:
            self.inputs.append(value)

    def add_output(self, value: str) -> None:
        if value not in self.outputs:
            self.outputs.append(value)

    def add_warning(self, value: str) -> None:
        self.warnings.append(value)

    def add_failure(self, value: str) -> None:
        self.failures.append(value)

    def add_error(self, error_type: str, message: str, stack: str | None = None) -> None:
        self.errors.append({"type": error_type, "message": message, "stack": stack})

    def add_quality_gates(self, issues: List[Dict[str, Any]]) -> None:
        self.quality_gates = issues

    def set_universe(self, tickers: List[str], universe_count: int | None) -> None:
        self.tickers_used = tickers
        if universe_count is not None:
            self.universe_count = universe_count
        else:
            self.universe_count = len(tickers)

    def finalize(self) -> None:
        ended = datetime.now(timezone.utc)
        self.ended_at = ended.isoformat()
        started = datetime.fromisoformat(self.started_at)
        self.duration_seconds = max(0.0, (ended - started).total_seconds())

    def to_payload(self) -> Dict:
        return {
            "command": self.command,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "finished_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "warnings": self.warnings,
            "failures": self.failures,
            "errors": self.errors,
            "quality_gates": self.quality_gates,
            "universe_count": self.universe_count,
            "tickers_used": self.tickers_used,
        }


class _SummaryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        summary = _ACTIVE_SUMMARY.get()
        if summary is None:
            return
        message = self.format(record)
        if record.levelno >= logging.ERROR:
            summary.add_failure(message)
            summary.add_error("log_error", message)
        elif record.levelno >= logging.WARNING:
            summary.add_warning(message)


@contextmanager
def run_summary(command: str, output_dir: Path | None = None) -> Iterator[RunSummary]:
    summary = RunSummary(command=command)
    token = _ACTIVE_SUMMARY.set(summary)
    handler = _SummaryLogHandler(level=logging.WARNING)
    formatter = logging.Formatter("%(levelname)s %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield summary
        summary.status = "success"
    except BaseException as exc:  # noqa: BLE001
        summary.status = "failed"
        summary.add_error(
            exc.__class__.__name__,
            str(exc),
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip(),
        )
        raise
    finally:
        root_logger.removeHandler(handler)
        summary.finalize()
        out_dir = output_dir or Path("reports/summary")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"run_summary_{timestamp}.json"
        path.write_text(json.dumps(summary.to_payload(), indent=2))
        _ACTIVE_SUMMARY.reset(token)
        logger.info("Run summary written to %s", path)


def record_input(value: str) -> None:
    summary = _ACTIVE_SUMMARY.get()
    if summary is None:
        return
    summary.add_input(value)


def record_output(value: str) -> None:
    summary = _ACTIVE_SUMMARY.get()
    if summary is None:
        return
    summary.add_output(value)


def record_quality_gates(issues: List[Dict[str, Any]]) -> None:
    summary = _ACTIVE_SUMMARY.get()
    if summary is None:
        return
    summary.add_quality_gates(issues)


def record_universe(tickers: List[str], universe_count: int | None = None) -> None:
    summary = _ACTIVE_SUMMARY.get()
    if summary is None:
        return
    summary.set_universe(tickers, universe_count)
