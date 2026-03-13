"""Governance prediction logger for SpectraQuant-AI.

Writes a per-ticker decision record to disk (JSON lines) so every prediction
is fully auditable.  This implements the governance / explainability layer
described in the system specification.

Each record contains:
    ticker              – ticker symbol
    asof_utc            – ISO-8601 UTC timestamp of the prediction
    action              – "BUY", "SELL", or "HOLD"
    reason              – human-readable explanation
    event_type          – dominant event type (from event classifier)
    analysis_model      – model chosen by model_selector
    expected_move_pct   – expected price move as percentage
    target_price        – base target price
    stop_price          – suggested stop-loss price
    confidence          – calibrated confidence [0, 1]
    risk_score          – composite risk score [0, 1]
    news_refs           – list of news article URLs
    historical_analogs  – number of analogs used
    regime              – market regime label

Usage
-----
>>> from spectraquant.governance.prediction_log import GovernanceLogger
>>> logger = GovernanceLogger("/tmp/governance")
>>> logger.write({...})
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

__all__ = ["GovernanceLogger", "PredictionRecord"]

_log = logging.getLogger(__name__)

# Mandatory keys that a complete prediction record must contain.
MANDATORY_KEYS: tuple[str, ...] = (
    "ticker",
    "asof_utc",
    "action",
    "reason",
    "event_type",
    "analysis_model",
    "expected_move_pct",
    "target_price",
    "stop_price",
    "confidence",
    "risk_score",
    "news_refs",
)


class PredictionRecord(dict):
    """Thin wrapper around a prediction dict with validation helpers."""

    def missing_keys(self) -> List[str]:
        """Return list of keys from :data:`MANDATORY_KEYS` that are absent."""
        return [k for k in MANDATORY_KEYS if k not in self]

    def is_complete(self) -> bool:
        """True iff all mandatory keys are present."""
        return len(self.missing_keys()) == 0


class GovernanceLogger:
    """Append-only JSON-lines governance log.

    Parameters
    ----------
    log_dir : str | Path
        Directory where log files are written.  Created on first write.
    filename : str
        Base filename (without extension).  Defaults to
        ``"prediction_log"`` → writes ``prediction_log.jsonl``.
    validate : bool
        If True (default), log a warning when a record is missing mandatory
        keys but still write it (never silently drops records).
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "reports/governance",
        filename: str = "prediction_log",
        validate: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._filename = filename
        self._validate = validate

    @property
    def log_path(self) -> Path:
        return self._log_dir / f"{self._filename}.jsonl"

    def write(self, record: Dict[str, Any]) -> None:
        """Append a prediction record to the JSONL log file.

        Parameters
        ----------
        record : dict
            Prediction record.  Should contain the keys listed in
            :data:`MANDATORY_KEYS`; missing keys produce a warning when
            ``validate=True``.
        """
        rec = PredictionRecord(record)

        # Ensure asof_utc is present
        if "asof_utc" not in rec:
            rec["asof_utc"] = datetime.now(timezone.utc).isoformat()

        if self._validate:
            missing = rec.missing_keys()
            if missing:
                _log.warning(
                    "GovernanceLogger: record for %s is missing keys: %s",
                    rec.get("ticker", "unknown"),
                    missing,
                )

        self._log_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

        _log.debug(
            "GovernanceLogger: wrote record for %s → %s",
            rec.get("ticker", "unknown"),
            self.log_path,
        )

    def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write multiple records in a single call."""
        for rec in records:
            self.write(rec)

    def read_all(self) -> List[PredictionRecord]:
        """Read and return all records from the log file."""
        if not self.log_path.exists():
            return []
        records: List[PredictionRecord] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(PredictionRecord(json.loads(line)))
                    except json.JSONDecodeError as exc:
                        _log.warning("GovernanceLogger: skipping malformed line: %s", exc)
        return records

    def count(self) -> int:
        """Return number of records in the log."""
        return len(self.read_all())
