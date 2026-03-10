"""Ingestion audit log for SpectraQuant-AI-V3."""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from spectraquant_v3.core.ingestion_result import IngestionResult

logger = logging.getLogger(__name__)


@dataclass
class IngestionAuditEntry:
    symbol: str
    provider: str
    status: str
    rows: int = 0
    run_id: str = ""
    error: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    recorded_at: str = field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    )

    @property
    def canonical_symbol(self) -> str:
        return self.symbol

    @property
    def success(self) -> bool:
        return self.status in {"success", "cache_hit"}

    @property
    def cache_hit(self) -> bool:
        return self.status == "cache_hit"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "canonical_symbol": self.symbol,
            "provider": self.provider,
            "status": self.status,
            "rows": self.rows,
            "rows_loaded": self.rows,
            "run_id": self.run_id,
            "error": self.error,
            "error_message": self.error,
            "metadata": self.metadata,
            "recorded_at": self.recorded_at,
            "success": self.success,
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IngestionAuditEntry":
        return cls(
            symbol=data.get("symbol") or data.get("canonical_symbol", ""),
            provider=data.get("provider", ""),
            status=data.get("status", "failure"),
            rows=int(data.get("rows", data.get("rows_loaded", 0) or 0)),
            run_id=data.get("run_id", ""),
            error=data.get("error", data.get("error_message", "")),
            metadata=data.get("metadata", {}),
            recorded_at=data.get("recorded_at", datetime.datetime.now(tz=datetime.timezone.utc).isoformat()),
        )

    @classmethod
    def from_ingestion_result(cls, result: IngestionResult, run_id: str = "") -> "IngestionAuditEntry":
        return cls(
            symbol=result.canonical_symbol,
            provider=result.provider,
            status="cache_hit" if result.cache_hit else ("success" if result.success else "failure"),
            rows=result.rows_loaded,
            run_id=run_id,
            error=result.error_message,
            metadata={"asset_class": result.asset_class, "error_code": result.error_code},
        )


class IngestionAuditLog:
    def __init__(self, log_path: str | Path, run_id: str = "") -> None:
        self.log_path = Path(log_path)
        self.run_id = run_id
        self._entries: list[IngestionAuditEntry] = []

    def record(self, *args, **kwargs) -> None:
        if args and isinstance(args[0], IngestionResult):
            entry = IngestionAuditEntry.from_ingestion_result(args[0], run_id=self.run_id)
        else:
            symbol = args[0]
            provider = args[1]
            status = args[2]
            entry = IngestionAuditEntry(
                symbol=symbol,
                provider=provider,
                status=status,
                rows=int(kwargs.get("rows", 0)),
                run_id=str(kwargs.get("run_id", self.run_id)),
                error=str(kwargs.get("error", "")),
                metadata=dict(kwargs.get("metadata", {})),
            )
        self._entries.append(entry)
        self._append_to_file(entry)

    def _append_to_file(self, entry: IngestionAuditEntry) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("IngestionAuditLog: failed to write audit entry: %s", exc)

    def read_entries(self, symbol: str | None = None, status: str | None = None) -> list[IngestionAuditEntry]:
        if not self.log_path.exists():
            return []
        out: list[IngestionAuditEntry] = []
        for raw in self.log_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = IngestionAuditEntry.from_dict(json.loads(raw))
            except Exception:
                continue
            if symbol and entry.symbol != symbol:
                continue
            if status and entry.status != status:
                continue
            out.append(entry)
        return out

    def load_from_file(self) -> list[dict]:
        return [e.to_dict() for e in self.read_entries()]

    def summary(self) -> dict:
        entries = self._entries
        total = len(entries)
        succeeded = sum(1 for e in entries if e.success)
        failed = total - succeeded
        cache_hits = sum(1 for e in entries if e.cache_hit)
        failed_symbols = [e.symbol for e in entries if not e.success]
        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "cache_hits": cache_hits,
            "failed_symbols": failed_symbols,
        }
