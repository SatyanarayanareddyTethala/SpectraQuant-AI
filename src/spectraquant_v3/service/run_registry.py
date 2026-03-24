"""Persistent run registry with idempotency and stage-event tracking."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from spectraquant_v3.service.models import RunRecord, StageEvent


class RunRegistry:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    asset_class TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    run_mode TEXT NOT NULL,
                    dry_run INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_code TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    at TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_run(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        asset_class: str,
        execution_mode: str,
        run_mode: str,
        dry_run: bool,
    ) -> RunRecord:
        now = self._now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return self._row_to_record(existing)

            conn.execute(
                """
                INSERT INTO runs (
                    run_id,idempotency_key,asset_class,execution_mode,run_mode,dry_run,
                    state,created_at,updated_at,error_code,error_message,result_json
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, '', '', '{}')
                """,
                (
                    run_id,
                    idempotency_key,
                    asset_class,
                    execution_mode,
                    run_mode,
                    1 if dry_run else 0,
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return self._row_to_record(row)

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return self._row_to_record(row) if row else None

    def get_by_idempotency_key(self, key: str) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE idempotency_key = ?", (key,)).fetchone()
            return self._row_to_record(row) if row else None

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_state(
        self,
        run_id: str,
        *,
        state: str,
        error_code: str = "",
        error_message: str = "",
        result: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET state = ?, updated_at = ?, error_code = ?, error_message = ?, result_json = ?
                WHERE run_id = ?
                """,
                (
                    state,
                    self._now(),
                    error_code,
                    error_message,
                    json.dumps(result or {}),
                    run_id,
                ),
            )

    def append_event(self, run_id: str, event: StageEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stage_events (run_id, stage, status, at, message, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    event.stage,
                    event.status,
                    event.at.isoformat(),
                    event.message,
                    json.dumps(event.details),
                ),
            )

    def list_events(self, run_id: str) -> list[StageEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT stage,status,at,message,details_json FROM stage_events WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
        return [
            StageEvent(
                stage=r["stage"],
                status=r["status"],
                at=datetime.fromisoformat(r["at"]),
                message=r["message"],
                details=json.loads(r["details_json"] or "{}"),
            )
            for r in rows
        ]

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            idempotency_key=row["idempotency_key"],
            asset_class=row["asset_class"],
            execution_mode=row["execution_mode"],
            run_mode=row["run_mode"],
            dry_run=bool(row["dry_run"]),
            state=row["state"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            result=json.loads(row["result_json"] or "{}"),
        )
