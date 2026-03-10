"""QA / data availability matrix for SpectraQuant-AI-V3.

Every run must produce a QA matrix with one row per symbol so that data
completeness is auditable.  Missing *required* baseline data must abort the
run; missing *optional* data degrades gracefully.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectraquant_v3.core.errors import EmptyUniverseError
from spectraquant_v3.core.schema import QARow


class QAMatrix:
    """Collects and validates QA rows across all symbols in a run.

    Args:
        run_id:      Identifier for the parent run.
        asset_class: Asset-class string (``'crypto'`` or ``'equity'``).
    """

    def __init__(self, run_id: str, asset_class: str) -> None:
        self.run_id = run_id
        self.asset_class = asset_class
        self.rows: list[QARow] = []
        # Maps canonical_symbol → (QARow, list_index) for O(1) lookup and
        # O(1) in-place replacement without scanning the list.
        self._index: dict[str, tuple[QARow, int]] = {}

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def add(self, row: QARow) -> None:
        """Append a :class:`QARow` to the matrix.

        If a row for ``row.canonical_symbol`` already exists it is replaced.
        Insertion order is preserved; replacement keeps the original position.
        Both the append and replace paths are O(1).
        """
        if row.canonical_symbol in self._index:
            # O(1) in-place replacement: reuse stored list index.
            _, idx = self._index[row.canonical_symbol]
            self.rows[idx] = row
        else:
            # Capture the index BEFORE appending so it equals len(rows) - 1 post-append.
            idx = len(self.rows)
            self.rows.append(row)
        self._index[row.canonical_symbol] = (row, idx)

    def get_row(self, canonical_symbol: str) -> QARow | None:
        """Return the :class:`QARow` for *canonical_symbol*, or *None*.

        Args:
            canonical_symbol: Symbol to look up.

        Returns:
            Matching :class:`QARow` or ``None`` if not present.
        """
        entry = self._index.get(canonical_symbol)
        return entry[0] if entry is not None else None

    def mark_failed(
        self,
        canonical_symbol: str,
        error_code: str,
        note: str = "",
    ) -> None:
        """Update an existing row to reflect a stage failure.

        If no row exists for *canonical_symbol* a minimal failed row is
        created automatically so failures are always recorded.

        Args:
            canonical_symbol: Symbol that failed.
            error_code:       Short uppercase code, e.g. ``'NO_OHLCV'``.
            note:             Optional human-readable description.
        """
        entry = self._index.get(canonical_symbol)
        row: QARow | None = entry[0] if entry is not None else None
        if row is None:
            row = QARow(
                run_id=self.run_id,
                as_of=datetime.now(timezone.utc).isoformat(),
                canonical_symbol=canonical_symbol,
                asset_class=self.asset_class,
            )
            self.add(row)
        row.stage_status = "FAILED"
        if error_code not in row.error_codes:
            row.error_codes.append(error_code)
        if note:
            row.notes = note if not row.notes else f"{row.notes}; {note}"

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def all_missing_ohlcv(self) -> bool:
        """Return True when every row has ``has_ohlcv=False``."""
        return bool(self.rows) and all(not r.has_ohlcv for r in self.rows)

    def assert_ohlcv_available(self) -> None:
        """Raise :exc:`EmptyUniverseError` when no symbol has OHLCV data.

        This is a hard guard: a run with zero price data must never proceed
        to the signal or allocation stages.

        Raises:
            EmptyUniverseError: If ``all_missing_ohlcv()`` is True.
        """
        if self.all_missing_ohlcv():
            raise EmptyUniverseError(
                f"EMPTY_OHLCV_UNIVERSE: no {self.asset_class} symbols have OHLCV "
                f"data in run '{self.run_id}'. "
                "Check your universe, cache, and provider configuration."
            )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        """Return the matrix as a list of plain dictionaries."""
        return [r.__dict__ for r in self.rows]

    def summary(self) -> dict[str, Any]:
        """Return high-level counts for logging / manifest inclusion."""
        total = len(self.rows)
        with_ohlcv = sum(1 for r in self.rows if r.has_ohlcv)
        failed = sum(1 for r in self.rows if r.stage_status == "FAILED")
        return {
            "total_symbols": total,
            "symbols_with_ohlcv": with_ohlcv,
            "symbols_missing_ohlcv": total - with_ohlcv,
            "symbols_failed": failed,
        }

    def write(self, output_dir: str | Path) -> Path:
        """Persist the QA matrix as a JSON file.

        Args:
            output_dir: Directory to write the file.  Created if absent.

        Returns:
            Path of the written JSON file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"qa_matrix_{self.asset_class}_{ts}_{self.run_id}.json"
        path = out / filename
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "asset_class": self.asset_class,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": self.summary(),
            "rows": self.to_records(),
        }
        path.write_text(json.dumps(payload, indent=2))
        return path
