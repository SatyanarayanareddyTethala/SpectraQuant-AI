"""Feature set metadata for SpectraQuant-AI-V3 feature store."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FeatureSetMetadata:
    """Metadata describing a persisted feature set.

    Attributes:
        feature_name:    Name of the feature (e.g. ``"momentum_20d"``).
        feature_version: Semantic version string (e.g. ``"1.0.0"``).
        symbol:          Canonical symbol the feature belongs to.
        asset_class:     ``"crypto"`` or ``"equity"``.
        source_run_id:   Identifier of the pipeline run that generated this set.
        date_start:      ISO-8601 date string of the earliest row.
        date_end:        ISO-8601 date string of the latest row.
        row_count:       Number of rows in the persisted frame.
        feature_columns: Ordered list of feature column names.
        metadata:        Free-form dict for additional provenance.
    """

    feature_name: str
    feature_version: str
    symbol: str
    asset_class: str
    source_run_id: str = ""
    date_start: str = ""
    date_end: str = ""
    row_count: int = 0
    feature_columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "feature_name": self.feature_name,
            "feature_version": self.feature_version,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "source_run_id": self.source_run_id,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "row_count": self.row_count,
            "feature_columns": self.feature_columns,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureSetMetadata":
        """Construct a :class:`FeatureSetMetadata` from a dictionary."""
        return cls(
            feature_name=d["feature_name"],
            feature_version=d["feature_version"],
            symbol=d["symbol"],
            asset_class=d["asset_class"],
            source_run_id=d.get("source_run_id", ""),
            date_start=d.get("date_start", ""),
            date_end=d.get("date_end", ""),
            row_count=d.get("row_count", 0),
            feature_columns=d.get("feature_columns", []),
            metadata=d.get("metadata", {}),
        )

    def write(self, path: str | Path) -> None:
        """Write metadata as a JSON sidecar file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def read(cls, path: str | Path) -> "FeatureSetMetadata":
        """Read metadata from a JSON sidecar file."""
        return cls.from_dict(json.loads(Path(path).read_text()))
