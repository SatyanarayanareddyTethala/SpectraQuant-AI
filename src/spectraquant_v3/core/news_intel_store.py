"""Cached news intelligence store for deterministic backtests.

Separates live discovery (via providers) from cached normalised event records
used during backtest feature generation.  This ensures that:

1. **Live runs** call the provider, normalise, and persist records.
2. **Backtest / TEST runs** read from the persisted cache only — no network.
3. **Feature generation** always consumes from the cache, never directly from
   providers, guaranteeing reproducibility.

Storage format
--------------
Records are stored as Parquet files keyed by canonical symbol.  Each row
is a serialised :class:`~spectraquant_v3.core.news_schema.NewsIntelligenceRecord`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from spectraquant_v3.core.news_schema import (
    NewsIntelligenceRecord,
    validate_news_intelligence_record,
)

logger = logging.getLogger(__name__)


class NewsIntelligenceStore:
    """Append-only, deduplication-aware cache for news intelligence records.

    Args:
        base_dir: Root directory for the news intelligence cache.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_key(row: dict[str, Any]) -> str:
        """Return a deduplication key for a single record dict."""
        return f"{row.get('canonical_symbol', '')}|{row.get('timestamp', '')}|{row.get('event_type', '')}"

    def _parquet_path(self, canonical_symbol: str) -> Path:
        return self.base_dir / f"news_intel__{canonical_symbol.upper()}.parquet"

    # ------------------------------------------------------------------
    # Write (live discovery path)
    # ------------------------------------------------------------------

    def write_records(
        self,
        canonical_symbol: str,
        records: list[NewsIntelligenceRecord],
    ) -> Path:
        """Persist records to the cache, merging with any existing data.

        Args:
            canonical_symbol: Upper-case canonical ticker.
            records:          Records to persist.

        Returns:
            Path of the written Parquet file.
        """
        path = self._parquet_path(canonical_symbol)

        new_rows = [r.to_dict() for r in records]

        # Serialise source_urls as JSON strings for Parquet compatibility.
        for row in new_rows:
            row["source_urls"] = json.dumps(row["source_urls"])

        existing_df = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()

        merged = pd.concat([existing_df, new_df], ignore_index=True)

        if not merged.empty:
            merged["_dedup"] = merged.apply(
                lambda r: self._dedup_key(r.to_dict()), axis=1
            )
            merged = merged.drop_duplicates(subset=["_dedup"], keep="first")
            merged = merged.drop(columns=["_dedup"])

        merged.to_parquet(path, index=False)
        logger.debug(
            "NewsIntelligenceStore: wrote %d records for %s to %s",
            len(merged),
            canonical_symbol,
            path,
        )
        return path

    # ------------------------------------------------------------------
    # Read (backtest / feature generation path)
    # ------------------------------------------------------------------

    def read_records(self, canonical_symbol: str) -> list[dict[str, Any]]:
        """Read cached records for a symbol.

        Args:
            canonical_symbol: Upper-case canonical ticker.

        Returns:
            List of record dicts.  Returns an empty list on cache miss.
        """
        path = self._parquet_path(canonical_symbol)
        if not path.exists():
            return []

        df = pd.read_parquet(path)
        rows: list[dict[str, Any]] = df.to_dict(orient="records")

        # Deserialise source_urls from JSON strings.
        for row in rows:
            if isinstance(row.get("source_urls"), str):
                try:
                    row["source_urls"] = json.loads(row["source_urls"])
                except json.JSONDecodeError:
                    row["source_urls"] = []

        return rows

    def has_records(self, canonical_symbol: str) -> bool:
        """Return ``True`` when cached records exist for *canonical_symbol*."""
        return self._parquet_path(canonical_symbol).exists()

    def read_as_dataframe(self, canonical_symbol: str) -> pd.DataFrame:
        """Return cached records as a DataFrame for feature generation.

        This is the recommended entry-point for backtest feature pipelines.
        """
        path = self._parquet_path(canonical_symbol)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
