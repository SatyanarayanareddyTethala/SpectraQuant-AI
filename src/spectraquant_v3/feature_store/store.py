"""Feature store for SpectraQuant-AI-V3.

Persists feature frames as Parquet files with JSON metadata sidecars.
Provides optional DuckDB-backed querying when DuckDB is available; falls
back gracefully to pure-Pandas operations when it is not installed.

Storage layout under *store_root*::

    <store_root>/
        <asset_class>/
            <feature_name>/
                <feature_version>/
                    <symbol>.parquet
                    <symbol>.meta.json

Usage::

    from spectraquant_v3.feature_store import FeatureStore

    store = FeatureStore("data/feature_store")

    store.write_feature_frame(
        df=feature_df,
        feature_name="momentum_20d",
        feature_version="1.0.0",
        symbol="BTC",
        asset_class="crypto",
        source_run_id="run_001",
    )

    df = store.read_feature_frame("momentum_20d", "1.0.0", "BTC", "crypto")

    sets = store.list_feature_sets()
    rows = store.query_feature_data(feature_name="momentum_20d", symbol="BTC")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from spectraquant_v3.feature_store.metadata import FeatureSetMetadata

logger = logging.getLogger(__name__)


class FeatureStore:
    """Parquet-backed persistent feature store.

    Args:
        store_root: Root directory for feature data.  Sub-directories are
                    created automatically on first write.
    """

    def __init__(self, store_root: str | Path = "data/feature_store") -> None:
        self.store_root = Path(store_root)
        self.store_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal path helpers
    # ------------------------------------------------------------------

    def _symbol_dir(
        self,
        feature_name: str,
        feature_version: str,
        asset_class: str,
    ) -> Path:
        """Return the directory where a specific feature version is stored."""
        return self.store_root / asset_class / feature_name / feature_version

    def _parquet_path(
        self,
        feature_name: str,
        feature_version: str,
        symbol: str,
        asset_class: str,
    ) -> Path:
        safe_sym = symbol.replace("/", "__").replace("\\", "__")
        return (
            self._symbol_dir(feature_name, feature_version, asset_class)
            / f"{safe_sym}.parquet"
        )

    def _meta_path(
        self,
        feature_name: str,
        feature_version: str,
        symbol: str,
        asset_class: str,
    ) -> Path:
        safe_sym = symbol.replace("/", "__").replace("\\", "__")
        return (
            self._symbol_dir(feature_name, feature_version, asset_class)
            / f"{safe_sym}.meta.json"
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_feature_frame(
        self,
        df: pd.DataFrame,
        feature_name: str,
        feature_version: str,
        symbol: str,
        asset_class: str,
        source_run_id: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> FeatureSetMetadata:
        """Persist a feature DataFrame and its metadata.

        The DataFrame is written atomically to Parquet using a
        ``.tmp`` staging file that is renamed on success, guarding against
        partial writes.

        Args:
            df:               Feature DataFrame with a DatetimeIndex.
            feature_name:     Logical name of the feature (e.g. ``"rsi_14"``).
            feature_version:  Semantic version string (e.g. ``"1.0.0"``).
            symbol:           Canonical symbol (e.g. ``"BTC"``).
            asset_class:      ``"crypto"`` or ``"equity"``.
            source_run_id:    Identifier of the producing pipeline run.
            extra_metadata:   Additional key/value pairs to embed in the sidecar.

        Returns:
            The :class:`~spectraquant_v3.feature_store.metadata.FeatureSetMetadata`
            that was written alongside the Parquet file.

        Raises:
            ValueError: When *df* is empty.
        """
        if df.empty:
            raise ValueError(
                f"write_feature_frame: empty DataFrame for "
                f"{feature_name}/{feature_version}/{symbol}"
            )

        target_dir = self._symbol_dir(feature_name, feature_version, asset_class)
        target_dir.mkdir(parents=True, exist_ok=True)

        pq_path = self._parquet_path(feature_name, feature_version, symbol, asset_class)
        tmp_path = pq_path.with_suffix(".tmp.parquet")

        # Atomic write: stage → rename
        df.to_parquet(tmp_path, index=True)
        tmp_path.rename(pq_path)

        # Build and persist metadata sidecar
        idx = df.index
        date_start = str(idx.min().date()) if len(idx) > 0 else ""
        date_end = str(idx.max().date()) if len(idx) > 0 else ""

        meta = FeatureSetMetadata(
            feature_name=feature_name,
            feature_version=feature_version,
            symbol=symbol,
            asset_class=asset_class,
            source_run_id=source_run_id,
            date_start=date_start,
            date_end=date_end,
            row_count=len(df),
            feature_columns=list(df.columns),
            metadata=extra_metadata or {},
        )
        meta.write(self._meta_path(feature_name, feature_version, symbol, asset_class))

        logger.debug(
            "FeatureStore: wrote %d rows for %s/%s/%s",
            len(df),
            feature_name,
            feature_version,
            symbol,
        )
        return meta

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_feature_frame(
        self,
        feature_name: str,
        feature_version: str,
        symbol: str,
        asset_class: str,
    ) -> pd.DataFrame:
        """Load a persisted feature frame.

        Args:
            feature_name:    Feature name.
            feature_version: Feature version.
            symbol:          Canonical symbol.
            asset_class:     Asset class.

        Returns:
            DataFrame loaded from Parquet.

        Raises:
            FileNotFoundError: When no matching feature file exists.
        """
        pq_path = self._parquet_path(feature_name, feature_version, symbol, asset_class)
        if not pq_path.exists():
            raise FileNotFoundError(
                f"No feature frame found at {pq_path}.  "
                f"Have you called write_feature_frame first?"
            )
        return pd.read_parquet(pq_path)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_feature_sets(
        self,
        asset_class: str | None = None,
        feature_name: str | None = None,
    ) -> list[FeatureSetMetadata]:
        """Return all stored feature-set metadata records.

        Args:
            asset_class:  Filter to a specific asset class (optional).
            feature_name: Filter to a specific feature name (optional).

        Returns:
            Sorted list of :class:`FeatureSetMetadata` objects.
        """
        results: list[FeatureSetMetadata] = []

        search_root = self.store_root
        if asset_class:
            search_root = search_root / asset_class
        if feature_name and asset_class:
            search_root = search_root / feature_name

        for meta_path in sorted(search_root.rglob("*.meta.json")):
            try:
                meta = FeatureSetMetadata.read(meta_path)
                if feature_name and meta.feature_name != feature_name:
                    continue
                results.append(meta)
            except Exception as exc:  # pragma: no cover
                logger.warning("FeatureStore: failed to read %s: %s", meta_path, exc)

        return results

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_feature_data(
        self,
        feature_name: str | None = None,
        symbol: str | None = None,
        asset_class: str | None = None,
        feature_version: str | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> pd.DataFrame:
        """Query feature data across all persisted frames matching the filters.

        Attempts to use DuckDB for efficient columnar querying when available;
        falls back to Pandas concat when DuckDB is not installed.

        Args:
            feature_name:    Filter by feature name.
            symbol:          Filter by symbol.
            asset_class:     Filter by asset class.
            feature_version: Filter by feature version.
            date_start:      Include rows on or after this ISO-8601 date.
            date_end:        Include rows on or before this ISO-8601 date.

        Returns:
            Concatenated DataFrame, or an empty DataFrame if no data matches.
        """
        # Collect matching metadata entries
        metas = self.list_feature_sets(
            asset_class=asset_class,
            feature_name=feature_name,
        )
        if feature_version:
            metas = [m for m in metas if m.feature_version == feature_version]
        if symbol:
            metas = [m for m in metas if m.symbol == symbol]

        if not metas:
            return pd.DataFrame()

        # Attempt DuckDB path first
        try:
            return self._query_duckdb(
                metas,
                feature_name=feature_name,
                symbol=symbol,
                date_start=date_start,
                date_end=date_end,
            )
        except Exception:
            pass

        # Pandas fallback
        return self._query_pandas(
            metas,
            date_start=date_start,
            date_end=date_end,
        )

    def _query_duckdb(
        self,
        metas: list[FeatureSetMetadata],
        *,
        feature_name: str | None,
        symbol: str | None,
        date_start: str | None,
        date_end: str | None,
    ) -> pd.DataFrame:
        """Query using DuckDB for efficient columnar scans."""
        import duckdb  # noqa: PLC0415

        frames = []
        for meta in metas:
            pq_path = self._parquet_path(
                meta.feature_name,
                meta.feature_version,
                meta.symbol,
                meta.asset_class,
            )
            if not pq_path.exists():
                continue

            where_clauses: list[str] = []
            if date_start:
                where_clauses.append(f"index >= '{date_start}'")
            if date_end:
                where_clauses.append(f"index <= '{date_end}'")

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            try:
                df = duckdb.query(
                    f"SELECT * FROM read_parquet('{pq_path}') {where_sql}"
                ).df()
                df["_symbol"] = meta.symbol
                df["_feature_name"] = meta.feature_name
                df["_feature_version"] = meta.feature_version
                df["_asset_class"] = meta.asset_class
                frames.append(df)
            except Exception as exc:
                logger.debug("DuckDB query failed for %s: %s", pq_path, exc)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _query_pandas(
        self,
        metas: list[FeatureSetMetadata],
        *,
        date_start: str | None,
        date_end: str | None,
    ) -> pd.DataFrame:
        """Query using Pandas concat as DuckDB fallback."""
        frames = []
        for meta in metas:
            pq_path = self._parquet_path(
                meta.feature_name,
                meta.feature_version,
                meta.symbol,
                meta.asset_class,
            )
            if not pq_path.exists():
                continue
            try:
                df = pd.read_parquet(pq_path)
                if date_start:
                    df = df[df.index >= pd.Timestamp(date_start)]
                if date_end:
                    df = df[df.index <= pd.Timestamp(date_end)]
                df = df.copy()
                df["_symbol"] = meta.symbol
                df["_feature_name"] = meta.feature_name
                df["_feature_version"] = meta.feature_version
                df["_asset_class"] = meta.asset_class
                frames.append(df)
            except Exception as exc:
                logger.warning("Pandas read failed for %s: %s", pq_path, exc)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames)
