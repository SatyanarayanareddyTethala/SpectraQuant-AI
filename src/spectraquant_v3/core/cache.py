"""Cache manager for SpectraQuant-AI-V3."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spectraquant_v3.core.enums import RunMode
from spectraquant_v3.core.errors import CacheCorruptionError, CacheOnlyViolationError

if TYPE_CHECKING:
    import pandas as pd


class CacheManager:
    def __init__(self, cache_dir: str | Path, run_mode: RunMode = RunMode.NORMAL) -> None:
        self.cache_dir = Path(cache_dir)
        self.run_mode = run_mode
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str) -> Path:
        safe_key = key.replace("/", "__").replace("\\", "__")
        return self.cache_dir / f"{safe_key}.parquet"

    def get_freshness_path(self, key: str) -> Path:
        safe_key = key.replace("/", "__").replace("\\", "__")
        return self.cache_dir / f"{safe_key}.freshness.json"

    def _freshness_path(self, key: str) -> Path:
        safe_key = key.replace("/", "__").replace("\\", "__")
        return self.cache_dir / f"{safe_key}.fresh"

    def exists(self, key: str) -> bool:
        return self.get_path(key).exists()

    def list_keys(self) -> list[str]:
        return sorted(p.stem.replace("__", "/") for p in self.cache_dir.glob("*.parquet"))

    def assert_network_allowed(self, key: str) -> None:
        if self.run_mode == RunMode.TEST and not self.exists(key):
            raise CacheOnlyViolationError(
                f"TEST mode: cache miss for '{key}' at {self.get_path(key)}. "
                "Network calls are forbidden in test mode."
            )

    def should_skip_download(self, key: str) -> bool:
        if self.run_mode == RunMode.REFRESH:
            return False
        if self.run_mode == RunMode.TEST:
            return True
        return self.exists(key)

    def read_parquet(self, key: str) -> "pd.DataFrame":
        import pandas as pd

        self.assert_network_allowed(key)
        path = self.get_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            raise CacheCorruptionError(f"Failed to read cached parquet '{path}': {exc}") from exc

    def write_parquet(self, key: str, df: "pd.DataFrame") -> Path:
        if df.empty:
            raise ValueError("Cannot write empty DataFrame to cache")
        path = self.get_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
        except Exception as exc:  # noqa: BLE001
            raise CacheCorruptionError(f"Failed to write parquet '{path}': {exc}") from exc
        self._write_freshness_sidecar(key, df)
        return path

    def _write_freshness_sidecar(self, key: str, df: "pd.DataFrame") -> None:
        try:
            idx = df.index
            meta: dict[str, Any] = {
                "key": key,
                "ingested_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
                "rows": len(df),
                "min_timestamp": str(idx.min()) if len(idx) else "",
                "max_timestamp": str(idx.max()) if len(idx) else "",
            }
            self.get_freshness_path(key).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            return

    def write_freshness(self, key: str, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "key": key,
            "last_updated": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        self._freshness_path(key).write_text(json.dumps(payload), encoding="utf-8")

    def read_freshness(self, key: str) -> dict[str, Any] | None:
        # Prefer detailed sidecar written by write_parquet, then legacy .fresh format.
        json_sidecar = self.get_freshness_path(key)
        if json_sidecar.exists():
            try:
                return json.loads(json_sidecar.read_text(encoding="utf-8"))
            except Exception:
                return None
        legacy_sidecar = self._freshness_path(key)
        if legacy_sidecar.exists():
            try:
                return json.loads(legacy_sidecar.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def is_stale(
        self,
        key: str,
        max_age_hours: float | None = None,
        max_age_seconds: float | None = None,
    ) -> bool:
        if max_age_seconds is None:
            max_age_seconds = (max_age_hours if max_age_hours is not None else 24.0) * 3600

        info = self.read_freshness(key)
        if info is not None:
            ts = info.get("ingested_at") or info.get("last_updated")
            if ts:
                try:
                    updated = datetime.datetime.fromisoformat(ts)
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=datetime.timezone.utc)
                    age = (datetime.datetime.now(tz=datetime.timezone.utc) - updated).total_seconds()
                    return age > max_age_seconds
                except Exception:
                    pass

        path = self.get_path(key)
        if not path.exists():
            return True
        mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc)
        age = (datetime.datetime.now(tz=datetime.timezone.utc) - mtime).total_seconds()
        return age > max_age_seconds

    def validate_parquet(self, key: str) -> bool:
        import pandas as pd

        path = self.get_path(key)
        if not path.exists():
            return False
        try:
            pd.read_parquet(path)
            return True
        except Exception:
            return False
