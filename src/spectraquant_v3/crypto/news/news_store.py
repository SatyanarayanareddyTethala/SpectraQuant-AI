from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class NewsStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _url_hash(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def _dedupe_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for row in rows:
            key = (str(row.get("article_id", "")), self._url_hash(str(row.get("url", ""))))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _load_existing_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def write_jsonl(self, key: str, rows: list[dict[str, Any]]) -> Path:
        path = self.base_dir / f"{key}.jsonl"
        existing = self._load_existing_jsonl(path)
        merged = self._dedupe_rows(existing + rows)
        path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in merged) + ("\n" if merged else ""))
        return path

    def write_parquet(self, key: str, rows: list[dict[str, Any]]) -> Path:
        path = self.base_dir / f"{key}.parquet"
        existing_df = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        new_df = pd.DataFrame(rows)
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        if not merged.empty:
            merged["_url_hash"] = merged["url"].astype(str).map(self._url_hash)
            merged = merged.drop_duplicates(subset=["article_id", "_url_hash"], keep="first")
            merged = merged.drop(columns=["_url_hash"])
        merged.to_parquet(path, index=False)
        return path
