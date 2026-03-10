from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    run_dir: Path | None
    manifest_path: Path | None
    created_at: datetime | None
    has_manifest: bool


def _parse_manifest_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _load_manifest_path(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    candidate = run_dir / MANIFEST_FILENAME
    if candidate.exists():
        return _load_manifest_path(candidate)
    manifests = sorted(run_dir.glob("run_manifest_*.json"))
    if manifests:
        return _load_manifest_path(manifests[-1])
    return None


def _manifest_run_id(manifest: dict[str, Any], fallback: str) -> str:
    return str(manifest.get("run_id") or manifest.get("run") or fallback)


def discover_runs(repo_root: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    reports_dir = repo_root / "reports"
    run_root = reports_dir / "run"
    if run_root.exists():
        for entry in run_root.iterdir():
            if entry.is_dir():
                manifest_path = entry / MANIFEST_FILENAME
                manifest = _load_manifest_path(manifest_path) if manifest_path.exists() else None
                created_at = _parse_manifest_timestamp(manifest.get("created_at")) if manifest else None
                run_id = _manifest_run_id(manifest, entry.name) if manifest else entry.name
                runs.append(
                    RunInfo(
                        run_id=run_id,
                        run_dir=entry,
                        manifest_path=manifest_path if manifest_path.exists() else None,
                        created_at=created_at,
                        has_manifest=manifest_path.exists(),
                    )
                )
        for manifest_path in sorted(run_root.glob("run_manifest_*.json")):
            manifest = _load_manifest_path(manifest_path)
            created_at = _parse_manifest_timestamp(manifest.get("created_at")) if manifest else None
            run_id = _manifest_run_id(manifest or {}, manifest_path.stem)
            runs.append(
                RunInfo(
                    run_id=run_id,
                    run_dir=manifest_path.parent,
                    manifest_path=manifest_path,
                    created_at=created_at,
                    has_manifest=manifest is not None,
                )
            )
    runs.append(RunInfo(run_id="latest", run_dir=None, manifest_path=None, created_at=None, has_manifest=False))
    runs = _dedupe_runs(runs)
    return sorted(runs, key=_run_sort_key, reverse=True)


def _dedupe_runs(runs: Iterable[RunInfo]) -> list[RunInfo]:
    seen = {}
    for run in runs:
        if run.run_id not in seen:
            seen[run.run_id] = run
    return list(seen.values())


def _run_sort_key(run: RunInfo) -> tuple[int, datetime]:
    if run.run_id == "latest":
        return (-1, datetime.min)
    created_at = run.created_at or datetime.min
    return (1, created_at)


def resolve_manifest_paths(manifest: dict[str, Any], run_dir: Path | None) -> dict[str, Path]:
    paths = manifest.get("paths") if isinstance(manifest, dict) else None
    if not isinstance(paths, dict):
        return {}
    resolved = {}
    for key, value in paths.items():
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute() and run_dir is not None:
            path = run_dir / path
        resolved[key] = path
    return resolved
