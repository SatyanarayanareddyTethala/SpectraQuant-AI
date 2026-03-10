"""Data retention utilities for historical market data."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/state.json")


def read_training_state(state_path: Path = STATE_PATH) -> dict | None:
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read training state %s: %s", state_path, exc)
        return None
    if isinstance(payload, dict) and "training" in payload:
        return payload.get("training")
    if isinstance(payload, dict) and "trained_at" in payload:
        return {
            "trained_at": payload.get("trained_at"),
            "retention_years": payload.get("retention_years"),
        }
    return None


def is_post_training(state_path: Path = STATE_PATH) -> bool:
    state = read_training_state(state_path)
    return bool(state and state.get("trained_at"))


def mark_training_complete(retention_years: int, state_path: Path = STATE_PATH) -> dict:
    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "retention_years": int(retention_years),
    }
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read existing state for merge %s: %s", state_path, exc)
            state = {}
    if not isinstance(state, dict):
        state = {}
    state["training"] = payload
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(state, indent=2))
    temp_path.replace(state_path)
    logger.info("Training state written to %s", state_path)
    return payload


def prune_dataframe_to_last_n_years(
    df: pd.DataFrame,
    years: int,
    date_column: str = "date",
) -> pd.DataFrame:
    pruned, meta = _prune_frame(df, years, date_column)
    if meta:
        _log_prune(meta, label="dataframe")
    return pruned


def prune_to_last_n_years(
    path_or_manifest: Path | str | Sequence[Path | str],
    years: int,
    date_column: str = "date",
) -> list[Path]:
    paths = _expand_paths(path_or_manifest)
    processed: list[Path] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.glob("*.csv")) + sorted(path.glob("*.parquet")):
                if _prune_file(child, years, date_column):
                    processed.append(child)
            continue
        if _prune_file(path, years, date_column):
            processed.append(path)
    return processed


def _expand_paths(path_or_manifest: Path | str | Sequence[Path | str]) -> list[Path]:
    if isinstance(path_or_manifest, (str, Path)):
        return [Path(path_or_manifest)]
    paths: list[Path] = []
    for item in path_or_manifest:
        paths.append(Path(item))
    return paths


def _prune_file(path: Path, years: int, date_column: str) -> bool:
    if not path.exists():
        return False
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        return False

    try:
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s for retention: %s", path, exc)
        return False

    pruned, meta = _prune_frame(df, years, date_column)
    if meta is None:
        logger.warning("Skipping retention for %s; no valid datetime data.", path)
        return False

    if meta["rows_after"] == meta["rows_before"]:
        _log_prune(meta, label=str(path))
        return False

    temp_path = path.with_suffix(path.suffix + ".tmp")
    write_index = isinstance(pruned.index, pd.DatetimeIndex) and "date" not in pruned.columns
    try:
        if suffix == ".parquet":
            pruned.to_parquet(temp_path, index=write_index)
        else:
            pruned.to_csv(temp_path, index=write_index)
        temp_path.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write retained data for %s: %s", path, exc)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return False

    _log_prune(meta, label=str(path))
    return True


def _prune_frame(
    df: pd.DataFrame,
    years: int,
    date_column: str,
) -> tuple[pd.DataFrame, dict | None]:
    if df is None or df.empty:
        return df, None

    had_date_column = date_column in df.columns
    working = df.copy()
    if had_date_column:
        working[date_column] = pd.to_datetime(working[date_column], utc=True, errors="coerce")
        working = working.dropna(subset=[date_column])
        working = working.sort_values(date_column)
        working = working.set_index(date_column, drop=False)
    else:
        if isinstance(working.index, pd.DatetimeIndex):
            index = pd.to_datetime(working.index, utc=True, errors="coerce")
            working.index = index
        else:
            coerced = pd.to_datetime(working.index, utc=True, errors="coerce")
            if coerced.isna().all():
                return df, None
            working.index = coerced
        working = working.sort_index()

    if working.empty:
        return working, None

    cutoff = pd.Timestamp.now('UTC').normalize() - pd.DateOffset(years=years)
    before_start = working.index.min()
    before_end = working.index.max()
    filtered = working.loc[working.index >= cutoff]
    after_start = filtered.index.min() if not filtered.empty else None
    after_end = filtered.index.max() if not filtered.empty else None

    meta = {
        "rows_before": int(len(working)),
        "rows_after": int(len(filtered)),
        "cutoff": cutoff,
        "before_start": before_start,
        "before_end": before_end,
        "after_start": after_start,
        "after_end": after_end,
    }
    if not had_date_column:
        filtered = filtered.sort_index()
    else:
        if filtered.index.name == date_column:
            filtered = filtered.sort_index()
        else:
            filtered = filtered.sort_values(date_column)

    return filtered, meta


def _log_prune(meta: dict, label: str) -> None:
    removed = meta["rows_before"] - meta["rows_after"]
    logger.info(
        "Retention (%s): kept=%s removed=%s cutoff=%s before=%s->%s after=%s->%s",
        label,
        meta["rows_after"],
        removed,
        meta["cutoff"],
        meta["before_start"],
        meta["before_end"],
        meta["after_start"],
        meta["after_end"],
    )
