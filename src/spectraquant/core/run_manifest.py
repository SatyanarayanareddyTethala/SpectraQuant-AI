"""Run manifest writer for SpectraQuant pipeline provenance.

Every pipeline run – successful or not – should produce a manifest that
records the parameters, effective universe, and per-stage outcome so that
results are auditable and reproducible.

Schema
------
run_id              – stable hash of (config_subset, from_news, date_bucket)
as_of_utc           – ISO-8601 UTC timestamp of the run
from_news           – whether news-first mode was active
news_universe_enabled – value of config.news_universe.enabled
candidate_count     – number of news candidates produced (0 on early exit)
effective_tickers   – list of tickers used downstream ([] on early exit)
stage_status        – dict mapping stage name → "ok" | "skipped" | reason
exit_reason         – human-readable reason if pipeline exited early (else null)
schema_version      – "1"
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1"
_MANIFEST_DIR = Path("reports/run")


def _compute_run_id(config: dict, from_news: bool, as_of_utc: str) -> str:
    """Compute a stable content-addressable run identifier.

    The ID is derived from a canonical subset of the config (news query
    parameters + from_news flag + date bucket) so the same inputs always
    produce the same ID, making runs comparable and de-duplicable.
    """
    news_cfg = config.get("news_universe", {}) if isinstance(config, dict) else {}
    perf_cfg = config.get("perf", {}) if isinstance(config, dict) else {}

    canonical = {
        "from_news": from_news,
        "news_enabled": bool(news_cfg.get("enabled", False)),
        "lookback_hours": news_cfg.get("lookback_hours", 12),
        "query_terms": sorted(news_cfg.get("query_terms", [])),
        "max_candidates": news_cfg.get("max_candidates", 50),
        "date_bucket": as_of_utc[:13],  # hourly bucket, e.g. "2026-02-21T10"
    }
    raw = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_run_manifest(
    *,
    config: dict,
    from_news: bool,
    effective_tickers: list[str],
    candidate_count: int,
    stage_status: dict[str, str] | None = None,
    exit_reason: str | None = None,
) -> dict[str, Any]:
    """Build a run manifest dict without writing it to disk."""
    as_of_utc = datetime.now(timezone.utc).isoformat()
    run_id = _compute_run_id(config, from_news, as_of_utc)
    news_cfg = config.get("news_universe", {}) if isinstance(config, dict) else {}
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "as_of_utc": as_of_utc,
        "from_news": from_news,
        "news_universe_enabled": bool(news_cfg.get("enabled", False)),
        "candidate_count": candidate_count,
        "effective_tickers": list(effective_tickers),
        "stage_status": stage_status or {},
        "exit_reason": exit_reason,
    }


def write_run_manifest(
    manifest: dict[str, Any],
    *,
    output_dir: Path | str | None = None,
) -> Path:
    """Persist a run manifest to *output_dir* (default: ``reports/run``).

    Returns the path of the written file.
    """
    dest = Path(output_dir) if output_dir else _MANIFEST_DIR
    dest.mkdir(parents=True, exist_ok=True)

    run_id = manifest.get("run_id", "unknown")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = dest / f"run_manifest_{ts}_{run_id}.json"

    filename.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Run manifest written to %s", filename)
    return filename


def write_early_exit_manifest(
    config: dict,
    *,
    from_news: bool,
    exit_reason: str,
    output_dir: Path | str | None = None,
) -> Path:
    """Convenience helper: build + write an early-exit manifest.

    Called when the pipeline cannot proceed (e.g. no news candidates in
    news-first mode) so that every run – even aborted ones – leaves an
    audit trail.
    """
    manifest = build_run_manifest(
        config=config,
        from_news=from_news,
        effective_tickers=[],
        candidate_count=0,
        stage_status={
            "download": "skipped_due_to_empty_candidates",
            "predict": "skipped_due_to_empty_candidates",
            "signals": "skipped_due_to_empty_candidates",
            "portfolio": "skipped_due_to_empty_candidates",
        },
        exit_reason=exit_reason,
    )
    return write_run_manifest(manifest, output_dir=output_dir)
