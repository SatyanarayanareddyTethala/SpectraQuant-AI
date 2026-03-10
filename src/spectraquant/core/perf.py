"""Performance budget enforcement for pipeline stages."""
from __future__ import annotations

from contextlib import contextmanager
import logging
import platform
import time
from typing import Dict, Iterator

import resource

logger = logging.getLogger(__name__)


def _rss_mb_from_ru_maxrss(ru_maxrss: float, system: str) -> float:
    if system == "Darwin":
        return ru_maxrss / (1024.0 * 1024.0)
    return ru_maxrss / 1024.0

def _current_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    ru_maxrss = float(getattr(usage, "ru_maxrss", 0.0))
    return _rss_mb_from_ru_maxrss(ru_maxrss, platform.system())


def _stage_limits(stage: str, config: Dict) -> tuple[float | None, float | None]:
    perf_cfg = config.get("perf", {}) if config else {}
    default_seconds = perf_cfg.get("max_seconds")
    default_mb = perf_cfg.get("max_mb")
    stage_cfg = perf_cfg.get("stages", {}).get(stage, {}) if isinstance(perf_cfg.get("stages"), dict) else {}
    max_seconds = stage_cfg.get("max_seconds", default_seconds)
    max_mb = stage_cfg.get("max_mb", default_mb)
    return (float(max_seconds) if max_seconds is not None else None, float(max_mb) if max_mb is not None else None)


@contextmanager
def enforce_stage_budget(stage: str, config: Dict) -> Iterator[None]:
    """Enforce runtime + memory budgets for a pipeline stage."""

    research_mode = bool(config.get("research_mode")) if config else False
    max_seconds, max_mb = _stage_limits(stage, config)
    start_time = time.monotonic()
    start_mb = _current_rss_mb()
    exc_raised: BaseException | None = None
    try:
        yield
    except BaseException as exc:  # noqa: BLE001
        exc_raised = exc
        raise
    finally:
        if not research_mode:
            elapsed = time.monotonic() - start_time
            end_mb = _current_rss_mb()
            used_mb = max(0.0, end_mb - start_mb)
            logger.info(
                "Stage %s perf: elapsed=%.2fs rss_delta=%.2fMB",
                stage,
                elapsed,
                used_mb,
            )
            if exc_raised is None:
                if max_seconds is not None and elapsed > max_seconds:
                    raise RuntimeError(
                        f"Performance budget exceeded for {stage}: {elapsed:.2f}s > {max_seconds:.2f}s"
                    )
                if max_mb is not None and used_mb > max_mb:
                    raise RuntimeError(
                        f"Performance budget exceeded for {stage}: {used_mb:.2f}MB > {max_mb:.2f}MB"
                    )
