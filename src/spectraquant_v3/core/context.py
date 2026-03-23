"""Run context for SpectraQuant-AI-V3.

``RunContext`` is the single object threaded through all pipeline stages.
It holds the run's identity, configuration, path layout, cache, manifest,
and QA matrix so that every stage receives consistent, validated state
without relying on module-level globals.

Usage (context manager form, recommended)::

    with RunContext.create(
        asset_class=AssetClass.CRYPTO,
        run_mode=RunMode.NORMAL,
        config=get_crypto_config(),
    ) as ctx:
        ctx.manifest.mark_stage(RunStage.UNIVERSE.value, "ok", n_symbols=10)
        ctx.qa_matrix.add(QARow(...))

    # manifest.write() is called automatically on exit, even if an exception
    # is raised inside the ``with`` block.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectraquant_v3.core.cache import CacheManager
from spectraquant_v3.core.enums import AssetClass, RunMode, RunStatus
from spectraquant_v3.core.manifest import RunManifest
from spectraquant_v3.core.paths import ProjectPaths, RunPaths
from spectraquant_v3.core.qa import QAMatrix


class RunContext:
    """All state for a single pipeline run.

    Attributes:
        run_id:      Short unique identifier (8 hex chars by default).
        as_of:       ISO-8601 UTC timestamp recorded at context creation.
        asset_class: Which pipeline is running (CRYPTO or EQUITY).
        run_mode:    Cache mode (NORMAL / TEST / REFRESH).
        config:      Merged configuration dictionary.
        paths:       :class:`~spectraquant_v3.core.paths.RunPaths` for this run.
        cache:       :class:`~spectraquant_v3.core.cache.CacheManager` wired to the
                     per-asset-class cache directory.
        manifest:    :class:`~spectraquant_v3.core.manifest.RunManifest` that will
                     be written on exit.
        qa_matrix:   :class:`~spectraquant_v3.core.qa.QAMatrix` collecting one row
                     per symbol.
    """

    def __init__(
        self,
        run_id: str,
        asset_class: AssetClass,
        run_mode: RunMode,
        config: dict[str, Any],
        paths: RunPaths,
        cache: CacheManager,
        manifest: RunManifest,
        qa_matrix: QAMatrix,
    ) -> None:
        self.run_id = run_id
        self.as_of = datetime.now(timezone.utc).isoformat()
        self.asset_class = asset_class
        self.run_mode = run_mode
        self.config = config
        self.paths = paths
        self.cache = cache
        self.manifest = manifest
        self.qa_matrix = qa_matrix

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        asset_class: AssetClass,
        run_mode: RunMode = RunMode.NORMAL,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
        project_root: str | Path | None = None,
    ) -> "RunContext":
        """Build a fully-wired :class:`RunContext`.

        This is the canonical entry point for pipeline orchestrators.  All
        sub-objects are constructed consistently so pipeline code never needs
        to wire them together manually.

        Args:
            asset_class:   Pipeline asset class.
            run_mode:      Cache behaviour mode.
            config:        Merged configuration dict.  When *None*, an empty
                           dict is used (callers should always supply config).
            run_id:        Override the auto-generated run identifier.
            project_root:  Override repo-root discovery.  Useful in tests.

        Returns:
            A fully-initialised :class:`RunContext`.
        """
        resolved_run_id = run_id or str(uuid.uuid4())[:8]
        resolved_config = config or {}

        run_cfg = resolved_config.get("run", {})
        cache_cfg = resolved_config.get("cache", {})
        project = ProjectPaths(
            root=project_root,
            cache_root=cache_cfg.get("root"),
            reports_root=run_cfg.get("reports_root"),
        )
        run_paths = RunPaths.from_project(
            project=project,
            run_id=resolved_run_id,
            asset_class=asset_class,
        )

        # Manifest always lives in the per-run directory so every run's output
        # is self-contained under reports/<asset_class>/<run_id>/.
        # Any reports_dir specified in config is used as the parent for the
        # asset-class subdirectory only; it does not override the per-run path.
        manifest = RunManifest(
            asset_class=asset_class,
            run_mode=run_mode,
            run_id=resolved_run_id,
            output_dir=run_paths.manifest_dir,
        )

        cache = CacheManager(
            cache_dir=run_paths.cache_dir,
            run_mode=run_mode,
        )

        qa_matrix = QAMatrix(
            run_id=resolved_run_id,
            asset_class=asset_class.value,
        )

        return cls(
            run_id=resolved_run_id,
            asset_class=asset_class,
            run_mode=run_mode,
            config=resolved_config,
            paths=run_paths,
            cache=cache,
            manifest=manifest,
            qa_matrix=qa_matrix,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RunContext":
        """Enter the run context, creating output directories."""
        self.paths.make_dirs()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        """Exit the run context.

        On normal exit, marks the manifest as SUCCESS (unless already marked).
        On exception exit, appends the error to the manifest.
        In both cases, :meth:`~spectraquant_v3.core.manifest.RunManifest.write`
        is always called so the manifest is persisted.

        Returns:
            False — exceptions are never suppressed.
        """
        if exc_val is None:
            if self.manifest.status == RunStatus.ABORTED:
                self.manifest.mark_complete(RunStatus.SUCCESS)
        else:
            self.manifest.add_error(
                f"{type(exc_val).__name__}: {exc_val}"
            )
        self.manifest.write()
        return False

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def mark_stage_ok(self, stage: str, **metadata: Any) -> None:
        """Convenience: record a stage as completed successfully."""
        self.manifest.mark_stage(stage, "ok", **metadata)

    def mark_stage_failed(self, stage: str, reason: str, **metadata: Any) -> None:
        """Convenience: record a stage failure and append to error list."""
        self.manifest.mark_stage(stage, "failed", reason=reason, **metadata)
        self.manifest.add_error(f"{stage}: {reason}")

    def mark_stage_skipped(self, stage: str, reason: str = "", **metadata: Any) -> None:
        """Convenience: record a stage as intentionally skipped."""
        self.manifest.mark_stage(stage, "skipped", reason=reason, **metadata)

    def write_qa_matrix(self) -> Path:
        """Persist the QA matrix and embed its summary in the manifest.

        Returns:
            Path to the written QA matrix JSON file.
        """
        qa_path = self.qa_matrix.write(self.paths.qa_matrix_dir)
        summary = self.qa_matrix.summary()
        self.manifest.add_qa_summary(summary)
        return qa_path

    def __repr__(self) -> str:
        return (
            f"RunContext("
            f"run_id={self.run_id!r}, "
            f"asset_class={self.asset_class.value!r}, "
            f"run_mode={self.run_mode.value!r})"
        )
