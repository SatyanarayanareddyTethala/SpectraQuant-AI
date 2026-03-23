"""Run manifest writer for SpectraQuant-AI-V3.

Every pipeline run – including aborted runs – must write a run manifest so
that failures are traceable without relying on transient log output.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectraquant_v3.core.enums import AssetClass, RunMode, RunStatus
from spectraquant_v3.core.errors import ManifestValidationError, ManifestWriteError


class RunManifest:
    """Captures metadata and status for a single pipeline run.

    Usage::

        manifest = RunManifest(asset_class=AssetClass.CRYPTO, run_mode=RunMode.NORMAL)
        try:
            # ... run pipeline stages ...
            manifest.mark_complete(RunStatus.SUCCESS)
        except Exception as exc:
            manifest.add_error(str(exc))
            raise
        finally:
            manifest.write()

    Args:
        asset_class: Which pipeline was executed.
        run_mode:    Cache mode for this run.
        run_id:      Unique identifier.  Auto-generated when *None*.
        output_dir:  Directory to write the JSON manifest file.
    """

    def __init__(
        self,
        asset_class: AssetClass,
        run_mode: RunMode,
        run_id: str | None = None,
        output_dir: str | Path = "reports",
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.asset_class = asset_class
        self.run_mode = run_mode
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: str | None = None
        self.status = RunStatus.ABORTED
        self.stages: dict[str, Any] = {}
        self.errors: list[str] = []
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def mark_stage(self, stage: str, status: str, **metadata: Any) -> None:
        """Record completion status for a named pipeline stage.

        Args:
            stage:    Stage identifier (use :class:`~spectraquant_v3.core.enums.RunStage`
                      values).
            status:   Outcome string, e.g. ``'ok'``, ``'skipped'``, ``'failed'``.
            metadata: Optional extra key/value pairs stored alongside status.
        """
        self.stages[stage] = {"status": status, **metadata}

    def mark_complete(self, status: RunStatus = RunStatus.SUCCESS) -> None:
        """Set the run to a terminal successful (or partial) state."""
        self.status = status
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def add_error(self, error: str) -> None:
        """Append an error message to the error list."""
        self.errors.append(error)

    def add_qa_summary(self, summary: dict[str, Any]) -> None:
        """Embed a QA matrix summary into the manifest.

        Call this after the QA matrix has been finalised so the manifest
        contains a human-readable overview of data availability.

        Args:
            summary: Dictionary returned by :meth:`~spectraquant_v3.core.qa.QAMatrix.summary`.
        """
        self.stages["_qa_summary"] = summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write(self) -> Path:
        """Persist manifest as JSON.  Always call this, even on failure.

        Returns:
            Path of the written manifest file.

        Raises:
            ManifestWriteError: If the directory cannot be created or file cannot be written.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = (
                f"run_manifest_{self.asset_class.value}_{ts}_{self.run_id}.json"
            )
            path = self.output_dir / filename
            payload: dict[str, Any] = {
                "run_id": self.run_id,
                "asset_class": self.asset_class.value,
                "run_mode": self.run_mode.value,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "status": self.status.value,
                "stages": self.stages,
                "errors": self.errors,
            }
            path.write_text(json.dumps(payload, indent=2))
            return path
        except OSError as exc:
            raise ManifestWriteError(
                f"Failed to write manifest for run '{self.run_id}' to {self.output_dir}: {exc}"
            ) from exc

    @classmethod
    def from_file(cls, path: str | Path) -> "RunManifest":
        """Load a previously-written manifest from a JSON file.

        This is useful for post-run inspection, debugging, and test assertions.

        Args:
            path: Path to the JSON manifest file.

        Returns:
            A :class:`RunManifest` populated from the file contents.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file cannot be parsed as a valid manifest.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")

        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"Cannot parse manifest JSON at {path}: {exc}"
            ) from exc

        try:
            obj = cls(
                asset_class=AssetClass(data["asset_class"]),
                run_mode=RunMode(data["run_mode"]),
                run_id=data["run_id"],
                output_dir=path.parent,
            )
            obj.started_at = data.get("started_at", obj.started_at)
            obj.completed_at = data.get("completed_at")
            obj.status = RunStatus(data.get("status", RunStatus.ABORTED.value))
            obj.stages = data.get("stages", {})
            obj.errors = data.get("errors", [])
        except (KeyError, ValueError) as exc:
            raise ManifestValidationError(
                f"Manifest at {path} is missing required fields: {exc}"
            ) from exc

        return obj
