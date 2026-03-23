"""Path helpers for SpectraQuant-AI-V3.

``ProjectPaths`` resolves all canonical directory locations from a single
repo-root anchor.  ``RunPaths`` extends it with per-run output paths.

Usage::

    pp = ProjectPaths(root="/repo")
    rp = RunPaths.from_project(pp, run_id="abc123", asset_class=AssetClass.CRYPTO)

    rp.cache_dir          # data/cache/crypto
    rp.reports_dir        # reports/crypto/abc123
    rp.manifest_path_prefix  # reports/crypto/abc123/run_manifest_crypto_…
    rp.qa_matrix_dir      # reports/crypto/abc123
"""

from __future__ import annotations

import os
from pathlib import Path

from spectraquant_v3.core.enums import AssetClass


class ProjectPaths:
    """Canonical directory layout for the whole repository.

    All paths are derived from *root* so the project can be relocated or
    run from a temporary directory without hard-coded absolute paths.

    Args:
        root: Repository root directory.  Defaults to the repo root discovered
            by walking up from this source file, then falls back to cwd.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        cache_root: str | Path | None = None,
        reports_root: str | Path | None = None,
    ) -> None:
        if root is not None:
            self.root = Path(root).resolve()
        else:
            self.root = self._discover_root()
        self._cache_root_override = self._resolve_optional_path(cache_root)
        self._reports_root_override = self._resolve_optional_path(reports_root)

    # ------------------------------------------------------------------
    # Root discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_root() -> Path:
        """Walk up from this file looking for a ``pyproject.toml`` anchor."""
        env_root = os.getenv("SPECTRAQUANT_V3_ROOT")
        if env_root:
            return Path(env_root).resolve()
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "pyproject.toml").exists():
                return parent
        return Path.cwd()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @property
    def config_v3_dir(self) -> Path:
        """``config/v3/`` directory containing YAML config files."""
        return self.root / "config" / "v3"

    # ------------------------------------------------------------------
    # Data / cache
    # ------------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        """``data/`` root directory for all cached data."""
        return self.root / "data"

    @property
    def cache_root(self) -> Path:
        """``data/cache/`` shared cache root."""
        if self._cache_root_override is not None:
            return self._cache_root_override
        return self.data_dir / "cache"

    @property
    def crypto_cache_dir(self) -> Path:
        """``data/cache/crypto/`` for crypto OHLCV parquet files."""
        return self.cache_root / "crypto"

    @property
    def equity_cache_dir(self) -> Path:
        """``data/cache/equities/`` for equity OHLCV parquet files."""
        return self.cache_root / "equities"

    def cache_dir_for(self, asset_class: AssetClass) -> Path:
        """Return the cache directory for *asset_class*."""
        if asset_class == AssetClass.CRYPTO:
            return self.crypto_cache_dir
        return self.equity_cache_dir

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    @property
    def reports_root(self) -> Path:
        """``reports/`` root for all pipeline outputs."""
        if self._reports_root_override is not None:
            return self._reports_root_override
        return self.root / "reports"

    def _resolve_optional_path(self, value: str | Path | None) -> Path | None:
        if value in (None, ""):
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate.resolve()

    @property
    def crypto_reports_dir(self) -> Path:
        """``reports/crypto/`` for crypto-pipeline outputs."""
        return self.reports_root / "crypto"

    @property
    def equity_reports_dir(self) -> Path:
        """``reports/equities/`` for equity-pipeline outputs."""
        return self.reports_root / "equities"

    def reports_dir_for(self, asset_class: AssetClass) -> Path:
        """Return the reports directory for *asset_class*."""
        if asset_class == AssetClass.CRYPTO:
            return self.crypto_reports_dir
        return self.equity_reports_dir

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def make_dirs(self) -> None:
        """Create all standard directories (idempotent)."""
        for path in (
            self.cache_root,
            self.crypto_cache_dir,
            self.equity_cache_dir,
            self.reports_root,
            self.crypto_reports_dir,
            self.equity_reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"ProjectPaths(root={self.root!r})"


class RunPaths:
    """Per-run output directory layout.

    Every run gets its own subdirectory so outputs from different runs
    do not overwrite each other.

    Args:
        project:     Parent :class:`ProjectPaths` instance.
        run_id:      Unique identifier for this run.
        asset_class: Which pipeline is running.
    """

    def __init__(
        self,
        project: ProjectPaths,
        run_id: str,
        asset_class: AssetClass,
    ) -> None:
        self.project = project
        self.run_id = run_id
        self.asset_class = asset_class

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_project(
        cls,
        project: ProjectPaths,
        run_id: str,
        asset_class: AssetClass,
    ) -> "RunPaths":
        """Create a :class:`RunPaths` from an existing :class:`ProjectPaths`."""
        return cls(project=project, run_id=run_id, asset_class=asset_class)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        """Per-run output directory: ``reports/<asset_class>/<run_id>/``."""
        return self.project.reports_dir_for(self.asset_class) / self.run_id

    @property
    def cache_dir(self) -> Path:
        """Shared cache directory for this asset class."""
        return self.project.cache_dir_for(self.asset_class)

    @property
    def manifest_dir(self) -> Path:
        """Directory where the run manifest JSON is written."""
        return self.run_dir

    @property
    def qa_matrix_dir(self) -> Path:
        """Directory where the QA matrix JSON is written."""
        return self.run_dir

    @property
    def stage_outputs_dir(self) -> Path:
        """Directory for intermediate stage output artefacts."""
        return self.run_dir / "stages"

    @property
    def feature_store_dir(self) -> Path:
        """Per-run feature store directory (parquet files keyed by stage)."""
        return self.run_dir / "features"

    @property
    def signals_dir(self) -> Path:
        """Directory for signal-agent output parquet files."""
        return self.run_dir / "signals"

    def make_dirs(self) -> None:
        """Create all per-run directories (idempotent)."""
        for path in (
            self.run_dir,
            self.cache_dir,
            self.stage_outputs_dir,
            self.feature_store_dir,
            self.signals_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"RunPaths(run_id={self.run_id!r}, "
            f"asset_class={self.asset_class.value!r}, "
            f"run_dir={self.run_dir!r})"
        )
