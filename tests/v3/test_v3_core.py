"""Tests for the SpectraQuant-AI-V3 shared core framework.

Covers the fully-implemented components:
- paths.py         (ProjectPaths, RunPaths)
- context.py       (RunContext — create, context-manager, helpers)
- errors.py        (ConfigValidationError, DataSchemaError additions)
- config.py        (validate_config, get_run_mode_from_config)
- cache.py         (read_parquet, write_parquet, list_keys, corruption)
- manifest.py      (add_qa_summary, from_file)
- qa.py            (write, get_row, mark_failed, summary with failed count)
- schema.py        (validate_ohlcv_dataframe)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


# ==========================================================================
# 1. Path helpers
# ==========================================================================


class TestProjectPaths:
    def test_init_with_explicit_root(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.paths import ProjectPaths

        pp = ProjectPaths(root=tmp_path)
        assert pp.root == tmp_path

    def test_standard_paths_derived_from_root(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.paths import ProjectPaths

        pp = ProjectPaths(root=tmp_path)
        assert pp.cache_root == tmp_path / "data" / "cache"
        assert pp.crypto_cache_dir == tmp_path / "data" / "cache" / "crypto"
        assert pp.equity_cache_dir == tmp_path / "data" / "cache" / "equities"
        assert pp.reports_root == tmp_path / "reports"
        assert pp.crypto_reports_dir == tmp_path / "reports" / "crypto"
        assert pp.equity_reports_dir == tmp_path / "reports" / "equities"
        assert pp.config_v3_dir == tmp_path / "config" / "v3"

    def test_cache_dir_for_crypto(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.paths import ProjectPaths

        pp = ProjectPaths(root=tmp_path)
        assert pp.cache_dir_for(AssetClass.CRYPTO) == pp.crypto_cache_dir

    def test_cache_dir_for_equity(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.paths import ProjectPaths

        pp = ProjectPaths(root=tmp_path)
        assert pp.cache_dir_for(AssetClass.EQUITY) == pp.equity_cache_dir

    def test_reports_dir_for_equity(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.paths import ProjectPaths

        pp = ProjectPaths(root=tmp_path)
        assert pp.reports_dir_for(AssetClass.EQUITY) == pp.equity_reports_dir

    def test_make_dirs_creates_directories(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.paths import ProjectPaths

        pp = ProjectPaths(root=tmp_path)
        pp.make_dirs()
        assert pp.crypto_cache_dir.is_dir()
        assert pp.equity_cache_dir.is_dir()
        assert pp.crypto_reports_dir.is_dir()
        assert pp.equity_reports_dir.is_dir()

    def test_make_dirs_idempotent(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.paths import ProjectPaths

        pp = ProjectPaths(root=tmp_path)
        pp.make_dirs()
        pp.make_dirs()  # should not raise

    def test_discover_root_finds_pyproject(self) -> None:
        from spectraquant_v3.core.paths import ProjectPaths

        # The actual repo has pyproject.toml — discovery should find it
        pp = ProjectPaths()
        assert (pp.root / "pyproject.toml").exists()

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from spectraquant_v3.core.paths import ProjectPaths

        monkeypatch.setenv("SPECTRAQUANT_V3_ROOT", str(tmp_path))
        pp = ProjectPaths()
        assert pp.root == tmp_path

    def test_repr(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.paths import ProjectPaths

        pp = ProjectPaths(root=tmp_path)
        assert "ProjectPaths" in repr(pp)
        assert str(tmp_path) in repr(pp)


class TestRunPaths:
    def test_run_dir_structure(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.paths import ProjectPaths, RunPaths

        pp = ProjectPaths(root=tmp_path)
        rp = RunPaths.from_project(pp, run_id="abc123", asset_class=AssetClass.CRYPTO)

        assert rp.run_dir == tmp_path / "reports" / "crypto" / "abc123"
        assert rp.cache_dir == pp.crypto_cache_dir
        assert rp.signals_dir == rp.run_dir / "signals"
        assert rp.feature_store_dir == rp.run_dir / "features"
        assert rp.stage_outputs_dir == rp.run_dir / "stages"

    def test_run_dir_equity(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.paths import ProjectPaths, RunPaths

        pp = ProjectPaths(root=tmp_path)
        rp = RunPaths.from_project(pp, run_id="xyz", asset_class=AssetClass.EQUITY)
        assert rp.run_dir == tmp_path / "reports" / "equities" / "xyz"

    def test_make_dirs_creates_all(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.paths import ProjectPaths, RunPaths

        pp = ProjectPaths(root=tmp_path)
        rp = RunPaths.from_project(pp, run_id="r1", asset_class=AssetClass.CRYPTO)
        rp.make_dirs()

        assert rp.run_dir.is_dir()
        assert rp.cache_dir.is_dir()
        assert rp.stage_outputs_dir.is_dir()
        assert rp.signals_dir.is_dir()

    def test_manifest_dir_equals_run_dir(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.paths import ProjectPaths, RunPaths

        pp = ProjectPaths(root=tmp_path)
        rp = RunPaths.from_project(pp, run_id="r1", asset_class=AssetClass.EQUITY)
        assert rp.manifest_dir == rp.run_dir

    def test_repr(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass
        from spectraquant_v3.core.paths import ProjectPaths, RunPaths

        pp = ProjectPaths(root=tmp_path)
        rp = RunPaths.from_project(pp, "run42", AssetClass.CRYPTO)
        assert "RunPaths" in repr(rp)
        assert "run42" in repr(rp)


# ==========================================================================
# 2. RunContext
# ==========================================================================


class TestRunContext:
    def test_create_crypto(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode

        ctx = RunContext.create(
            asset_class=AssetClass.CRYPTO,
            run_mode=RunMode.NORMAL,
            config={},
            run_id="test01",
            project_root=tmp_path,
        )
        assert ctx.run_id == "test01"
        assert ctx.asset_class == AssetClass.CRYPTO
        assert ctx.run_mode == RunMode.NORMAL
        assert ctx.cache is not None
        assert ctx.manifest is not None
        assert ctx.qa_matrix is not None
        assert ctx.paths is not None

    def test_create_equity(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode

        ctx = RunContext.create(
            asset_class=AssetClass.EQUITY,
            run_mode=RunMode.TEST,
            config={},
            project_root=tmp_path,
        )
        assert ctx.asset_class == AssetClass.EQUITY
        assert ctx.run_mode == RunMode.TEST

    def test_auto_run_id_generated(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode

        ctx = RunContext.create(AssetClass.CRYPTO, RunMode.NORMAL, project_root=tmp_path)
        assert ctx.run_id
        assert len(ctx.run_id) == 8

    def test_as_of_is_iso_utc(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode

        ctx = RunContext.create(AssetClass.CRYPTO, RunMode.NORMAL, project_root=tmp_path)
        assert "T" in ctx.as_of
        assert "+00:00" in ctx.as_of or "Z" in ctx.as_of

    def test_context_manager_success_writes_manifest(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode, RunStatus

        with RunContext.create(
            AssetClass.CRYPTO, RunMode.NORMAL,
            config={},
            run_id="cm01",
            project_root=tmp_path,
        ) as ctx:
            ctx.mark_stage_ok("universe", n_symbols=5)

        # Manifest lives in reports/crypto/<run_id>/
        manifest_dir = tmp_path / "reports" / "crypto" / "cm01"
        manifests = list(manifest_dir.glob("run_manifest_*.json"))
        assert len(manifests) == 1
        data = json.loads(manifests[0].read_text())
        assert data["status"] == RunStatus.SUCCESS.value

    def test_context_manager_exception_writes_aborted_manifest(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode, RunStatus

        with pytest.raises(RuntimeError, match="deliberate"):
            with RunContext.create(
                AssetClass.EQUITY, RunMode.NORMAL,
                config={},
                run_id="cm02",
                project_root=tmp_path,
            ):
                raise RuntimeError("deliberate test failure")

        # Manifest lives in reports/equities/<run_id>/
        manifest_dir = tmp_path / "reports" / "equities" / "cm02"
        manifests = list(manifest_dir.glob("run_manifest_*.json"))
        assert len(manifests) == 1
        data = json.loads(manifests[0].read_text())
        assert data["status"] == RunStatus.ABORTED.value
        assert any("deliberate test failure" in e for e in data["errors"])

    def test_mark_stage_ok(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode

        ctx = RunContext.create(AssetClass.CRYPTO, RunMode.NORMAL, project_root=tmp_path)
        ctx.mark_stage_ok("universe", n_symbols=10)
        assert ctx.manifest.stages["universe"]["status"] == "ok"
        assert ctx.manifest.stages["universe"]["n_symbols"] == 10

    def test_mark_stage_failed(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode

        ctx = RunContext.create(AssetClass.CRYPTO, RunMode.NORMAL, project_root=tmp_path)
        ctx.mark_stage_failed("ingestion", "NO_DATA")
        assert ctx.manifest.stages["ingestion"]["status"] == "failed"
        assert any("ingestion" in e for e in ctx.manifest.errors)

    def test_mark_stage_skipped(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode

        ctx = RunContext.create(AssetClass.CRYPTO, RunMode.NORMAL, project_root=tmp_path)
        ctx.mark_stage_skipped("onchain", reason="disabled in config")
        assert ctx.manifest.stages["onchain"]["status"] == "skipped"

    def test_write_qa_matrix_embeds_summary(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.schema import QARow

        ctx = RunContext.create(
            AssetClass.CRYPTO, RunMode.NORMAL, project_root=tmp_path, run_id="qa01"
        )
        ctx.qa_matrix.add(
            QARow("qa01", "2025-01-01", "BTC", "crypto", has_ohlcv=True, rows_loaded=100)
        )
        qa_path = ctx.write_qa_matrix()

        assert qa_path.exists()
        assert "_qa_summary" in ctx.manifest.stages

    def test_repr(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode

        ctx = RunContext.create(
            AssetClass.EQUITY, RunMode.TEST, run_id="repr01", project_root=tmp_path
        )
        r = repr(ctx)
        assert "RunContext" in r
        assert "repr01" in r
        assert "equity" in r

    def test_cache_wired_to_asset_class_dir(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.context import RunContext
        from spectraquant_v3.core.enums import AssetClass, RunMode

        ctx = RunContext.create(
            AssetClass.CRYPTO, RunMode.NORMAL, project_root=tmp_path, run_id="c1"
        )
        assert "crypto" in str(ctx.cache.cache_dir)


# ==========================================================================
# 3. New error types
# ==========================================================================


class TestNewErrors:
    def test_config_validation_error_importable(self) -> None:
        from spectraquant_v3.core.errors import ConfigValidationError, SpectraQuantError

        assert issubclass(ConfigValidationError, SpectraQuantError)

    def test_data_schema_error_importable(self) -> None:
        from spectraquant_v3.core.errors import DataSchemaError, SpectraQuantError

        assert issubclass(DataSchemaError, SpectraQuantError)

    def test_errors_are_raiseable(self) -> None:
        from spectraquant_v3.core.errors import ConfigValidationError, DataSchemaError

        with pytest.raises(ConfigValidationError):
            raise ConfigValidationError("missing key 'run'")

        with pytest.raises(DataSchemaError):
            raise DataSchemaError("missing column 'close'")


# ==========================================================================
# 4. Config enhancements
# ==========================================================================


@pytest.fixture()
def full_config() -> dict:
    return {
        "run": {"mode": "normal", "dry_run": False},
        "cache": {"root": "data/cache", "min_history_days": 60},
        "qa": {"min_ohlcv_coverage": 1.0},
        "execution": {"mode": "paper"},
        "portfolio": {"max_weight": 0.20},
    }


class TestConfigValidation:
    def test_validate_config_passes_with_all_keys(self, full_config: dict) -> None:
        from spectraquant_v3.core.config import validate_config

        validate_config(full_config)  # should not raise

    def test_validate_config_raises_on_missing_key(self, full_config: dict) -> None:
        from spectraquant_v3.core.config import validate_config
        from spectraquant_v3.core.errors import ConfigValidationError

        del full_config["run"]
        with pytest.raises(ConfigValidationError, match="run"):
            validate_config(full_config)

    def test_validate_config_raises_on_wrong_type(self, full_config: dict) -> None:
        from spectraquant_v3.core.config import validate_config
        from spectraquant_v3.core.errors import ConfigValidationError

        full_config["run"] = "not-a-dict"
        with pytest.raises(ConfigValidationError, match="mapping"):
            validate_config(full_config)

    def test_validate_multiple_missing_keys(self) -> None:
        from spectraquant_v3.core.config import validate_config
        from spectraquant_v3.core.errors import ConfigValidationError

        with pytest.raises(ConfigValidationError):
            validate_config({})


class TestEquityTickerResolution:
    def test_resolve_equity_tickers_from_explicit_list(self) -> None:
        from spectraquant_v3.equities.symbols.registry import resolve_equity_tickers_from_config

        cfg = {"equities": {"universe": {"tickers": ["infy", "TCS.NS", " infy "]}}}
        assert resolve_equity_tickers_from_config(cfg) == ["INFY.NS", "TCS.NS"]

    def test_resolve_equity_tickers_from_nse_file(self, tmp_path: Path) -> None:
        from spectraquant_v3.equities.symbols.registry import resolve_equity_tickers_from_config

        universe_file = tmp_path / "nse.csv"
        universe_file.write_text("SYMBOL\nINFY\nTCS\n")
        cfg = {"equities": {"universe": {"tickers_file": str(universe_file)}}}

        assert resolve_equity_tickers_from_config(cfg) == ["INFY.NS", "TCS.NS"]


class TestGetRunModeFromConfig:
    def test_reads_from_config(self, full_config: dict) -> None:
        from spectraquant_v3.core.config import get_run_mode_from_config
        from spectraquant_v3.core.enums import RunMode

        full_config["run"]["mode"] = "refresh"
        assert get_run_mode_from_config(full_config) == RunMode.REFRESH

    def test_falls_back_to_env(
        self, full_config: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from spectraquant_v3.core.config import get_run_mode_from_config
        from spectraquant_v3.core.enums import RunMode

        del full_config["run"]["mode"]
        monkeypatch.setenv("SPECTRAQUANT_RUN_MODE", "test")
        assert get_run_mode_from_config(full_config) == RunMode.TEST

    def test_falls_back_to_normal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from spectraquant_v3.core.config import get_run_mode_from_config
        from spectraquant_v3.core.enums import RunMode

        monkeypatch.delenv("SPECTRAQUANT_RUN_MODE", raising=False)
        assert get_run_mode_from_config({}) == RunMode.NORMAL

    def test_raises_on_invalid_mode(self, full_config: dict) -> None:
        from spectraquant_v3.core.config import get_run_mode_from_config
        from spectraquant_v3.core.errors import InvalidRunModeError

        full_config["run"]["mode"] = "invalid_mode"
        with pytest.raises(InvalidRunModeError):
            get_run_mode_from_config(full_config)


# ==========================================================================
# 5. CacheManager enhancements
# ==========================================================================


class TestCacheManagerParquetIO:
    def _make_df(self) -> pd.DataFrame:
        import numpy as np

        idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "open": np.ones(10),
                "high": np.ones(10) * 1.1,
                "low": np.ones(10) * 0.9,
                "close": np.ones(10),
                "volume": np.ones(10) * 1000,
            },
            index=idx,
        )

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        df = self._make_df()
        path = cm.write_parquet("BTC", df)

        assert path.exists()
        loaded = cm.read_parquet("BTC")
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)

    def test_write_empty_df_raises(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        with pytest.raises(ValueError, match="empty DataFrame"):
            cm.write_parquet("BTC", pd.DataFrame())

    def test_read_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        with pytest.raises(FileNotFoundError):
            cm.read_parquet("NONEXISTENT")

    def test_read_corrupt_file_raises_cache_corruption(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.core.errors import CacheCorruptionError

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        cm.get_path("ETH").write_bytes(b"not a valid parquet file")
        with pytest.raises(CacheCorruptionError, match="Failed to read"):
            cm.read_parquet("ETH")

    def test_read_test_mode_miss_raises_cache_only_violation(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode
        from spectraquant_v3.core.errors import CacheOnlyViolationError

        cm = CacheManager(tmp_path, RunMode.TEST)
        with pytest.raises(CacheOnlyViolationError):
            cm.read_parquet("MISSING")

    def test_list_keys_empty(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        assert cm.list_keys() == []

    def test_list_keys_after_writes(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        df = self._make_df()
        cm.write_parquet("BTC", df)
        cm.write_parquet("ETH", df)

        keys = cm.list_keys()
        assert "BTC" in keys
        assert "ETH" in keys
        assert len(keys) == 2

    def test_slash_separator_in_key(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        df = self._make_df()
        path = cm.write_parquet("BTC/USDT", df)

        # File should be named BTC__USDT.parquet
        assert path.name == "BTC__USDT.parquet"
        # list_keys should restore the slash
        assert "BTC/USDT" in cm.list_keys()
        # read_parquet should also work with the slash key
        loaded = cm.read_parquet("BTC/USDT")
        assert len(loaded) == 10

    def test_validate_parquet_true_for_valid(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        df = self._make_df()
        cm.write_parquet("SOL", df)
        assert cm.validate_parquet("SOL") is True

    def test_validate_parquet_false_for_missing(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        assert cm.validate_parquet("MISSING") is False

    def test_validate_parquet_false_for_corrupt(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        cm.get_path("BAD").write_bytes(b"garbage")
        assert cm.validate_parquet("BAD") is False

    def test_atomic_write_cleans_up_tmp_on_error(self, tmp_path: Path) -> None:
        """tmp file should not linger if write_parquet succeeds."""
        from spectraquant_v3.core.cache import CacheManager
        from spectraquant_v3.core.enums import RunMode

        cm = CacheManager(tmp_path, RunMode.NORMAL)
        df = self._make_df()
        cm.write_parquet("ADA", df)

        # .tmp file must be gone
        tmp_file = cm.get_path("ADA").with_suffix(".tmp")
        assert not tmp_file.exists()


# ==========================================================================
# 6. Manifest enhancements
# ==========================================================================


class TestManifestEnhancements:
    def test_add_qa_summary_stored_in_stages(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.manifest import RunManifest

        m = RunManifest(AssetClass.CRYPTO, RunMode.NORMAL, output_dir=tmp_path)
        summary = {"total_symbols": 5, "symbols_with_ohlcv": 4}
        m.add_qa_summary(summary)

        assert m.stages["_qa_summary"] == summary

    def test_add_qa_summary_appears_in_json(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass, RunMode
        from spectraquant_v3.core.manifest import RunManifest

        m = RunManifest(AssetClass.CRYPTO, RunMode.NORMAL, output_dir=tmp_path)
        m.add_qa_summary({"total_symbols": 3})
        path = m.write()
        data = json.loads(path.read_text())
        assert data["stages"]["_qa_summary"]["total_symbols"] == 3

    def test_from_file_roundtrip(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.enums import AssetClass, RunMode, RunStatus
        from spectraquant_v3.core.manifest import RunManifest

        m = RunManifest(
            AssetClass.EQUITY, RunMode.TEST, run_id="ff01", output_dir=tmp_path
        )
        m.mark_stage("universe", "ok", n=10)
        m.mark_complete(RunStatus.SUCCESS)
        path = m.write()

        loaded = RunManifest.from_file(path)
        assert loaded.run_id == "ff01"
        assert loaded.asset_class == AssetClass.EQUITY
        assert loaded.run_mode == RunMode.TEST
        assert loaded.status == RunStatus.SUCCESS
        assert loaded.stages["universe"]["status"] == "ok"

    def test_from_file_raises_on_missing(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.manifest import RunManifest

        with pytest.raises(FileNotFoundError):
            RunManifest.from_file(tmp_path / "nonexistent.json")

    def test_from_file_raises_on_corrupt_json(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.manifest import RunManifest
        from spectraquant_v3.core.errors import ManifestValidationError

        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(ManifestValidationError, match="Cannot parse"):
            RunManifest.from_file(bad)

    def test_from_file_raises_on_missing_fields(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.manifest import RunManifest
        from spectraquant_v3.core.errors import ManifestValidationError

        incomplete = tmp_path / "incomplete.json"
        incomplete.write_text('{"run_id": "x"}')
        with pytest.raises(ManifestValidationError, match="missing required fields"):
            RunManifest.from_file(incomplete)


# ==========================================================================
# 7. QAMatrix enhancements
# ==========================================================================


class TestQAMatrixEnhancements:
    def _make_row(self, symbol: str, has_ohlcv: bool = True) -> object:
        from spectraquant_v3.core.schema import QARow

        return QARow("r1", "2025-01-01", symbol, "crypto", has_ohlcv=has_ohlcv)

    def test_get_row_returns_existing(self) -> None:
        from spectraquant_v3.core.qa import QAMatrix

        m = QAMatrix("r1", "crypto")
        m.add(self._make_row("BTC"))
        row = m.get_row("BTC")
        assert row is not None
        assert row.canonical_symbol == "BTC"

    def test_get_row_returns_none_for_missing(self) -> None:
        from spectraquant_v3.core.qa import QAMatrix

        m = QAMatrix("r1", "crypto")
        assert m.get_row("NONEXISTENT") is None

    def test_add_replaces_existing_row(self) -> None:
        from spectraquant_v3.core.qa import QAMatrix
        from spectraquant_v3.core.schema import QARow

        m = QAMatrix("r1", "crypto")
        m.add(QARow("r1", "2025-01-01", "BTC", "crypto", has_ohlcv=False))
        m.add(QARow("r1", "2025-01-01", "BTC", "crypto", has_ohlcv=True, rows_loaded=100))

        # Only one row should exist
        assert len(m.rows) == 1
        assert m.get_row("BTC").has_ohlcv is True
        assert m.get_row("BTC").rows_loaded == 100

    def test_mark_failed_updates_existing_row(self) -> None:
        from spectraquant_v3.core.qa import QAMatrix

        m = QAMatrix("r1", "crypto")
        m.add(self._make_row("ETH"))
        m.mark_failed("ETH", "NO_OHLCV", note="provider timeout")

        row = m.get_row("ETH")
        assert row.stage_status == "FAILED"
        assert "NO_OHLCV" in row.error_codes
        assert "provider timeout" in row.notes

    def test_mark_failed_creates_row_if_absent(self) -> None:
        from spectraquant_v3.core.qa import QAMatrix

        m = QAMatrix("r1", "crypto")
        m.mark_failed("SOL", "DOWNLOAD_ERROR")

        row = m.get_row("SOL")
        assert row is not None
        assert row.stage_status == "FAILED"
        assert "DOWNLOAD_ERROR" in row.error_codes

    def test_mark_failed_deduplicates_error_codes(self) -> None:
        from spectraquant_v3.core.qa import QAMatrix

        m = QAMatrix("r1", "crypto")
        m.mark_failed("BTC", "NO_OHLCV")
        m.mark_failed("BTC", "NO_OHLCV")  # duplicate

        row = m.get_row("BTC")
        assert row.error_codes.count("NO_OHLCV") == 1

    def test_summary_includes_failed_count(self) -> None:
        from spectraquant_v3.core.qa import QAMatrix

        m = QAMatrix("r1", "crypto")
        m.add(self._make_row("BTC", has_ohlcv=True))
        m.mark_failed("ETH", "NO_OHLCV")

        summary = m.summary()
        assert summary["symbols_failed"] == 1
        assert summary["total_symbols"] == 2

    def test_write_creates_json(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.qa import QAMatrix

        m = QAMatrix("r1", "crypto")
        m.add(self._make_row("BTC", has_ohlcv=True))
        path = m.write(tmp_path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["run_id"] == "r1"
        assert data["asset_class"] == "crypto"
        assert len(data["rows"]) == 1
        assert "summary" in data

    def test_write_includes_summary(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.qa import QAMatrix

        m = QAMatrix("r1", "equity")
        m.add(self._make_row("INFY", has_ohlcv=True))
        m.add(self._make_row("TCS", has_ohlcv=False))
        path = m.write(tmp_path)

        data = json.loads(path.read_text())
        assert data["summary"]["total_symbols"] == 2
        assert data["summary"]["symbols_with_ohlcv"] == 1

    def test_write_creates_output_dir(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.qa import QAMatrix

        m = QAMatrix("r1", "crypto")
        m.add(self._make_row("BTC"))
        nested = tmp_path / "deep" / "nested"
        path = m.write(nested)
        assert path.exists()


# ==========================================================================
# 8. validate_ohlcv_dataframe
# ==========================================================================


class TestValidateOhlcvDataFrame:
    def _valid_df(self, rows: int = 10) -> pd.DataFrame:
        import numpy as np

        return pd.DataFrame(
            {
                "open": np.ones(rows),
                "high": np.ones(rows),
                "low": np.ones(rows),
                "close": np.ones(rows),
                "volume": np.ones(rows),
            }
        )

    def test_valid_df_passes(self) -> None:
        from spectraquant_v3.core.schema import validate_ohlcv_dataframe

        validate_ohlcv_dataframe(self._valid_df())  # should not raise

    def test_empty_df_raises_empty_price_data(self) -> None:
        from spectraquant_v3.core.errors import EmptyPriceDataError
        from spectraquant_v3.core.schema import validate_ohlcv_dataframe

        with pytest.raises(EmptyPriceDataError, match="empty"):
            validate_ohlcv_dataframe(pd.DataFrame())

    def test_non_dataframe_raises_data_schema_error(self) -> None:
        from spectraquant_v3.core.errors import DataSchemaError
        from spectraquant_v3.core.schema import validate_ohlcv_dataframe

        with pytest.raises(DataSchemaError, match="DataFrame"):
            validate_ohlcv_dataframe({"close": [1, 2, 3]})

    def test_missing_column_raises_data_schema_error(self) -> None:
        from spectraquant_v3.core.errors import DataSchemaError
        from spectraquant_v3.core.schema import validate_ohlcv_dataframe

        df = self._valid_df().drop(columns=["volume"])
        with pytest.raises(DataSchemaError, match="volume"):
            validate_ohlcv_dataframe(df)

    def test_case_insensitive_column_matching(self) -> None:
        from spectraquant_v3.core.schema import validate_ohlcv_dataframe

        df = self._valid_df()
        df.columns = [c.upper() for c in df.columns]
        validate_ohlcv_dataframe(df)  # should not raise

    def test_min_rows_enforced(self) -> None:
        from spectraquant_v3.core.errors import EmptyPriceDataError
        from spectraquant_v3.core.schema import validate_ohlcv_dataframe

        with pytest.raises(EmptyPriceDataError, match="60 are required"):
            validate_ohlcv_dataframe(self._valid_df(rows=10), min_rows=60)

    def test_symbol_appears_in_error_message(self) -> None:
        from spectraquant_v3.core.errors import EmptyPriceDataError
        from spectraquant_v3.core.schema import validate_ohlcv_dataframe

        with pytest.raises(EmptyPriceDataError, match="BTC"):
            validate_ohlcv_dataframe(pd.DataFrame(), symbol="BTC")

    def test_multiple_missing_columns_listed(self) -> None:
        from spectraquant_v3.core.errors import DataSchemaError
        from spectraquant_v3.core.schema import validate_ohlcv_dataframe

        df = pd.DataFrame({"price": [1, 2, 3]})
        with pytest.raises(DataSchemaError) as exc_info:
            validate_ohlcv_dataframe(df)
        msg = str(exc_info.value)
        # At least one required column name should appear in the message
        assert any(col in msg for col in ("open", "high", "low", "close", "volume"))
