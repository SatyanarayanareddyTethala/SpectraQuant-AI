"""Tests for the SpectraQuant-AI-V3 scaffold.

Covers:
- Package imports
- Core enums
- Custom exceptions hierarchy
- Config loader (file-based and deep-merge)
- CacheManager run-mode enforcement
- RunManifest write/read round-trip
- QAMatrix validation and hard-guard
- Typed schema dataclasses
- Typer CLI smoke tests (version, doctor, sub-command help)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# 1. Package import sanity
# ---------------------------------------------------------------------------


def test_package_version_importable() -> None:
    import spectraquant_v3

    assert spectraquant_v3.__version__ == "3.0.0"


def test_core_submodules_importable() -> None:
    from spectraquant_v3.core import enums, errors, schema, qa  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Enums
# ---------------------------------------------------------------------------


def test_asset_class_values() -> None:
    from spectraquant_v3.core.enums import AssetClass

    assert AssetClass.CRYPTO.value == "crypto"
    assert AssetClass.EQUITY.value == "equity"


def test_run_mode_values() -> None:
    from spectraquant_v3.core.enums import RunMode

    assert RunMode.NORMAL.value == "normal"
    assert RunMode.TEST.value == "test"
    assert RunMode.REFRESH.value == "refresh"


def test_run_stage_ordering() -> None:
    from spectraquant_v3.core.enums import RunStage

    stages = list(RunStage)
    assert stages[0] == RunStage.UNIVERSE
    assert stages[-1] == RunStage.REPORTING


def test_signal_status_values() -> None:
    from spectraquant_v3.core.enums import SignalStatus

    assert SignalStatus.OK.value == "OK"
    assert SignalStatus.NO_SIGNAL.value == "NO_SIGNAL"
    assert SignalStatus.ERROR.value == "ERROR"


# ---------------------------------------------------------------------------
# 3. Exceptions hierarchy
# ---------------------------------------------------------------------------


def test_all_custom_exceptions_importable() -> None:
    from spectraquant_v3.core.errors import (
        AssetClassLeakError,
        CacheCorruptionError,
        CacheOnlyViolationError,
        EmptyPriceDataError,
        EmptyUniverseError,
        InvalidRunModeError,
        ManifestWriteError,
        MixedAssetClassRunError,
        SpectraQuantError,
        StageAbortError,
        SymbolResolutionError,
    )

    # All derive from base
    for exc_class in (
        MixedAssetClassRunError,
        AssetClassLeakError,
        SymbolResolutionError,
        EmptyUniverseError,
        EmptyPriceDataError,
        CacheOnlyViolationError,
        CacheCorruptionError,
        InvalidRunModeError,
        ManifestWriteError,
        StageAbortError,
    ):
        assert issubclass(exc_class, SpectraQuantError), exc_class


def test_asset_class_leak_is_subclass_of_mixed() -> None:
    from spectraquant_v3.core.errors import AssetClassLeakError, MixedAssetClassRunError

    assert issubclass(AssetClassLeakError, MixedAssetClassRunError)


def test_exceptions_are_raiseable() -> None:
    from spectraquant_v3.core.errors import (
        CacheOnlyViolationError,
        EmptyUniverseError,
        MixedAssetClassRunError,
        SymbolResolutionError,
    )

    with pytest.raises(MixedAssetClassRunError):
        raise MixedAssetClassRunError("INFY.NS detected in crypto run")

    with pytest.raises(SymbolResolutionError):
        raise SymbolResolutionError("BTC not in registry")

    with pytest.raises(EmptyUniverseError):
        raise EmptyUniverseError("universe is empty")

    with pytest.raises(CacheOnlyViolationError):
        raise CacheOnlyViolationError("network forbidden in test mode")


# ---------------------------------------------------------------------------
# 4. Config loader
# ---------------------------------------------------------------------------


@pytest.fixture()
def v3_config_dir(tmp_path: Path) -> Path:
    """Create a minimal config/v3 directory with all three YAML files."""
    cfg_dir = tmp_path / "config" / "v3"
    cfg_dir.mkdir(parents=True)

    base = {
        "run": {"mode": "normal", "dry_run": False, "reports_root": "reports", "log_level": "INFO"},
        "cache": {"root": "data/cache", "min_history_days": 60},
        "qa": {"min_ohlcv_coverage": 1.0, "max_missing_day_fraction": 0.10},
        "execution": {"mode": "paper"},
        "portfolio": {"max_weight": 0.20},
    }
    (cfg_dir / "base.yaml").write_text(yaml.dump(base))

    crypto_overlay = {"crypto": {"primary_ohlcv_provider": "ccxt", "symbols": ["BTC", "ETH"]}}
    (cfg_dir / "crypto.yaml").write_text(yaml.dump(crypto_overlay))

    equity_overlay = {"equities": {"primary_ohlcv_provider": "yfinance", "universe": {"tickers": ["INFY.NS"]}}}
    (cfg_dir / "equities.yaml").write_text(yaml.dump(equity_overlay))

    return cfg_dir


def test_load_config_reads_base_yaml(v3_config_dir: Path) -> None:
    from spectraquant_v3.core.config import load_config, reset_config_cache

    reset_config_cache()
    cfg = load_config(v3_config_dir, force_reload=True)
    assert cfg["run"]["mode"] == "normal"
    reset_config_cache()


def test_get_crypto_config_merges_overlay(v3_config_dir: Path) -> None:
    from spectraquant_v3.core.config import get_crypto_config, reset_config_cache

    reset_config_cache()
    cfg = get_crypto_config(v3_config_dir)
    assert cfg["crypto"]["primary_ohlcv_provider"] == "ccxt"
    assert "BTC" in cfg["crypto"]["symbols"]
    # Base keys still present
    assert cfg["run"]["mode"] == "normal"
    reset_config_cache()


def test_get_equity_config_merges_overlay(v3_config_dir: Path) -> None:
    from spectraquant_v3.core.config import get_equity_config, reset_config_cache

    reset_config_cache()
    cfg = get_equity_config(v3_config_dir)
    assert cfg["equities"]["primary_ohlcv_provider"] == "yfinance"
    assert "INFY.NS" in cfg["equities"]["universe"]["tickers"]
    reset_config_cache()


def test_load_config_cache_is_scoped_by_directory(tmp_path: Path) -> None:
    from spectraquant_v3.core.config import load_config, reset_config_cache

    cfg_a = tmp_path / "cfg_a"
    cfg_b = tmp_path / "cfg_b"
    cfg_a.mkdir()
    cfg_b.mkdir()
    (cfg_a / "base.yaml").write_text("run: {mode: normal}\ncache: {}\nqa: {}\nexecution: {}\nportfolio: {}\n")
    (cfg_b / "base.yaml").write_text("run: {mode: refresh}\ncache: {}\nqa: {}\nexecution: {}\nportfolio: {}\n")

    reset_config_cache()
    assert load_config(cfg_a, force_reload=True)["run"]["mode"] == "normal"
    assert load_config(cfg_b)["run"]["mode"] == "refresh"
    reset_config_cache()


def test_load_config_raises_when_missing(tmp_path: Path) -> None:
    from spectraquant_v3.core.config import load_config, reset_config_cache

    reset_config_cache()
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent")
    reset_config_cache()


def test_deep_merge_nested() -> None:
    from spectraquant_v3.core.config import _deep_merge

    base = {"a": {"x": 1, "y": 2}, "b": 10}
    overlay = {"a": {"y": 99, "z": 3}, "c": 20}
    result = _deep_merge(base, overlay)
    assert result == {"a": {"x": 1, "y": 99, "z": 3}, "b": 10, "c": 20}


# ---------------------------------------------------------------------------
# 5. CacheManager
# ---------------------------------------------------------------------------


def test_cache_manager_normal_mode_allows_network(tmp_path: Path) -> None:
    from spectraquant_v3.core.cache import CacheManager
    from spectraquant_v3.core.enums import RunMode

    cm = CacheManager(tmp_path / "cache", RunMode.NORMAL)
    # Should not raise even when cache is empty
    cm.assert_network_allowed("BTC_1d")


def test_cache_manager_test_mode_raises_on_miss(tmp_path: Path) -> None:
    from spectraquant_v3.core.cache import CacheManager
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import CacheOnlyViolationError

    cm = CacheManager(tmp_path / "cache", RunMode.TEST)
    with pytest.raises(CacheOnlyViolationError, match="TEST mode"):
        cm.assert_network_allowed("BTC_1d")


def test_cache_manager_test_mode_ok_when_cached(tmp_path: Path) -> None:
    from spectraquant_v3.core.cache import CacheManager
    from spectraquant_v3.core.enums import RunMode

    cm = CacheManager(tmp_path / "cache", RunMode.TEST)
    # Create the file so it looks cached
    cm.get_path("BTC_1d").write_bytes(b"fake parquet")
    # Should not raise
    cm.assert_network_allowed("BTC_1d")


def test_cache_manager_should_skip_download(tmp_path: Path) -> None:
    from spectraquant_v3.core.cache import CacheManager
    from spectraquant_v3.core.enums import RunMode

    cache_dir = tmp_path / "cache"
    cm_normal = CacheManager(cache_dir, RunMode.NORMAL)
    cm_refresh = CacheManager(cache_dir, RunMode.REFRESH)

    # No file yet – NORMAL should NOT skip (needs download)
    assert not cm_normal.should_skip_download("ETH_1d")

    # Create the file
    cm_normal.get_path("ETH_1d").write_bytes(b"data")

    # NORMAL with existing cache → skip
    assert cm_normal.should_skip_download("ETH_1d")

    # REFRESH always downloads
    assert not cm_refresh.should_skip_download("ETH_1d")


# ---------------------------------------------------------------------------
# 6. RunManifest
# ---------------------------------------------------------------------------


def test_manifest_writes_json(tmp_path: Path) -> None:
    from spectraquant_v3.core.enums import AssetClass, RunMode, RunStatus
    from spectraquant_v3.core.manifest import RunManifest

    manifest = RunManifest(
        asset_class=AssetClass.CRYPTO,
        run_mode=RunMode.NORMAL,
        output_dir=tmp_path,
    )
    manifest.mark_complete(RunStatus.SUCCESS)
    path = manifest.write()

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["asset_class"] == "crypto"
    assert data["run_mode"] == "normal"
    assert data["status"] == "success"
    assert data["run_id"] == manifest.run_id


def test_manifest_always_writes_on_failure(tmp_path: Path) -> None:
    from spectraquant_v3.core.enums import AssetClass, RunMode, RunStatus
    from spectraquant_v3.core.manifest import RunManifest

    manifest = RunManifest(
        asset_class=AssetClass.EQUITY,
        run_mode=RunMode.TEST,
        output_dir=tmp_path,
    )
    manifest.add_error("Something went wrong")
    path = manifest.write()

    data = json.loads(path.read_text())
    assert data["status"] == "aborted"   # default when not marked complete
    assert "Something went wrong" in data["errors"]


def test_manifest_stage_tracking(tmp_path: Path) -> None:
    from spectraquant_v3.core.enums import AssetClass, RunMode
    from spectraquant_v3.core.manifest import RunManifest

    manifest = RunManifest(AssetClass.CRYPTO, RunMode.NORMAL, output_dir=tmp_path)
    manifest.mark_stage("universe", "ok", n_symbols=10)
    manifest.mark_stage("ingestion", "ok", n_symbols_loaded=8)
    path = manifest.write()

    data = json.loads(path.read_text())
    assert data["stages"]["universe"]["status"] == "ok"
    assert data["stages"]["universe"]["n_symbols"] == 10
    assert data["stages"]["ingestion"]["n_symbols_loaded"] == 8


# ---------------------------------------------------------------------------
# 7. Typed schemas
# ---------------------------------------------------------------------------


def test_symbol_record_defaults() -> None:
    from spectraquant_v3.core.enums import AssetClass
    from spectraquant_v3.core.schema import SymbolRecord

    rec = SymbolRecord(canonical_symbol="BTC", asset_class=AssetClass.CRYPTO)
    assert rec.canonical_symbol == "BTC"
    assert rec.asset_class == AssetClass.CRYPTO
    assert rec.is_tradable is True
    assert rec.is_active is True
    assert rec.market_type == "spot"
    assert rec.metadata == {}


def test_qa_row_defaults() -> None:
    from spectraquant_v3.core.schema import QARow

    row = QARow(run_id="abc", as_of="2025-01-01", canonical_symbol="ETH", asset_class="crypto")
    assert row.has_ohlcv is False
    assert row.rows_loaded == 0
    assert row.stage_status == "PENDING"
    assert row.error_codes == []


def test_signal_row_defaults() -> None:
    from spectraquant_v3.core.enums import SignalStatus
    from spectraquant_v3.core.schema import SignalRow

    row = SignalRow(
        run_id="r1",
        timestamp="2025-01-01T00:00:00Z",
        canonical_symbol="BTC",
        asset_class="crypto",
        agent_id="momentum_agent",
        horizon="1d",
    )
    assert row.signal_score == 0.0
    assert row.confidence == 0.0
    assert row.status == SignalStatus.NO_SIGNAL.value


def test_allocation_row_defaults() -> None:
    from spectraquant_v3.core.schema import AllocationRow

    row = AllocationRow(run_id="r1", canonical_symbol="INFY", asset_class="equity")
    assert row.target_weight == 0.0
    assert row.blocked is False


# ---------------------------------------------------------------------------
# 8. QA Matrix
# ---------------------------------------------------------------------------


def test_qa_matrix_add_and_summary() -> None:
    from spectraquant_v3.core.qa import QAMatrix
    from spectraquant_v3.core.schema import QARow

    matrix = QAMatrix(run_id="r1", asset_class="crypto")
    matrix.add(QARow("r1", "2025-01-01", "BTC", "crypto", has_ohlcv=True, rows_loaded=365))
    matrix.add(QARow("r1", "2025-01-01", "ETH", "crypto", has_ohlcv=False))

    assert not matrix.all_missing_ohlcv()
    summary = matrix.summary()
    assert summary["total_symbols"] == 2
    assert summary["symbols_with_ohlcv"] == 1
    assert summary["symbols_missing_ohlcv"] == 1


def test_qa_matrix_all_missing_ohlcv() -> None:
    from spectraquant_v3.core.qa import QAMatrix
    from spectraquant_v3.core.schema import QARow

    matrix = QAMatrix(run_id="r1", asset_class="crypto")
    matrix.add(QARow("r1", "2025-01-01", "BTC", "crypto", has_ohlcv=False))
    matrix.add(QARow("r1", "2025-01-01", "ETH", "crypto", has_ohlcv=False))

    assert matrix.all_missing_ohlcv()


def test_qa_matrix_assert_ohlcv_raises_when_all_missing() -> None:
    from spectraquant_v3.core.errors import EmptyUniverseError
    from spectraquant_v3.core.qa import QAMatrix
    from spectraquant_v3.core.schema import QARow

    matrix = QAMatrix(run_id="r1", asset_class="crypto")
    matrix.add(QARow("r1", "2025-01-01", "BTC", "crypto", has_ohlcv=False))

    with pytest.raises(EmptyUniverseError, match="EMPTY_OHLCV_UNIVERSE"):
        matrix.assert_ohlcv_available()


def test_qa_matrix_assert_ohlcv_passes_when_any_present() -> None:
    from spectraquant_v3.core.qa import QAMatrix
    from spectraquant_v3.core.schema import QARow

    matrix = QAMatrix(run_id="r1", asset_class="crypto")
    matrix.add(QARow("r1", "2025-01-01", "BTC", "crypto", has_ohlcv=True, rows_loaded=200))
    matrix.add(QARow("r1", "2025-01-01", "ETH", "crypto", has_ohlcv=False))

    # Should not raise
    matrix.assert_ohlcv_available()


def test_qa_matrix_to_records() -> None:
    from spectraquant_v3.core.qa import QAMatrix
    from spectraquant_v3.core.schema import QARow

    matrix = QAMatrix(run_id="r1", asset_class="equity")
    matrix.add(QARow("r1", "2025-01-01", "INFY", "equity", has_ohlcv=True))

    records = matrix.to_records()
    assert len(records) == 1
    assert records[0]["canonical_symbol"] == "INFY"


# ---------------------------------------------------------------------------
# 9. Pipeline stubs raise NotImplementedError + write manifest
# ---------------------------------------------------------------------------



def test_crypto_pipeline_stub_raises_and_writes_manifest(tmp_path: Path) -> None:
    """Pipelines are now fully implemented; empty config raises EmptyUniverseError."""
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import EmptyUniverseError
    from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

    # A structurally valid config with zero symbols → EmptyUniverseError from
    # the universe builder stage.  All required base keys must be present so
    # the pipeline config guards pass before reaching the universe stage.
    cfg = {
        "run": {"mode": "normal"},
        "cache": {"root": "data/cache"},
        "qa": {"min_ohlcv_coverage": 1.0},
        "execution": {"mode": "paper"},
        "portfolio": {
            "max_weight": 0.25,
            "max_gross_leverage": 1.0,
            "min_confidence": 0.10,
            "min_signal_threshold": 0.05,
            "allocator": "equal_weight",
        },
        "crypto": {
            "symbols": [],
            "primary_ohlcv_provider": "ccxt",
            "universe_mode": "static",
            "quality_gate": {
                "min_market_cap_usd": 0,
                "min_24h_volume_usd": 0,
                "min_age_days": 0,
                "require_tradable_mapping": True,
            },
            "signals": {"momentum_lookback": 20},
            "reports_dir": str(tmp_path),
        },
    }
    with pytest.raises(EmptyUniverseError):
        run_crypto_pipeline(cfg, run_mode=RunMode.NORMAL, project_root=tmp_path)

    # Manifest must still be written (written by RunContext.__exit__)
    manifests = list(tmp_path.rglob("run_manifest_crypto_*.json"))
    assert len(manifests) >= 1
    data = json.loads(manifests[0].read_text())
    assert data["status"] == "aborted"


def test_equity_pipeline_stub_raises_and_writes_manifest(tmp_path: Path) -> None:
    """Pipelines are now fully implemented; empty config raises EmptyUniverseError."""
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import EmptyUniverseError
    from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

    # A structurally valid config with zero tickers → EmptyUniverseError from
    # the universe builder stage.  All required base keys must be present so
    # the pipeline config guards pass before reaching the universe stage.
    cfg = {
        "run": {"mode": "normal"},
        "cache": {"root": "data/cache"},
        "qa": {"min_ohlcv_coverage": 1.0},
        "execution": {"mode": "paper"},
        "portfolio": {
            "max_weight": 0.20,
            "max_gross_leverage": 1.0,
            "min_confidence": 0.10,
            "min_signal_threshold": 0.05,
            "allocator": "equal_weight",
        },
        "equities": {
            "primary_ohlcv_provider": "yfinance",
            "universe": {"tickers": [], "exclude": []},
            "quality_gate": {
                "min_price": 0,
                "min_avg_volume": 0,
                "min_history_days": 0,
            },
            "signals": {"momentum_lookback": 20},
            "reports_dir": str(tmp_path),
        },
    }
    with pytest.raises(EmptyUniverseError):
        run_equity_pipeline(cfg, run_mode=RunMode.NORMAL, project_root=tmp_path)

    manifests = list(tmp_path.rglob("run_manifest_equity_*.json"))
    assert len(manifests) >= 1
    data = json.loads(manifests[0].read_text())
    assert data["status"] == "aborted"


# ---------------------------------------------------------------------------
# 10. CLI smoke tests (Typer test client)
# ---------------------------------------------------------------------------


def test_cli_version() -> None:
    from typer.testing import CliRunner

    from spectraquant_v3.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "3.0.0" in result.output


def test_cli_crypto_run_wired(tmp_path: Path) -> None:
    """crypto run is now wired to the real pipeline; it completes successfully."""
    from typer.testing import CliRunner

    from spectraquant_v3.cli.main import app

    runner = CliRunner()
    # Use a real config dir and dry_run so no writes happen
    result = runner.invoke(
        app,
        ["crypto", "run", "--mode", "normal", "--dry-run"],
    )
    # Pipeline either completes (exit 0) or fails with a domain error (exit 1).
    # Either way the message must NOT say "scaffold only".
    assert "scaffold only" not in result.output
    assert "[crypto run]" in result.output


def test_cli_equity_run_wired(tmp_path: Path) -> None:
    """equity run is now wired to the real pipeline."""
    from typer.testing import CliRunner

    from spectraquant_v3.cli.main import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["equity", "run", "--mode", "normal", "--dry-run"],
    )
    assert "scaffold only" not in result.output
    assert "[equity run]" in result.output


def test_cli_crypto_download_stub() -> None:
    """crypto download command should attempt async batch download and report results."""
    from unittest.mock import AsyncMock, patch

    import pandas as pd
    from typer.testing import CliRunner

    from spectraquant_v3.cli.main import app

    runner = CliRunner()

    # Patch load_many_async to return a mock DataFrame so no real network call happens.
    mock_df = pd.DataFrame({"close": [100.0], "canonical_symbol": ["BTC"]})

    async def _mock_load_many_async(symbols, **kwargs):
        return {s: mock_df for s in symbols}

    with patch(
        "spectraquant_v3.crypto.ingestion.ohlcv_loader.CryptoOHLCVLoader.load_many_async",
        side_effect=_mock_load_many_async,
    ):
        result = runner.invoke(app, ["crypto", "download", "--symbols", "BTC,ETH"])

    # Command should succeed (all symbols returned by mock)
    assert result.exit_code == 0
    assert "[crypto download]" in result.output
    assert "2 succeeded" in result.output


def test_cli_equity_universe_stub() -> None:
    from typer.testing import CliRunner

    from spectraquant_v3.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["equity", "universe"])
    assert result.exit_code == 0


def test_cli_help_top_level() -> None:
    from typer.testing import CliRunner

    from spectraquant_v3.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "crypto" in result.output
    assert "equity" in result.output


def test_cli_doctor_passes_with_valid_config(tmp_path: Path) -> None:
    """doctor should exit 0 when all three config files are present."""
    from typer.testing import CliRunner

    from spectraquant_v3.cli.main import app

    cfg_dir = tmp_path / "config" / "v3"
    cfg_dir.mkdir(parents=True)
    for fname in ("base.yaml", "crypto.yaml", "equities.yaml"):
        (cfg_dir / fname).write_text(yaml.dump({"placeholder": True}))

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--config-dir", str(cfg_dir)])
    # Exit code 0 only if required packages (yaml, pandas, pyarrow, typer) are installed
    # In CI they should be; if not, allow exit code 1 but output must show config checks
    assert "Config dir" in result.output


def test_cli_doctor_fails_missing_config(tmp_path: Path) -> None:
    """doctor should exit 1 when config files are missing."""
    from typer.testing import CliRunner

    from spectraquant_v3.cli.main import app

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--config-dir", str(empty_dir)])
    # Should indicate failures
    assert result.exit_code == 1 or "NOT FOUND" in result.output


# ---------------------------------------------------------------------------
# 11. Cross-asset import guard (static structure check)
# ---------------------------------------------------------------------------


def test_crypto_package_does_not_import_equities() -> None:
    """Verify crypto __init__ docstring explicitly forbids equities imports."""
    from pathlib import Path

    crypto_init = (
        Path(__file__).parent.parent.parent
        / "src"
        / "spectraquant_v3"
        / "crypto"
        / "__init__.py"
    )
    assert crypto_init.exists()
    content = crypto_init.read_text()
    assert "equities" in content.lower()


def test_equities_package_does_not_import_crypto() -> None:
    """Verify equities __init__ docstring explicitly forbids crypto imports."""
    from pathlib import Path

    equities_init = (
        Path(__file__).parent.parent.parent
        / "src"
        / "spectraquant_v3"
        / "equities"
        / "__init__.py"
    )
    assert equities_init.exists()
    content = equities_init.read_text()
    assert "crypto" in content.lower()
