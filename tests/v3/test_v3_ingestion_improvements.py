"""Tests for the new SpectraQuant-AI-V3 features:
- CacheManager freshness metadata (sidecar JSON, is_stale)
- Missing-bar diagnostics (MissingBarReport, diagnose_missing_bars)
- Symbol-level audit logging (IngestionAuditLog, IngestionAuditEntry)
- Config v3 presence (providers.yaml, news.yaml, strategies.yaml, risk.yaml)
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


# ===========================================================================
# Helpers
# ===========================================================================


def _ohlcv_df(n: int = 100, canonical_symbol: str = "BTC") -> pd.DataFrame:
    """Generate a synthetic daily OHLCV DataFrame."""
    import numpy as np

    rng = np.random.default_rng(42)
    close = 30_000.0 + np.cumsum(rng.standard_normal(n) * 100)
    close = np.maximum(close, 1.0)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": close * 1.001,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1_000, 50_000, n),
        },
        index=idx,
    )
    df["canonical_symbol"] = canonical_symbol
    df["provider"] = "test"
    df["exchange_id"] = "test"
    df["timeframe"] = "1d"
    df["ingested_at"] = datetime.datetime.now(tz=datetime.timezone.utc)
    df["timestamp"] = df.index
    return df


# ===========================================================================
# 1. CacheManager freshness metadata
# ===========================================================================


class TestCacheFreshness:
    def test_write_parquet_creates_freshness_sidecar(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager

        cache = CacheManager(cache_dir=tmp_path)
        df = _ohlcv_df(50)
        cache.write_parquet("BTC", df)

        sidecar = tmp_path / "BTC.freshness.json"
        assert sidecar.exists(), "Freshness sidecar should be created on write."

        meta = json.loads(sidecar.read_text())
        assert meta["key"] == "BTC"
        assert "ingested_at" in meta
        assert meta["rows"] == 50
        assert meta["min_timestamp"]
        assert meta["max_timestamp"]

    def test_read_freshness_returns_dict_for_existing_key(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager

        cache = CacheManager(cache_dir=tmp_path)
        df = _ohlcv_df(30)
        cache.write_parquet("ETH", df)

        meta = cache.read_freshness("ETH")
        assert meta is not None
        assert meta["rows"] == 30

    def test_read_freshness_returns_none_for_missing_key(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager

        cache = CacheManager(cache_dir=tmp_path)
        assert cache.read_freshness("MISSING") is None

    def test_get_freshness_path_uses_double_underscore_for_slash(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager

        cache = CacheManager(cache_dir=tmp_path)
        fp = cache.get_freshness_path("BTC/USDT")
        assert "BTC__USDT" in fp.name
        assert fp.suffix == ".json"

    def test_is_stale_returns_false_for_fresh_entry(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager

        cache = CacheManager(cache_dir=tmp_path)
        df = _ohlcv_df(20)
        cache.write_parquet("SOL", df)

        assert not cache.is_stale("SOL", max_age_hours=24.0)

    def test_is_stale_returns_true_for_absent_entry(self, tmp_path: Path) -> None:
        from spectraquant_v3.core.cache import CacheManager

        cache = CacheManager(cache_dir=tmp_path)
        assert cache.is_stale("NONEXISTENT", max_age_hours=24.0)

    def test_is_stale_returns_true_when_sidecar_shows_old_ingestion(
        self, tmp_path: Path
    ) -> None:
        from spectraquant_v3.core.cache import CacheManager

        cache = CacheManager(cache_dir=tmp_path)
        df = _ohlcv_df(20)
        cache.write_parquet("DOGE", df)

        # Overwrite the sidecar with an old timestamp
        old_meta = {
            "key": "DOGE",
            "ingested_at": "2020-01-01T00:00:00+00:00",
            "rows": 20,
            "min_timestamp": "2023-01-01",
            "max_timestamp": "2023-04-10",
        }
        cache.get_freshness_path("DOGE").write_text(json.dumps(old_meta))
        assert cache.is_stale("DOGE", max_age_hours=24.0)


# ===========================================================================
# 2. Missing-bar diagnostics
# ===========================================================================


class TestMissingBarDiagnostics:
    def test_no_gaps_returns_zero_missing(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        df = _ohlcv_df(30, "BTC")
        report = diagnose_missing_bars(df, symbol="BTC", expected_freq="1d")

        assert report.canonical_symbol == "BTC"
        assert report.total_present == 30
        assert report.has_gaps is False
        assert report.missing_count == 0

    def test_detects_single_gap(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        df = _ohlcv_df(30, "BTC")
        # Drop one row in the middle to create a gap
        df = df.drop(df.index[15])
        report = diagnose_missing_bars(df, symbol="BTC", expected_freq="1d")

        assert report.has_gaps
        assert report.missing_count == 1
        assert report.first_missing is not None
        assert report.last_missing is not None

    def test_detects_multiple_consecutive_gaps(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        df = _ohlcv_df(30, "ETH")
        # Drop 3 consecutive rows
        df = df.drop(df.index[10:13])
        report = diagnose_missing_bars(df, symbol="ETH", expected_freq="1d")

        assert report.has_gaps
        assert report.missing_count == 3
        assert len(report.gap_runs) >= 1

    def test_empty_df_returns_zero_missing(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        df = pd.DataFrame()
        report = diagnose_missing_bars(df, symbol="SOL", expected_freq="1d")
        assert report.total_expected == 0
        assert report.missing_count == 0
        assert not report.has_gaps

    def test_missing_fraction_is_correct(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        df = _ohlcv_df(20, "BTC")
        df = df.drop(df.index[5:10])  # drop 5 rows
        report = diagnose_missing_bars(df, symbol="BTC", expected_freq="1d")

        assert report.has_gaps
        # missing_fraction = missing / total_expected
        assert 0.0 < report.missing_fraction < 1.0

    def test_summary_string_format(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        df = _ohlcv_df(30, "AVAX")
        report = diagnose_missing_bars(df, symbol="AVAX", expected_freq="1d")
        summary = report.summary()
        assert "AVAX" in summary
        assert "no missing bars" in summary

    def test_to_dict_is_serialisable(self) -> None:
        from spectraquant_v3.crypto.ingestion.missing_bars import diagnose_missing_bars

        df = _ohlcv_df(30, "BTC")
        report = diagnose_missing_bars(df, symbol="BTC", expected_freq="1d")
        d = report.to_dict()
        assert json.dumps(d)  # must be JSON-serialisable
        assert d["canonical_symbol"] == "BTC"


# ===========================================================================
# 3. Symbol-level audit log
# ===========================================================================


class TestIngestionAuditLog:
    def _make_result(
        self,
        symbol: str = "BTC",
        success: bool = True,
        rows: int = 100,
    ):
        from spectraquant_v3.core.ingestion_result import IngestionResult

        return IngestionResult(
            canonical_symbol=symbol,
            asset_class="crypto",
            provider="ccxt/binance",
            success=success,
            rows_loaded=rows,
            cache_hit=False,
            cache_path=f"/tmp/{symbol}.parquet",
            min_timestamp="2023-01-01T00:00:00Z",
            max_timestamp="2023-04-10T00:00:00Z",
            warning_codes=[],
            error_code="" if success else "EMPTY_PRICE_DATA",
            error_message="" if success else "All providers returned no data.",
        )

    def test_record_appends_to_jsonl_file(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditLog

        log_path = tmp_path / "audit.jsonl"
        log = IngestionAuditLog(log_path=log_path, run_id="test_run_001")
        log.record(self._make_result("BTC"))
        log.record(self._make_result("ETH"))

        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        entry = json.loads(lines[0])
        assert entry["canonical_symbol"] == "BTC"
        assert entry["run_id"] == "test_run_001"

    def test_summary_counts_succeed_fail(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditLog

        log = IngestionAuditLog(log_path=tmp_path / "audit.jsonl", run_id="r1")
        log.record(self._make_result("BTC", success=True))
        log.record(self._make_result("ETH", success=True))
        log.record(self._make_result("XRP", success=False, rows=0))

        summary = log.summary()
        assert summary["total"] == 3
        assert summary["succeeded"] == 2
        assert summary["failed"] == 1
        assert "XRP" in summary["failed_symbols"]

    def test_load_from_file_returns_entries(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditLog

        log_path = tmp_path / "audit.jsonl"
        log = IngestionAuditLog(log_path=log_path)
        log.record(self._make_result("SOL"))
        log.record(self._make_result("DOGE"))

        # Load from a new instance
        log2 = IngestionAuditLog(log_path=log_path)
        entries = log2.load_from_file()
        assert len(entries) == 2
        symbols = {e["canonical_symbol"] for e in entries}
        assert symbols == {"SOL", "DOGE"}

    def test_record_creates_parent_directories(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import IngestionAuditLog

        log_path = tmp_path / "deep" / "nested" / "audit.jsonl"
        log = IngestionAuditLog(log_path=log_path)
        log.record(self._make_result("BTC"))
        assert log_path.exists()

    def test_audit_entry_contains_expected_fields(self, tmp_path: Path) -> None:
        from spectraquant_v3.crypto.ingestion.audit_log import (
            IngestionAuditEntry,
            IngestionAuditLog,
        )

        log = IngestionAuditLog(log_path=tmp_path / "audit.jsonl", run_id="run42")
        result = self._make_result("LINK", success=True, rows=200)
        log.record(result)

        entries = log.load_from_file()
        assert len(entries) == 1
        e = entries[0]
        assert e["canonical_symbol"] == "LINK"
        assert e["run_id"] == "run42"
        assert e["rows_loaded"] == 200
        assert e["success"] is True
        assert "recorded_at" in e


# ===========================================================================
# 4. Config v3 files present
# ===========================================================================


class TestConfigV3Files:
    """Verify all required config/v3/ YAML files exist and are parseable."""

    _CONFIG_DIR = Path(__file__).parents[2] / "config" / "v3"

    @pytest.mark.parametrize(
        "filename",
        ["base.yaml", "crypto.yaml", "equities.yaml", "providers.yaml", "news.yaml", "strategies.yaml", "risk.yaml"],
    )
    def test_config_file_exists(self, filename: str) -> None:
        path = self._CONFIG_DIR / filename
        assert path.exists(), f"config/v3/{filename} is required but missing."

    @pytest.mark.parametrize(
        "filename",
        ["base.yaml", "crypto.yaml", "equities.yaml", "providers.yaml", "news.yaml", "strategies.yaml", "risk.yaml"],
    )
    def test_config_file_is_valid_yaml(self, filename: str) -> None:
        path = self._CONFIG_DIR / filename
        content = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), f"{filename} must parse to a dict."

    def test_providers_yaml_has_ccxt_section(self) -> None:
        path = self._CONFIG_DIR / "providers.yaml"
        cfg = yaml.safe_load(path.read_text())
        assert "ccxt" in cfg, "providers.yaml must have a 'ccxt' section."
        assert "retry" in cfg, "providers.yaml must have a 'retry' section."
        assert "rate_limit" in cfg, "providers.yaml must have a 'rate_limit' section."

    def test_news_yaml_has_required_sections(self) -> None:
        path = self._CONFIG_DIR / "news.yaml"
        cfg = yaml.safe_load(path.read_text())
        assert "providers" in cfg
        assert "sentiment" in cfg
        assert "aggregation" in cfg

    def test_strategies_yaml_has_strategies_list(self) -> None:
        path = self._CONFIG_DIR / "strategies.yaml"
        cfg = yaml.safe_load(path.read_text())
        assert "strategies" in cfg
        strategies = cfg["strategies"]
        assert isinstance(strategies, (list, dict))
        assert len(strategies) > 0

    def test_strategies_yaml_entries_have_required_keys(self) -> None:
        path = self._CONFIG_DIR / "strategies.yaml"
        cfg = yaml.safe_load(path.read_text())
        required = {"strategy_id", "asset_class", "agents", "allocator", "policy"}
        entries = cfg["strategies"]
        if isinstance(entries, dict):
            entries = [dict(strategy_id=k, **v) for k, v in entries.items()]
        for s in entries:
            missing = required - set(s.keys())
            assert not missing, (
                f"Strategy {s.get('strategy_id', '?')} is missing keys: {missing}"
            )

    def test_risk_yaml_has_required_sections(self) -> None:
        path = self._CONFIG_DIR / "risk.yaml"
        cfg = yaml.safe_load(path.read_text())
        assert "vol_target" in cfg
        assert "position_limits" in cfg
        assert "drawdown" in cfg

    def test_risk_yaml_vol_target_fields(self) -> None:
        path = self._CONFIG_DIR / "risk.yaml"
        cfg = yaml.safe_load(path.read_text())
        vt = cfg["vol_target"]
        assert "target_vol" in vt
        assert 0.0 < vt["target_vol"] < 1.0, "target_vol should be a fraction < 1."
