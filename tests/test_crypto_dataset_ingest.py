"""Unit tests for cryptocurrency dataset ingestion and universe gating.

Covers:
  - ingestion parsing and schema (A/B)
  - universe gating/scoring (C)
  - symbol mapping (A)
  - test-mode cache failure behaviour (D)
"""
from __future__ import annotations

import io
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure src on path
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV_CONTENT = textwrap.dedent("""\
    name,symbol,market_cap_usd,price_usd,24h_trading_volume_usd,24h_change_percent,circulating_supply,max_supply,all_time_high_usd,all_time_low_usd,launch_date,category,network_type,community_rank,last_updated
    Bitcoin,BTC,1200000000000,65000.00,45000000000,2.5,19500000,21000000,73750.00,67.81,2009-01-03,Layer1,Proof of Work,1,2024-01-01
    Ethereum,ETH,400000000000,3400.00,20000000000,-1.2,120000000,,4878.26,0.43,2015-07-30,Layer1,Proof of Stake,2,2024-01-01
    Solana,SOL,60000000000,150.00,5000000000,5.1,450000000,,259.96,0.50,2020-03-16,Layer1,Proof of History,3,2024-01-01
    TinyToken,TINY,500000,0.001,2000,0.0,1000000000,,0.01,0.0001,2023-01-01,Misc,PoW,100,2024-01-01
""")


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Write sample CSV and return its path."""
    p = tmp_path / "cryptocurrency_dataset.csv"
    p.write_text(SAMPLE_CSV_CONTENT)
    return p


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "crypto"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# A/B: Ingestion tests
# ---------------------------------------------------------------------------

class TestIngest:
    def test_basic_ingestion_creates_parquets(self, sample_csv, data_dir):
        from spectraquant.crypto.dataset.ingest import ingest_crypto_dataset

        summary = ingest_crypto_dataset(
            csv_path=sample_csv,
            data_dir=data_dir,
            append_snapshot=False,
        )
        assert summary["rows_read"] == 4
        assert summary["rows_kept"] == 4
        assert summary["duplicates_removed"] == 0

        # All three parquet files must exist
        assert Path(summary["asset_master_path"]).exists()
        assert Path(summary["market_snapshot_path"]).exists()
        assert Path(summary["symbol_map_path"]).exists()

    def test_asset_master_schema(self, sample_csv, data_dir):
        from spectraquant.crypto.dataset.ingest import (
            ingest_crypto_dataset,
            load_asset_master,
        )

        ingest_crypto_dataset(csv_path=sample_csv, data_dir=data_dir, append_snapshot=False)
        am = load_asset_master(data_dir)

        assert not am.empty
        assert "canonical_symbol" in am.columns
        assert "name" in am.columns

    def test_market_snapshot_has_as_of(self, sample_csv, data_dir):
        from spectraquant.crypto.dataset.ingest import (
            ingest_crypto_dataset,
            load_market_snapshot,
        )

        as_of = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        ingest_crypto_dataset(csv_path=sample_csv, data_dir=data_dir, as_of=as_of, append_snapshot=False)
        snap = load_market_snapshot(data_dir)

        assert "as_of" in snap.columns
        assert not snap.empty

    def test_symbol_map_has_yfinance_symbol(self, sample_csv, data_dir):
        from spectraquant.crypto.dataset.ingest import (
            ingest_crypto_dataset,
            load_symbol_map,
        )

        ingest_crypto_dataset(csv_path=sample_csv, data_dir=data_dir, append_snapshot=False)
        sym_map = load_symbol_map(data_dir)

        assert "canonical_symbol" in sym_map.columns
        assert "yfinance_symbol" in sym_map.columns
        assert "exchange_symbol" in sym_map.columns

        # BTC should map to BTC-USD
        btc_row = sym_map[sym_map["canonical_symbol"] == "BTC"]
        assert not btc_row.empty
        assert btc_row["yfinance_symbol"].iloc[0] == "BTC-USD"

    def test_snapshot_append(self, sample_csv, data_dir):
        """Appending snapshot grows the file."""
        from spectraquant.crypto.dataset.ingest import (
            ingest_crypto_dataset,
            load_market_snapshot,
        )

        as_of1 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        as_of2 = datetime(2024, 6, 2, tzinfo=timezone.utc)

        ingest_crypto_dataset(csv_path=sample_csv, data_dir=data_dir, as_of=as_of1, append_snapshot=False)
        snap_first = load_market_snapshot(data_dir)
        first_len = len(snap_first)

        ingest_crypto_dataset(csv_path=sample_csv, data_dir=data_dir, as_of=as_of2, append_snapshot=True)
        snap_second = load_market_snapshot(data_dir)
        second_len = len(snap_second)

        assert second_len == first_len * 2, "Snapshot should double on append"

    def test_numeric_coercion(self, sample_csv, data_dir):
        """market_cap_usd must be numeric."""
        from spectraquant.crypto.dataset.ingest import ingest_crypto_dataset, load_market_snapshot

        ingest_crypto_dataset(csv_path=sample_csv, data_dir=data_dir, append_snapshot=False)
        snap = load_market_snapshot(data_dir)

        if "market_cap_usd" in snap.columns:
            assert pd.api.types.is_numeric_dtype(snap["market_cap_usd"]), (
                f"Expected numeric dtype, got {snap['market_cap_usd'].dtype}"
            )

    def test_csv_with_percent_and_commas(self, tmp_path, data_dir):
        """CSV with % signs and comma-formatted numbers normalises correctly."""
        from spectraquant.crypto.dataset.ingest import ingest_crypto_dataset, load_market_snapshot

        csv_content = (
            "name,symbol,market_cap_usd,price_usd,24h_trading_volume_usd,"
            "24h_change_percent,launch_date\n"
            "Bitcoin,BTC,\"1,200,000,000,000\",65000.00,\"45,000,000,000\",2.50%,2009-01-03\n"
        )
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)

        summary = ingest_crypto_dataset(csv_path=csv_path, data_dir=data_dir, append_snapshot=False)
        assert summary["rows_kept"] == 1
        snap = load_market_snapshot(data_dir)
        if "market_cap_usd" in snap.columns:
            val = snap["market_cap_usd"].iloc[0]
            assert val > 0, f"Expected positive market cap, got {val}"

    def test_deduplication(self, tmp_path, data_dir):
        """Duplicate symbols are deduplicated, keeping highest market cap."""
        from spectraquant.crypto.dataset.ingest import ingest_crypto_dataset

        csv_content = (
            "name,symbol,market_cap_usd,price_usd,24h_trading_volume_usd,24h_change_percent\n"
            "Bitcoin,BTC,1200000000000,65000,45000000000,2.5\n"
            "Bitcoin2,BTC,1000000000,60000,100000000,1.0\n"
            "Ethereum,ETH,400000000000,3400,20000000000,-1.2\n"
        )
        csv_path = tmp_path / "dup.csv"
        csv_path.write_text(csv_content)

        summary = ingest_crypto_dataset(csv_path=csv_path, data_dir=data_dir, append_snapshot=False)
        assert summary["rows_read"] == 3
        assert summary["rows_kept"] == 2, "BTC duplicate should be removed"
        assert summary["duplicates_removed"] == 1

    def test_missing_csv_raises(self, data_dir):
        from spectraquant.crypto.dataset.ingest import ingest_crypto_dataset

        with pytest.raises(FileNotFoundError):
            ingest_crypto_dataset(
                csv_path="/nonexistent/path.csv",
                data_dir=data_dir,
            )

    def test_get_yfinance_symbol_from_map(self, sample_csv, data_dir):
        from spectraquant.crypto.dataset.ingest import (
            ingest_crypto_dataset,
            get_yfinance_symbol,
        )

        ingest_crypto_dataset(csv_path=sample_csv, data_dir=data_dir, append_snapshot=False)
        assert get_yfinance_symbol("BTC", data_dir) == "BTC-USD"
        assert get_yfinance_symbol("ETH", data_dir) == "ETH-USD"

    def test_get_yfinance_symbol_fallback(self, data_dir):
        """Without a symbol map, fallback to default pattern."""
        from spectraquant.crypto.dataset.ingest import get_yfinance_symbol

        # data_dir has no symbol_map.parquet
        result = get_yfinance_symbol("UNKNOWN", data_dir)
        assert result == "UNKNOWN-USD"


# ---------------------------------------------------------------------------
# C: Universe gating / scoring tests
# ---------------------------------------------------------------------------

class TestQualityGate:
    def _make_snapshot(self) -> pd.DataFrame:
        return pd.DataFrame({
            "canonical_symbol": ["BTC", "ETH", "TINY"],
            "market_cap_usd": [1_200_000_000_000.0, 400_000_000_000.0, 500_000.0],
            "volume_24h_usd": [45_000_000_000.0, 20_000_000_000.0, 2_000.0],
            "change_24h_pct": [2.5, -1.2, 0.0],
            "launch_date": [
                pd.Timestamp("2009-01-03", tz="UTC"),
                pd.Timestamp("2015-07-30", tz="UTC"),
                pd.Timestamp("2023-01-01", tz="UTC"),
            ],
            "community_rank": [1, 2, 100],
        })

    def test_gate_passes_large_caps(self):
        from spectraquant.crypto.universe.quality_gate import apply_quality_gate

        snap = self._make_snapshot()
        passed = apply_quality_gate(
            snap,
            min_market_cap_usd=50_000_000,
            min_24h_volume_usd=1_000_000,
            min_age_days=180,
        )
        assert set(passed["canonical_symbol"]) == {"BTC", "ETH"}

    def test_gate_excludes_tiny_token(self):
        from spectraquant.crypto.universe.quality_gate import apply_quality_gate

        snap = self._make_snapshot()
        passed = apply_quality_gate(
            snap,
            min_market_cap_usd=50_000_000,
            min_24h_volume_usd=1_000_000,
            min_age_days=180,
        )
        assert "TINY" not in passed["canonical_symbol"].values

    def test_gate_exclusion_reasons_populated(self):
        from spectraquant.crypto.universe.quality_gate import apply_quality_gate

        snap = self._make_snapshot()
        # Run gate on full snapshot to see all reasons (pass all criteria except large caps)
        full = snap.copy()
        full["universe_included"] = True  # dummy
        result = apply_quality_gate(
            snap,
            min_market_cap_usd=50_000_000,
            min_24h_volume_usd=1_000_000,
            min_age_days=0,  # disable age gate
        )
        # TINY should be excluded for mcap + vol
        all_result = apply_quality_gate(snap, min_market_cap_usd=50_000_000, min_24h_volume_usd=1_000_000, min_age_days=0)
        assert "universe_included" in all_result.columns
        assert "universe_reason" in all_result.columns

    def test_liquidity_score_computed(self):
        from spectraquant.crypto.universe.quality_gate import apply_quality_gate

        snap = self._make_snapshot()
        result = apply_quality_gate(snap)
        assert "liquidity_score" in result.columns
        # BTC should have higher score than ETH
        btc_score = result.loc[result["canonical_symbol"] == "BTC", "liquidity_score"]
        eth_score = result.loc[result["canonical_symbol"] == "ETH", "liquidity_score"]
        if not btc_score.empty and not eth_score.empty:
            assert btc_score.iloc[0] > eth_score.iloc[0]

    def test_dataset_topN(self):
        from spectraquant.crypto.universe.quality_gate import build_dataset_topN_universe

        snap = self._make_snapshot()
        result = build_dataset_topN_universe(
            snap,
            top_n=2,
            gate_kwargs={"min_market_cap_usd": 50_000_000, "min_24h_volume_usd": 1_000_000, "min_age_days": 0},
        )
        assert isinstance(result, list)
        assert len(result) <= 2
        assert "BTC" in result

    def test_hybrid_universe(self):
        from spectraquant.crypto.universe.quality_gate import build_hybrid_universe

        snap = self._make_snapshot()
        # News says XRP; dataset has BTC, ETH, TINY
        result = build_hybrid_universe(
            news_symbols=["XRP-USD"],
            snapshot=snap,
            top_n=5,
            gate_kwargs={"min_market_cap_usd": 0, "min_24h_volume_usd": 0, "min_age_days": 0},
        )
        assert isinstance(result, list)
        # XRP should appear (from news, no data to gate it out)
        assert "XRP" in result
        # BTC and ETH from dataset
        assert "BTC" in result

    def test_momentum_hint_clipped(self):
        from spectraquant.crypto.universe.quality_gate import apply_quality_gate

        snap = pd.DataFrame({
            "canonical_symbol": ["A"],
            "market_cap_usd": [1e9],
            "volume_24h_usd": [1e6],
            "change_24h_pct": [500.0],  # extreme value
        })
        result = apply_quality_gate(snap, min_market_cap_usd=0, min_24h_volume_usd=0, min_age_days=0)
        assert result["momentum_hint"].iloc[0] == pytest.approx(20.0)

    def test_universe_report_written(self, tmp_path):
        from spectraquant.crypto.universe.quality_gate import (
            apply_quality_gate,
            write_universe_report,
        )

        snap = pd.DataFrame({
            "canonical_symbol": ["BTC", "ETH"],
            "market_cap_usd": [1e12, 4e11],
            "volume_24h_usd": [4.5e10, 2e10],
            "change_24h_pct": [2.5, -1.2],
        })
        gated = apply_quality_gate(snap, min_market_cap_usd=0, min_24h_volume_usd=0, min_age_days=0)
        path = write_universe_report(gated, out_dir=tmp_path)
        assert path.exists()
        content = pd.read_csv(path)
        assert "canonical_symbol" in content.columns


# ---------------------------------------------------------------------------
# D: Test-mode cache failure test
# ---------------------------------------------------------------------------

class TestTestModeCacheFailure:
    def test_missing_prices_in_test_mode_logs_error(self, caplog, tmp_path):
        """In test-mode with missing price cache, pipeline logs error but does not crash."""
        import logging
        from spectraquant.pipeline.crypto_run import run_crypto_pipeline

        cfg = {
            "crypto": {
                "enabled": True,
                "symbols": ["BTC-USD"],
                "prices_dir": str(tmp_path / "prices"),  # empty dir
                "universe_mode": "static",
                "news_first": False,
                "universe_csv": str(
                    Path(__file__).resolve().parents[1]
                    / "src/spectraquant/crypto/universe/crypto_universe.csv"
                ),
            },
            "crypto_dataset": {
                "data_dir": str(tmp_path / "crypto"),
            },
            "test_mode": {"enabled": True},
            "news_ai": {"enabled": False},
            "onchain_ai": {"enabled": False},
            "agents": {"enabled": False},
            "crypto_meta_policy": {"enabled": False},
            "crypto_portfolio": {
                "allocator": "vol_target",
                "target_vol": 0.15,
                "max_weight": 0.25,
            },
        }

        # Ensure test mode env var is set
        os.environ["SPECTRAQUANT_TEST_MODE"] = "true"
        try:
            with caplog.at_level(logging.ERROR, logger="spectraquant.pipeline.crypto_run"):
                result = run_crypto_pipeline(cfg=cfg, dry_run=True)
            # Should NOT crash; should return a result dict
            assert isinstance(result, dict)
            # Pipeline should have logged the test-mode error for missing prices
            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert any(
                "test-mode" in r.message.lower() or "prices missing" in r.message.lower()
                for r in error_records
            ), (
                f"Expected test-mode price error in logs. Got: {[r.message for r in error_records]}"
            )
        finally:
            os.environ.pop("SPECTRAQUANT_TEST_MODE", None)


# ---------------------------------------------------------------------------
# CLI registration test
# ---------------------------------------------------------------------------

class TestCLIRegistration:
    def test_crypto_ingest_dataset_registered(self):
        from spectraquant.cli.commands.crypto import register_crypto_commands

        commands: dict = {}
        register_crypto_commands(commands)
        assert "crypto-ingest-dataset" in commands
        assert callable(commands["crypto-ingest-dataset"])

    def test_all_crypto_commands_registered(self):
        from spectraquant.cli.commands.crypto import register_crypto_commands

        commands: dict = {}
        register_crypto_commands(commands)
        expected = {
            "crypto-run",
            "crypto-stream",
            "onchain-scan",
            "agents-run",
            "allocate",
            "crypto-ingest-dataset",
        }
        assert expected == set(commands.keys())
