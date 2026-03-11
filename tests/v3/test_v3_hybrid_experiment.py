"""Tests for hybrid strategy experiment support in SpectraQuant-AI-V3.

Covers:
1. HybridStrategyParams – validation, injection, round-trip serialisation,
   run_id determinism, config_hash stability.
2. run_hybrid_backtest_experiment – runs a BacktestEngine with injected params,
   records blend params, returns structured result dict.
3. compare_hybrid_variants – returns blend params alongside performance metrics
   for each experiment; handles missing experiments cleanly.
4. Baseline vs hybrid comparability – momentum_v1 is unaffected; hybrid
   results differ by variant params.
5. No regression to existing run_experiment / compare_experiments behaviour.

All tests are self-contained (no network calls, no permanent file-system side-effects).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 80, seed: int = 0, trend: float = 0.002) -> pd.DataFrame:
    """Synthetic OHLCV with a mild upward trend (deterministic)."""
    rng = np.random.default_rng(seed)
    daily_ret = trend + rng.standard_normal(n) * 0.01
    close = 100.0 * np.exp(np.cumsum(daily_ret))
    high = close * (1.0 + rng.uniform(0.001, 0.015, n))
    low = close * (1.0 - rng.uniform(0.001, 0.015, n))
    open_ = close * (1.0 + rng.uniform(-0.005, 0.005, n))
    volume = rng.uniform(1_000_000, 5_000_000, n)
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_news_df(n: int = 80, score: float = 0.6, start: str = "2024-01-02") -> pd.DataFrame:
    """Synthetic news sentiment DataFrame with a constant score."""
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"news_sentiment_score": [score] * n}, index=idx)


def _equity_cfg() -> dict:
    from spectraquant_v3.core.config import get_equity_config

    cfg = get_equity_config()
    cfg["equities"]["universe"]["require_exchange_coverage"] = False
    return cfg


def _minimal_equity_cfg() -> dict:
    return {
        "run": {"mode": "normal"},
        "cache": {"root": "data/cache"},
        "qa": {"min_ohlcv_coverage": 1.0},
        "execution": {"mode": "paper"},
        "portfolio": {
            "max_weight": 0.50,
            "max_gross_leverage": 1.0,
            "min_confidence": 0.05,
            "min_signal_threshold": 0.01,
            "target_vol": 0.15,
            "allocator": "equal_weight",
        },
        "equities": {
            "universe": {
                "tickers": ["INFY.NS"],
                "mode": "static",
                "require_exchange_coverage": False,
            },
            "primary_ohlcv_provider": "yfinance",
            "quality_gate": {
                "min_market_cap_usd": 0,
                "min_30d_volume_usd": 0,
                "require_tradable_mapping": True,
            },
            "signals": {
                "momentum_lookback": 20,
                "rsi_period": 14,
                "hybrid_vol_gate": 0.25,
            },
        },
    }


# ===========================================================================
# 1. HybridStrategyParams tests
# ===========================================================================


class TestHybridStrategyParams:
    """Unit tests for HybridStrategyParams dataclass."""

    def test_default_construction(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import (
            HybridStrategyParams,
            EQUITY_HYBRID_ID,
        )

        p = HybridStrategyParams()
        assert p.strategy_id == EQUITY_HYBRID_ID
        assert p.momentum_weight == 0.7
        assert p.news_weight == 0.3
        assert p.vol_gate_threshold == 0.25
        assert p.min_confidence == 0.10
        assert p.min_signal_threshold == 0.05
        assert p.extra == {}

    def test_round_trip_serialisation(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        params = HybridStrategyParams(
            momentum_weight=0.6,
            news_weight=0.4,
            vol_gate_threshold=0.30,
            min_confidence=0.12,
            min_signal_threshold=0.03,
            extra={"description": "test variant"},
        )
        d = params.to_dict()
        restored = HybridStrategyParams.from_dict(d)

        assert restored.momentum_weight == params.momentum_weight
        assert restored.news_weight == params.news_weight
        assert restored.vol_gate_threshold == params.vol_gate_threshold
        assert restored.min_confidence == params.min_confidence
        assert restored.min_signal_threshold == params.min_signal_threshold
        assert restored.extra == params.extra

    def test_run_id_is_deterministic(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        p = HybridStrategyParams(momentum_weight=0.6, news_weight=0.4)
        assert p.run_id() == p.run_id()
        assert "emnh" in p.run_id()
        assert "mw0.60" in p.run_id()
        assert "nw0.40" in p.run_id()

    def test_run_id_differs_between_variants(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        p1 = HybridStrategyParams(momentum_weight=0.7, news_weight=0.3)
        p2 = HybridStrategyParams(momentum_weight=0.5, news_weight=0.5)
        assert p1.run_id() != p2.run_id()

    def test_config_hash_stable(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        p = HybridStrategyParams(momentum_weight=0.7, news_weight=0.3)
        assert p.config_hash() == p.config_hash()
        assert len(p.config_hash()) == 12

    def test_config_hash_differs_between_variants(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        p1 = HybridStrategyParams(momentum_weight=0.7, news_weight=0.3)
        p2 = HybridStrategyParams(momentum_weight=0.5, news_weight=0.5)
        assert p1.config_hash() != p2.config_hash()

    def test_inject_into_cfg_does_not_mutate_original(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        base = {"portfolio": {"min_confidence": 0.10}, "strategies": {}}
        p = HybridStrategyParams(momentum_weight=0.6, news_weight=0.4)
        cfg = p.inject_into_cfg(base)

        # Original unchanged
        assert base["portfolio"]["min_confidence"] == 0.10
        assert "equity_momentum_news_hybrid_v1" not in base.get("strategies", {})

        # Injected config has correct values
        sid = p.strategy_id
        assert cfg["strategies"][sid]["signal_blend"]["momentum_weight"] == 0.6
        assert cfg["strategies"][sid]["signal_blend"]["news_weight"] == 0.4
        assert cfg["strategies"][sid]["vol_gate"]["threshold"] == p.vol_gate_threshold
        assert cfg["portfolio"]["min_confidence"] == p.min_confidence

    def test_inject_preserves_other_portfolio_keys(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        base = {
            "portfolio": {
                "max_weight": 0.25,
                "allocator": "equal_weight",
                "min_confidence": 0.10,
            }
        }
        p = HybridStrategyParams()
        cfg = p.inject_into_cfg(base)
        assert cfg["portfolio"]["max_weight"] == 0.25
        assert cfg["portfolio"]["allocator"] == "equal_weight"

    def test_invalid_strategy_id_raises(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        with pytest.raises(ValueError, match="strategy_id"):
            HybridStrategyParams(strategy_id="not_a_hybrid_strategy")

    def test_invalid_momentum_weight_raises(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        with pytest.raises(ValueError, match="momentum_weight"):
            HybridStrategyParams(momentum_weight=0.0)

    def test_invalid_news_weight_raises(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        with pytest.raises(ValueError, match="news_weight"):
            HybridStrategyParams(news_weight=-0.1)

    def test_invalid_vol_gate_raises(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        with pytest.raises(ValueError, match="vol_gate_threshold"):
            HybridStrategyParams(vol_gate_threshold=-0.01)

    def test_crypto_hybrid_id_accepted(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import (
            HybridStrategyParams,
            CRYPTO_HYBRID_ID,
        )

        p = HybridStrategyParams(strategy_id=CRYPTO_HYBRID_ID)
        assert p.strategy_id == CRYPTO_HYBRID_ID
        assert "cmnh" in p.run_id()

    def test_to_dict_contains_all_expected_keys(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        p = HybridStrategyParams()
        d = p.to_dict()
        for key in (
            "strategy_id",
            "momentum_weight",
            "news_weight",
            "vol_gate_threshold",
            "min_confidence",
            "min_signal_threshold",
            "extra",
        ):
            assert key in d, f"Missing key: {key}"


# ===========================================================================
# 2. run_hybrid_backtest_experiment tests
# ===========================================================================


class TestRunHybridBacktestExperiment:
    """Integration tests for ExperimentManager.run_hybrid_backtest_experiment."""

    def test_dry_run_returns_structured_result(self) -> None:
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        news = {"INFY.NS": _make_news_df(n=80)}
        params = HybridStrategyParams(momentum_weight=0.7, news_weight=0.3)
        mgr = ExperimentManager()

        result = mgr.run_hybrid_backtest_experiment(
            experiment_id="test_dry_hybrid_001",
            params=params,
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            news_feature_map=news,
            min_in_sample_periods=20,
            dry_run=True,
        )

        assert result["experiment_id"] == "test_dry_hybrid_001"
        assert result["strategy_id"] == params.strategy_id
        assert result["run_id"] == params.run_id()
        assert result["hybrid_params"]["momentum_weight"] == 0.7
        assert result["hybrid_params"]["news_weight"] == 0.3
        assert isinstance(result["metrics"], dict)
        assert isinstance(result["n_steps"], int)
        assert result["n_steps"] >= 0

    def test_persists_hybrid_params_json(self, tmp_path: "Path") -> None:
        import json
        from pathlib import Path
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        news = {"INFY.NS": _make_news_df(n=80)}
        params = HybridStrategyParams(momentum_weight=0.6, news_weight=0.4)
        mgr = ExperimentManager(base_dir=tmp_path)

        mgr.run_hybrid_backtest_experiment(
            experiment_id="exp_persist_001",
            params=params,
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            news_feature_map=news,
            min_in_sample_periods=20,
            dry_run=False,
        )

        hp_path = tmp_path / "exp_persist_001" / "hybrid_params.json"
        assert hp_path.exists(), "hybrid_params.json should be written"
        stored = json.loads(hp_path.read_text())
        assert stored["momentum_weight"] == 0.6
        assert stored["news_weight"] == 0.4
        assert stored["vol_gate_threshold"] == params.vol_gate_threshold

    def test_persists_config_json_with_strategy_id(self, tmp_path: "Path") -> None:
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        params = HybridStrategyParams()
        mgr = ExperimentManager(base_dir=tmp_path)

        mgr.run_hybrid_backtest_experiment(
            experiment_id="exp_config_001",
            params=params,
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            min_in_sample_periods=20,
            dry_run=False,
        )

        config_doc = mgr.store.read_config("exp_config_001")
        assert config_doc["strategy_id"] == params.strategy_id
        assert config_doc["experiment_id"] == "exp_config_001"
        assert "config_hash" in config_doc

    def test_blend_params_injected_into_backtest(self, tmp_path: "Path") -> None:
        """Different blend params produce different experiment records."""
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        news = {"INFY.NS": _make_news_df(n=80, score=0.9)}
        mgr = ExperimentManager(base_dir=tmp_path)

        params_a = HybridStrategyParams(momentum_weight=0.8, news_weight=0.2)
        params_b = HybridStrategyParams(momentum_weight=0.2, news_weight=0.8)

        result_a = mgr.run_hybrid_backtest_experiment(
            experiment_id="exp_blend_a",
            params=params_a,
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            news_feature_map=news,
            min_in_sample_periods=20,
            dry_run=False,
        )
        result_b = mgr.run_hybrid_backtest_experiment(
            experiment_id="exp_blend_b",
            params=params_b,
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            news_feature_map=news,
            min_in_sample_periods=20,
            dry_run=False,
        )

        # run_ids must differ
        assert result_a["run_id"] != result_b["run_id"]
        # hybrid_params recorded separately
        assert result_a["hybrid_params"]["momentum_weight"] == 0.8
        assert result_b["hybrid_params"]["momentum_weight"] == 0.2

    def test_no_news_degrades_gracefully(self) -> None:
        """run_hybrid_backtest_experiment completes without news_feature_map."""
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        params = HybridStrategyParams()
        mgr = ExperimentManager()

        result = mgr.run_hybrid_backtest_experiment(
            experiment_id="exp_no_news",
            params=params,
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            news_feature_map=None,
            min_in_sample_periods=20,
            dry_run=True,
        )
        assert result["experiment_id"] == "exp_no_news"
        assert isinstance(result["n_steps"], int)


# ===========================================================================
# 3. compare_hybrid_variants tests
# ===========================================================================


class TestCompareHybridVariants:
    """Tests for ExperimentManager.compare_hybrid_variants."""

    def test_compare_returns_blend_params_per_experiment(self, tmp_path: "Path") -> None:
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        mgr = ExperimentManager(base_dir=tmp_path)

        variants = [
            ("exp_cmp_a", HybridStrategyParams(momentum_weight=0.7, news_weight=0.3)),
            ("exp_cmp_b", HybridStrategyParams(momentum_weight=0.5, news_weight=0.5)),
        ]
        for eid, p in variants:
            mgr.run_hybrid_backtest_experiment(
                experiment_id=eid,
                params=p,
                price_data=price_data,
                base_cfg=_minimal_equity_cfg(),
                min_in_sample_periods=20,
                dry_run=False,
            )

        table = mgr.compare_hybrid_variants(["exp_cmp_a", "exp_cmp_b"])
        assert len(table) == 2

        row_a = next(r for r in table if r["experiment_id"] == "exp_cmp_a")
        row_b = next(r for r in table if r["experiment_id"] == "exp_cmp_b")

        assert row_a["momentum_weight"] == 0.7
        assert row_a["news_weight"] == 0.3
        assert row_b["momentum_weight"] == 0.5
        assert row_b["news_weight"] == 0.5

    def test_compare_includes_standard_metrics(self, tmp_path: "Path") -> None:
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        mgr = ExperimentManager(base_dir=tmp_path)

        mgr.run_hybrid_backtest_experiment(
            experiment_id="exp_metrics_check",
            params=HybridStrategyParams(),
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            min_in_sample_periods=20,
            dry_run=False,
        )

        table = mgr.compare_hybrid_variants(["exp_metrics_check"])
        assert len(table) == 1
        row = table[0]
        for key in (
            "experiment_id",
            "strategy_id",
            "momentum_weight",
            "news_weight",
            "vol_gate_threshold",
            "min_confidence",
            "sharpe",
            "cagr",
            "max_drawdown",
            "n_steps",
        ):
            assert key in row, f"Missing key in comparison row: {key}"

    def test_compare_handles_missing_experiment(self, tmp_path: "Path") -> None:
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager

        mgr = ExperimentManager(base_dir=tmp_path)
        table = mgr.compare_hybrid_variants(["nonexistent_exp"])
        assert len(table) == 1
        assert table[0]["experiment_id"] == "nonexistent_exp"
        # Should not raise; momentum_weight may be None (no data) or error key present

    def test_compare_mixed_found_and_missing(self, tmp_path: "Path") -> None:
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        mgr = ExperimentManager(base_dir=tmp_path)

        mgr.run_hybrid_backtest_experiment(
            experiment_id="exp_found",
            params=HybridStrategyParams(),
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            min_in_sample_periods=20,
            dry_run=False,
        )

        table = mgr.compare_hybrid_variants(["exp_found", "exp_missing"])
        assert len(table) == 2
        ids = [r["experiment_id"] for r in table]
        assert "exp_found" in ids
        assert "exp_missing" in ids

    def test_comparison_is_reproducible(self, tmp_path: "Path") -> None:
        """Same experiments produce the same comparison table on two calls."""
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        mgr = ExperimentManager(base_dir=tmp_path)

        mgr.run_hybrid_backtest_experiment(
            experiment_id="exp_repro",
            params=HybridStrategyParams(momentum_weight=0.6, news_weight=0.4),
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            min_in_sample_periods=20,
            dry_run=False,
        )

        table1 = mgr.compare_hybrid_variants(["exp_repro"])
        table2 = mgr.compare_hybrid_variants(["exp_repro"])

        assert table1[0]["momentum_weight"] == table2[0]["momentum_weight"]
        assert table1[0]["sharpe"] == table2[0]["sharpe"]
        assert table1[0]["config_hash"] == table2[0]["config_hash"]


# ===========================================================================
# 4. Baseline vs hybrid comparability
# ===========================================================================


class TestBaselineVsHybridComparability:
    """run_experiment (baseline) is unaffected; hybrid variants are distinct."""

    def test_hybrid_run_id_differs_by_params(self) -> None:
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        p_default = HybridStrategyParams()
        p_news_heavy = HybridStrategyParams(momentum_weight=0.3, news_weight=0.7)
        assert p_default.run_id() != p_news_heavy.run_id()

    def test_experiment_metadata_preserved_across_runs(self, tmp_path: "Path") -> None:
        """Running the same params twice produces the same config_hash."""
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        params = HybridStrategyParams(momentum_weight=0.65, news_weight=0.35)
        mgr1 = ExperimentManager(base_dir=tmp_path / "run1")
        mgr2 = ExperimentManager(base_dir=tmp_path / "run2")

        r1 = mgr1.run_hybrid_backtest_experiment(
            experiment_id="exp_repro1",
            params=params,
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            min_in_sample_periods=20,
            dry_run=False,
        )
        r2 = mgr2.run_hybrid_backtest_experiment(
            experiment_id="exp_repro1",
            params=params,
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            min_in_sample_periods=20,
            dry_run=False,
        )

        # run_id is deterministic from params
        assert r1["run_id"] == r2["run_id"]
        # hybrid_params round-trips correctly
        assert r1["hybrid_params"] == r2["hybrid_params"]
        # config_hash in persisted record matches
        cfg1 = mgr1.store.read_config("exp_repro1")
        cfg2 = mgr2.store.read_config("exp_repro1")
        assert cfg1["config_hash"] == cfg2["config_hash"]

    def test_compare_hybrid_variants_vs_compare_experiments(
        self, tmp_path: "Path"
    ) -> None:
        """compare_hybrid_variants is a superset of compare_experiments."""
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        params = HybridStrategyParams()
        mgr = ExperimentManager(base_dir=tmp_path)

        mgr.run_hybrid_backtest_experiment(
            experiment_id="exp_super",
            params=params,
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            min_in_sample_periods=20,
            dry_run=False,
        )

        base_row = mgr.compare_experiments(["exp_super"])[0]
        hybrid_row = mgr.compare_hybrid_variants(["exp_super"])[0]

        # Hybrid row contains all standard metric keys
        for key in ("sharpe", "cagr", "max_drawdown", "win_rate"):
            assert key in hybrid_row

        # Hybrid row additionally contains blend params
        assert hybrid_row["momentum_weight"] == params.momentum_weight
        assert hybrid_row["news_weight"] == params.news_weight
        assert hybrid_row["vol_gate_threshold"] == params.vol_gate_threshold


# ===========================================================================
# 5. Regression – existing run_experiment / compare_experiments unaffected
# ===========================================================================


class TestNoRegression:
    """Existing experiment manager methods continue to work correctly."""

    def test_run_experiment_dry_run_still_works(self) -> None:
        """run_experiment with dry_run=True completes without error."""
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager

        mgr = ExperimentManager()
        result = mgr.run_experiment(
            experiment_id="regression_dry",
            strategy_id="equity_momentum_v1",
            cfg=_minimal_equity_cfg(),
            dry_run=True,
            price_data={"INFY.NS": _make_ohlcv(n=80)},
        )
        assert result["experiment_id"] == "regression_dry"
        assert result["strategy_id"] == "equity_momentum_v1"

    def test_compare_experiments_no_change_in_keys(self, tmp_path: "Path") -> None:
        """compare_experiments still returns its expected keys."""
        from spectraquant_v3.experiments.experiment_manager import ExperimentManager
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

        price_data = {"INFY.NS": _make_ohlcv(n=80)}
        params = HybridStrategyParams()
        mgr = ExperimentManager(base_dir=tmp_path)

        mgr.run_hybrid_backtest_experiment(
            experiment_id="exp_reg_001",
            params=params,
            price_data=price_data,
            base_cfg=_minimal_equity_cfg(),
            min_in_sample_periods=20,
            dry_run=False,
        )

        rows = mgr.compare_experiments(["exp_reg_001"])
        assert len(rows) == 1
        row = rows[0]
        for key in (
            "experiment_id",
            "strategy_id",
            "sharpe",
            "cagr",
            "max_drawdown",
            "config_hash",
            "run_timestamp",
        ):
            assert key in row, f"compare_experiments missing key: {key}"

    def test_hybrid_params_module_exports(self) -> None:
        """Public exports from experiments package are importable."""
        from spectraquant_v3.experiments import (
            ExperimentManager,
            HybridStrategyParams,
            ResultStore,
            RunTracker,
            EQUITY_HYBRID_ID,
            CRYPTO_HYBRID_ID,
            HYBRID_STRATEGY_IDS,
        )

        assert issubclass(HybridStrategyParams, object)
        assert EQUITY_HYBRID_ID == "equity_momentum_news_hybrid_v1"
        assert CRYPTO_HYBRID_ID == "crypto_momentum_news_hybrid_v1"
        assert EQUITY_HYBRID_ID in HYBRID_STRATEGY_IDS
        assert CRYPTO_HYBRID_ID in HYBRID_STRATEGY_IDS
