from __future__ import annotations

import numpy as np
import pandas as pd

from spectraquant_v3.backtest.engine import BacktestEngine
from spectraquant_v3.crypto.signals.cross_sectional_momentum import (
    CryptoCrossSectionalMomentumAgent,
)
from spectraquant_v3.strategies.loader import StrategyLoader


def _df(ret_5: float, ret_20: float, ret_60: float, ret_120: float, vol: float = 0.2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ret_5d": [ret_5],
            "ret_20d": [ret_20],
            "ret_60d": [ret_60],
            "ret_120d": [ret_120],
            "vol_realised": [vol],
        }
    )


def _ohlcv_with_drift(drift: float, periods: int = 170) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    base = 100.0
    steps = np.arange(periods)
    close = base * np.exp(drift * steps)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(periods, 1000.0),
        },
        index=idx,
    )


def test_cross_sectional_agent_emits_one_row_per_symbol() -> None:
    agent = CryptoCrossSectionalMomentumAgent(run_id="r1", top_n=2)
    feature_map = {
        "BTC": _df(0.03, 0.08, 0.12, 0.20),
        "ETH": _df(0.02, 0.05, 0.07, 0.10),
        "SOL": _df(-0.01, -0.03, -0.04, -0.06),
    }

    rows = agent.evaluate_many(feature_map, as_of="2025-01-01T00:00:00+00:00")
    assert len(rows) == len(feature_map)
    assert sorted(r.canonical_symbol for r in rows) == sorted(feature_map)


def test_cross_sectional_ranking_correctness() -> None:
    agent = CryptoCrossSectionalMomentumAgent(run_id="r1", top_n=3)
    feature_map = {
        "BTC": _df(0.03, 0.08, 0.12, 0.20),
        "ETH": _df(0.02, 0.05, 0.07, 0.10),
        "SOL": _df(-0.01, -0.03, -0.04, -0.06),
    }

    metrics = agent.evaluate_cross_section(feature_map)

    assert metrics["BTC"]["rank"] == 1
    assert metrics["ETH"]["rank"] == 2
    assert metrics["SOL"]["rank"] == 3
    assert metrics["BTC"]["normalized_score"] > metrics["ETH"]["normalized_score"]


def test_zscore_behavior_when_all_raw_scores_equal() -> None:
    agent = CryptoCrossSectionalMomentumAgent(run_id="r1", top_n=2)
    feature_map = {
        "A": _df(0.01, 0.01, 0.01, 0.01),
        "B": _df(0.01, 0.01, 0.01, 0.01),
        "C": _df(0.01, 0.01, 0.01, 0.01),
    }

    metrics = agent.evaluate_cross_section(feature_map)
    assert all(v["normalized_score"] == 0.0 for v in metrics.values())
    assert all(v["confidence"] == 0.0 for v in metrics.values())


def test_top_n_filtering_and_deterministic_symbol_tiebreak() -> None:
    agent = CryptoCrossSectionalMomentumAgent(run_id="r1", top_n=2)
    feature_map = {
        "ETH": _df(0.01, 0.01, 0.01, 0.01),
        "BTC": _df(0.01, 0.01, 0.01, 0.01),
        "ADA": _df(0.01, 0.01, 0.01, 0.01),
        "SOL": _df(0.01, 0.01, 0.01, 0.01),
    }

    rows = agent.evaluate_many(feature_map, as_of="2025-01-01T00:00:00+00:00")
    rows_by_symbol = {r.canonical_symbol: r for r in rows}

    # Equal scores => alphabetical tie-break (ADA, BTC, ETH, SOL)
    assert "rank=1" in rows_by_symbol["ADA"].rationale
    assert "rank=2" in rows_by_symbol["BTC"].rationale
    assert "rank=3" in rows_by_symbol["ETH"].rationale
    assert "rank=4" in rows_by_symbol["SOL"].rationale

    assert rows_by_symbol["ADA"].status == "OK"
    assert rows_by_symbol["BTC"].status == "OK"
    assert rows_by_symbol["ETH"].status == "NO_SIGNAL"
    assert rows_by_symbol["SOL"].status == "NO_SIGNAL"


def test_cross_sectional_strategy_uses_rank_vol_target_allocator() -> None:
    defn = StrategyLoader.load("crypto_cross_sectional_momentum_v1")
    assert defn.allocator == "rank_vol_target_allocator"


def test_cross_sectional_backtest_produces_nonzero_allocations() -> None:
    cfg = {
        "portfolio": {
            "allocator": "vol_target",
            "min_confidence": 0.0,
            "min_signal_threshold": 0.0,
            "target_vol": 0.2,
            "max_weight": 0.8,
            "max_gross_leverage": 1.0,
        },
        "crypto": {
            "signals": {"top_n": 2},
            "universe_top_n": 3,
            "symbols": ["BTC", "ETH", "SOL"],
        },
        "backtest": {"target_vol": 0.2},
    }

    price_data = {
        "BTC": _ohlcv_with_drift(0.003),
        "ETH": _ohlcv_with_drift(0.002),
        "SOL": _ohlcv_with_drift(-0.001),
    }

    engine = BacktestEngine(
        cfg=cfg,
        asset_class="crypto",
        price_data=price_data,
        strategy_id="crypto_cross_sectional_momentum_v1",
        rebalance_freq="W",
        min_in_sample_periods=130,
        run_id="bt_cs",
    )
    results = engine.run()

    assert results.snapshots
    assert any(s.allocations for s in results.snapshots)
    assert any(any(abs(w) > 0 for w in s.allocations.values()) for s in results.snapshots)
    assert any(s.signals_ok > 0 for s in results.snapshots)
    assert any(s.policy_passed > 0 for s in results.snapshots)


def test_registries_include_cross_sectional_agent_and_strategy() -> None:
    from spectraquant_v3.strategies.agents.registry import AgentRegistry
    from spectraquant_v3.strategies.registry import StrategyRegistry

    assert "crypto_cross_sectional_momentum_v1" in AgentRegistry.list()
    assert "crypto_cross_sectional_momentum_v1" in StrategyRegistry.list()
