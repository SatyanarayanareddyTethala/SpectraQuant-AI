from __future__ import annotations

import pandas as pd
import pytest

from spectraquant_v3.core.config import get_crypto_config
from spectraquant_v3.crypto.signals.hybrid import (
    CryptoMomentumNewsHybridAgent,
    compose_hybrid_score,
)
from spectraquant_v3.pipeline import run_strategy
from spectraquant_v3.strategies.agents.registry import AgentRegistry
from spectraquant_v3.strategies.registry import StrategyRegistry


def test_compose_hybrid_blend_math() -> None:
    score = compose_hybrid_score(
        momentum_score=0.5,
        news_sentiment_24h=-0.2,
        momentum_weight=0.7,
        news_weight=0.3,
    )
    assert score == 0.29


def test_compose_hybrid_threshold_uplift_bounded() -> None:
    score = compose_hybrid_score(
        momentum_score=0.4,
        news_sentiment_24h=0.1,
        news_shock_zscore=3.0,
        momentum_weight=0.7,
        news_weight=0.3,
        shock_threshold=2.0,
        shock_max_uplift=0.15,
    )
    assert score == pytest.approx(0.46)


def test_compose_hybrid_fallback_when_news_missing() -> None:
    score = compose_hybrid_score(momentum_score=0.33, news_sentiment_24h=None)
    assert score == 0.33


def test_registry_contains_hybrid_agent_and_strategy() -> None:
    assert "crypto_momentum_news_hybrid_v1" in AgentRegistry.list()
    assert "crypto_momentum_news_hybrid_v1" in StrategyRegistry.list()


def test_run_strategy_accepts_single_dataset_interface_with_news_columns() -> None:
    cfg = get_crypto_config()
    cfg["crypto"]["universe_filters"]["require_exchange_coverage"] = False

    idx = pd.date_range("2025-01-01", periods=30, freq="D")
    ds = pd.DataFrame(
        {
            "ret_20d": [0.03] * 30,
            "rsi": [55.0] * 30,
            "news_sentiment_24h": [0.2] * 30,
            "news_shock_zscore": [2.5] * 30,
            "vol_realised": [0.2] * 30,
        },
        index=idx,
    )

    result = run_strategy(
        "crypto_momentum_news_hybrid_v1",
        cfg=cfg,
        dry_run=True,
        market_data={"BTC": {"market_cap_usd": 1e10, "volume_24h_usd": 1e9, "age_days": 3650}},
        dataset={"BTC": ds},
    )

    assert result["status"] == "success"
    btc_rows = [s for s in result["signals"] if s.canonical_symbol == "BTC"]
    assert btc_rows
    assert btc_rows[0].agent_id == "crypto_momentum_news_hybrid_v1"
    assert btc_rows[0].signal_score > 0.0


def test_hybrid_agent_uses_strategy_config_weights() -> None:
    cfg = {
        "crypto": {"signals": {"momentum_lookback": 20}},
        "strategies": {
            "crypto_momentum_news_hybrid_v1": {
                "signal_blend": {"momentum_weight": 0.6, "news_weight": 0.4},
                "shock_uplift": {"threshold": 1.5, "max_uplift": 0.1},
            }
        },
    }
    agent = CryptoMomentumNewsHybridAgent.from_config(cfg, run_id="r1")
    assert agent.momentum_weight == 0.6
    assert agent.news_weight == 0.4
    assert agent.shock_threshold == 1.5
    assert agent.shock_max_uplift == 0.1
