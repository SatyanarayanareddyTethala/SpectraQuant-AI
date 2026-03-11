"""V3 QA and validation uplift — regression tests.

Covers the four guardrails added in the QA uplift:

1. ``validate_equity_config`` / ``validate_crypto_config`` — fail-fast on
   missing asset-class section in config.
2. ``run_equity_pipeline`` / ``run_crypto_pipeline`` — pipelines call
   validate_config + asset-class validation before touching any stage.
3. ``BacktestEngine.__init__`` — strategy/asset-class mismatch raises
   ``MixedAssetClassRunError`` before any processing starts.
4. ``BacktestEngine._build_signal_agent`` equity path — uses ``strategy_id``
   + ``AgentRegistry`` when ``strategy_id`` is provided (mirrors crypto path).

All tests are self-contained; no network calls, no file-system writes beyond
``tmp_path`` fixtures provided by pytest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100.0 + np.cumsum(rng.standard_normal(n))
    c = np.maximum(c, 1.0)
    return pd.DataFrame(
        {
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": np.ones(n) * 1e6,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
    )


def _equity_cfg(tickers: list[str] | None = None) -> dict:
    return {
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
            "universe": {
                "tickers": tickers or ["INFY.NS", "TCS.NS"],
                "exclude": [],
            },
            "quality_gate": {
                "min_price": 0,
                "min_avg_volume": 0,
                "min_history_days": 0,
            },
            "signals": {"momentum_lookback": 20, "rsi_period": 14},
        },
    }


def _crypto_cfg(symbols: list[str] | None = None) -> dict:
    return {
        "run": {"mode": "normal"},
        "cache": {"root": "data/cache"},
        "qa": {"min_ohlcv_coverage": 1.0},
        "execution": {"mode": "paper"},
        "portfolio": {
            "max_weight": 0.25,
            "max_gross_leverage": 1.0,
            "min_confidence": 0.10,
            "min_signal_threshold": 0.05,
            "target_vol": 0.15,
            "allocator": "equal_weight",
        },
        "crypto": {
            "symbols": symbols or ["BTC", "ETH"],
            "primary_ohlcv_provider": "ccxt",
            "universe_mode": "static",
            "quality_gate": {
                "min_market_cap_usd": 0,
                "min_24h_volume_usd": 0,
                "min_age_days": 0,
                "require_tradable_mapping": True,
            },
            "signals": {"momentum_lookback": 20},
        },
    }


# ===========================================================================
# 1. validate_equity_config / validate_crypto_config
# ===========================================================================


class TestValidateAssetClassConfig:
    """validate_equity_config and validate_crypto_config fail fast on missing key."""

    # ---- equity ----

    def test_validate_equity_config_passes_with_equities_key(self) -> None:
        from spectraquant_v3.core.config import validate_equity_config

        validate_equity_config(_equity_cfg())  # must not raise

    def test_validate_equity_config_raises_on_missing_equities_key(self) -> None:
        from spectraquant_v3.core.config import validate_equity_config
        from spectraquant_v3.core.errors import ConfigValidationError

        cfg_no_equities = {
            k: v for k, v in _equity_cfg().items() if k != "equities"
        }
        with pytest.raises(ConfigValidationError, match="equities"):
            validate_equity_config(cfg_no_equities)

    def test_validate_equity_config_raises_with_crypto_cfg(self) -> None:
        """A crypto config passed to validate_equity_config must fail fast."""
        from spectraquant_v3.core.config import validate_equity_config
        from spectraquant_v3.core.errors import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="equities"):
            validate_equity_config(_crypto_cfg())

    # ---- crypto ----

    def test_validate_crypto_config_passes_with_crypto_key(self) -> None:
        from spectraquant_v3.core.config import validate_crypto_config

        validate_crypto_config(_crypto_cfg())  # must not raise

    def test_validate_crypto_config_raises_on_missing_crypto_key(self) -> None:
        from spectraquant_v3.core.config import validate_crypto_config
        from spectraquant_v3.core.errors import ConfigValidationError

        cfg_no_crypto = {k: v for k, v in _crypto_cfg().items() if k != "crypto"}
        with pytest.raises(ConfigValidationError, match="crypto"):
            validate_crypto_config(cfg_no_crypto)

    def test_validate_crypto_config_raises_with_equity_cfg(self) -> None:
        """An equity config passed to validate_crypto_config must fail fast."""
        from spectraquant_v3.core.config import validate_crypto_config
        from spectraquant_v3.core.errors import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="crypto"):
            validate_crypto_config(_equity_cfg())

    # ---- both helpers are importable from config module ----

    def test_helpers_importable(self) -> None:
        from spectraquant_v3.core.config import (  # noqa: F401
            validate_crypto_config,
            validate_equity_config,
        )


# ===========================================================================
# 2. Pipeline entry-point fail-fast on bad config
# ===========================================================================


class TestPipelineConfigGuards:
    """run_equity_pipeline and run_crypto_pipeline raise ConfigValidationError
    when required top-level or asset-class keys are missing."""

    def test_equity_pipeline_raises_on_missing_portfolio(
        self, tmp_path: Path
    ) -> None:
        from spectraquant_v3.core.errors import ConfigValidationError
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        bad_cfg = {k: v for k, v in _equity_cfg().items() if k != "portfolio"}
        with pytest.raises(ConfigValidationError, match="portfolio"):
            run_equity_pipeline(bad_cfg, dry_run=True, project_root=str(tmp_path))

    def test_equity_pipeline_raises_on_missing_equities_section(
        self, tmp_path: Path
    ) -> None:
        from spectraquant_v3.core.errors import ConfigValidationError
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        bad_cfg = {k: v for k, v in _equity_cfg().items() if k != "equities"}
        with pytest.raises(ConfigValidationError, match="equities"):
            run_equity_pipeline(bad_cfg, dry_run=True, project_root=str(tmp_path))

    def test_equity_pipeline_raises_on_completely_empty_config(
        self, tmp_path: Path
    ) -> None:
        from spectraquant_v3.core.errors import ConfigValidationError
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        with pytest.raises(ConfigValidationError):
            run_equity_pipeline({}, dry_run=True, project_root=str(tmp_path))

    def test_equity_pipeline_raises_when_given_crypto_config(
        self, tmp_path: Path
    ) -> None:
        """Passing a crypto config to run_equity_pipeline must raise immediately."""
        from spectraquant_v3.core.errors import ConfigValidationError
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        with pytest.raises(ConfigValidationError, match="equities"):
            run_equity_pipeline(
                _crypto_cfg(), dry_run=True, project_root=str(tmp_path)
            )

    def test_crypto_pipeline_raises_on_missing_portfolio(
        self, tmp_path: Path
    ) -> None:
        from spectraquant_v3.core.errors import ConfigValidationError
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

        bad_cfg = {k: v for k, v in _crypto_cfg().items() if k != "portfolio"}
        with pytest.raises(ConfigValidationError, match="portfolio"):
            run_crypto_pipeline(bad_cfg, dry_run=True, project_root=str(tmp_path))

    def test_crypto_pipeline_raises_on_missing_crypto_section(
        self, tmp_path: Path
    ) -> None:
        from spectraquant_v3.core.errors import ConfigValidationError
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

        bad_cfg = {k: v for k, v in _crypto_cfg().items() if k != "crypto"}
        with pytest.raises(ConfigValidationError, match="crypto"):
            run_crypto_pipeline(bad_cfg, dry_run=True, project_root=str(tmp_path))

    def test_crypto_pipeline_raises_when_given_equity_config(
        self, tmp_path: Path
    ) -> None:
        """Passing an equity config to run_crypto_pipeline must raise immediately."""
        from spectraquant_v3.core.errors import ConfigValidationError
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

        with pytest.raises(ConfigValidationError, match="crypto"):
            run_crypto_pipeline(
                _equity_cfg(), dry_run=True, project_root=str(tmp_path)
            )

    def test_valid_equity_pipeline_still_succeeds(self, tmp_path: Path) -> None:
        """Guard must not block valid configs from running."""
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        cfg = _equity_cfg(["INFY.NS"])
        result = run_equity_pipeline(
            cfg, dry_run=True, project_root=str(tmp_path)
        )
        assert result["status"] == "success"

    def test_valid_crypto_pipeline_still_succeeds(self, tmp_path: Path) -> None:
        """Guard must not block valid configs from running."""
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

        cfg = _crypto_cfg(["BTC"])
        result = run_crypto_pipeline(
            cfg, dry_run=True, project_root=str(tmp_path)
        )
        assert result["status"] == "success"


# ===========================================================================
# 3. BacktestEngine strategy / asset-class mismatch guard
# ===========================================================================


class TestBacktestEngineMismatchGuard:
    """BacktestEngine must reject a strategy whose asset_class differs from
    the engine's asset_class before any processing starts."""

    def test_equity_strategy_with_crypto_asset_class_raises(self) -> None:
        from spectraquant_v3.backtest.engine import BacktestEngine
        from spectraquant_v3.core.errors import MixedAssetClassRunError

        price_data = {"BTC": _ohlcv(60)}
        with pytest.raises(MixedAssetClassRunError, match="equity_momentum_v1"):
            BacktestEngine(
                cfg=_crypto_cfg(["BTC"]),
                asset_class="crypto",
                price_data=price_data,
                strategy_id="equity_momentum_v1",
                run_id="mismatch_test",
            )

    def test_crypto_strategy_with_equity_asset_class_raises(self) -> None:
        from spectraquant_v3.backtest.engine import BacktestEngine
        from spectraquant_v3.core.errors import MixedAssetClassRunError

        price_data = {"INFY.NS": _ohlcv(60)}
        with pytest.raises(MixedAssetClassRunError, match="crypto_momentum_v1"):
            BacktestEngine(
                cfg=_equity_cfg(["INFY.NS"]),
                asset_class="equity",
                price_data=price_data,
                strategy_id="crypto_momentum_v1",
                run_id="mismatch_test",
            )

    def test_equity_strategy_with_equity_asset_class_does_not_raise(
        self,
    ) -> None:
        from spectraquant_v3.backtest.engine import BacktestEngine

        price_data = {"INFY.NS": _ohlcv(60)}
        engine = BacktestEngine(
            cfg=_equity_cfg(["INFY.NS"]),
            asset_class="equity",
            price_data=price_data,
            strategy_id="equity_momentum_v1",
            run_id="ok_test",
        )
        assert engine._strategy_id == "equity_momentum_v1"

    def test_crypto_strategy_with_crypto_asset_class_does_not_raise(
        self,
    ) -> None:
        from spectraquant_v3.backtest.engine import BacktestEngine

        price_data = {"BTC": _ohlcv(60)}
        engine = BacktestEngine(
            cfg=_crypto_cfg(["BTC"]),
            asset_class="crypto",
            price_data=price_data,
            strategy_id="crypto_momentum_v1",
            run_id="ok_test",
        )
        assert engine._strategy_id == "crypto_momentum_v1"

    def test_no_strategy_id_never_raises_mismatch(self) -> None:
        """When strategy_id is None the mismatch check must not run."""
        from spectraquant_v3.backtest.engine import BacktestEngine

        price_data = {"INFY.NS": _ohlcv(60)}
        engine = BacktestEngine(
            cfg=_equity_cfg(["INFY.NS"]),
            asset_class="equity",
            price_data=price_data,
            strategy_id=None,
            run_id="no_strategy_test",
        )
        assert engine._strategy_id is None


# ===========================================================================
# 4. BacktestEngine equity path uses strategy_id for agent dispatch
# ===========================================================================


class TestBacktestEngineEquityAgentDispatch:
    """BacktestEngine._build_signal_agent must return the correct agent class
    for equity strategies when strategy_id is provided."""

    @pytest.mark.parametrize(
        "strategy_id,expected_class",
        [
            ("equity_breakout_v1", "EquityBreakoutAgent"),
            ("equity_mean_reversion_v1", "EquityMeanReversionAgent"),
            ("equity_volatility_v1", "EquityVolatilityAgent"),
            ("equity_quality_v1", "EquityQualityAgent"),
            ("equity_momentum_v1", "EquityMomentumAgent"),
        ],
    )
    def test_equity_agent_dispatch_by_strategy_id(
        self, strategy_id: str, expected_class: str
    ) -> None:
        from spectraquant_v3.backtest.engine import BacktestEngine

        price_data = {"INFY.NS": _ohlcv(60)}
        engine = BacktestEngine(
            cfg=_equity_cfg(["INFY.NS"]),
            asset_class="equity",
            price_data=price_data,
            strategy_id=strategy_id,
            run_id=f"dispatch_{strategy_id}",
        )
        assert engine._signal_agent.__class__.__name__ == expected_class, (
            f"strategy_id={strategy_id!r}: expected agent class "
            f"{expected_class!r}, got "
            f"{engine._signal_agent.__class__.__name__!r}"
        )

    def test_equity_no_strategy_id_defaults_to_momentum(self) -> None:
        """Without strategy_id the equity engine defaults to EquityMomentumAgent."""
        from spectraquant_v3.backtest.engine import BacktestEngine

        price_data = {"INFY.NS": _ohlcv(60)}
        engine = BacktestEngine(
            cfg=_equity_cfg(["INFY.NS"]),
            asset_class="equity",
            price_data=price_data,
            run_id="default_agent",
        )
        assert engine._signal_agent.__class__.__name__ == "EquityMomentumAgent"

    @pytest.mark.parametrize(
        "strategy_id",
        [
            "equity_breakout_v1",
            "equity_mean_reversion_v1",
            "equity_volatility_v1",
            "equity_quality_v1",
        ],
    )
    def test_equity_backtest_run_produces_signals_with_correct_agent(
        self, strategy_id: str
    ) -> None:
        """Full engine.run() end-to-end with each new equity strategy."""
        from spectraquant_v3.backtest.engine import BacktestEngine
        from spectraquant_v3.backtest.results import BacktestResults

        tickers = ["INFY.NS", "TCS.NS"]
        price_data = {t: _ohlcv(120, seed=i) for i, t in enumerate(tickers)}

        engine = BacktestEngine(
            cfg=_equity_cfg(tickers),
            asset_class="equity",
            price_data=price_data,
            strategy_id=strategy_id,
            rebalance_freq="ME",
            min_in_sample_periods=30,
            run_id=f"e2e_{strategy_id}",
        )
        # Confirm correct agent was wired
        from spectraquant_v3.strategies.agents.registry import AgentRegistry
        from spectraquant_v3.strategies.loader import StrategyLoader

        defn = StrategyLoader.load(strategy_id)
        expected_cls = AgentRegistry.get(defn.agents[0])
        assert isinstance(engine._signal_agent, expected_cls), (
            f"Expected instance of {expected_cls.__name__}, "
            f"got {engine._signal_agent.__class__.__name__}"
        )

        results = engine.run()
        assert isinstance(results, BacktestResults)
        assert results.asset_class == "equity"
        # Steps may be 0 if data is insufficient for the given freq/periods,
        # but the run must not raise.
        assert results.n_steps >= 0


# ===========================================================================
# 5. AgentRegistry.build_from_config callable guard
# ===========================================================================


class TestAgentRegistryBuildFromConfig:
    """build_from_config must only call from_config when it is a real callable.

    A class-level attribute named 'from_config' that is not callable (e.g. a
    plain string or integer set by mistake) must not be invoked; the method
    should fall back to the bare constructor.
    """

    def teardown_method(self, method) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        AgentRegistry.unregister("_test_no_from_config")
        AgentRegistry.unregister("_test_noncallable_from_config")
        AgentRegistry.unregister("_test_callable_from_config")

    def test_agent_without_from_config_uses_constructor(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        class _AgentNoConfig:
            def __init__(self, run_id: str = "default") -> None:
                self.run_id = run_id

        AgentRegistry.register("_test_no_from_config", _AgentNoConfig)
        inst = AgentRegistry.build_from_config(
            "_test_no_from_config", {}, run_id="my_run"
        )
        assert isinstance(inst, _AgentNoConfig)
        assert inst.run_id == "my_run"

    def test_agent_with_noncallable_from_config_falls_back_to_constructor(
        self,
    ) -> None:
        """When from_config is an attribute but not callable, the constructor
        must be used instead of raising TypeError."""
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        class _AgentBadFromConfig:
            from_config = "not_a_callable"  # attribute, not a method

            def __init__(self, run_id: str = "default") -> None:
                self.run_id = run_id

        AgentRegistry.register("_test_noncallable_from_config", _AgentBadFromConfig)
        # Must not raise even though from_config exists but is not callable
        inst = AgentRegistry.build_from_config(
            "_test_noncallable_from_config", {}, run_id="safe_run"
        )
        assert isinstance(inst, _AgentBadFromConfig)
        assert inst.run_id == "safe_run"

    def test_agent_with_callable_from_config_is_used(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        class _AgentGoodFromConfig:
            def __init__(self, run_id: str = "default", source: str = "constructor") -> None:
                self.run_id = run_id
                self.source = source

            @classmethod
            def from_config(cls, cfg: dict, run_id: str = "default") -> "_AgentGoodFromConfig":
                obj = cls.__new__(cls)
                obj.run_id = run_id
                obj.source = "from_config"
                return obj

        AgentRegistry.register("_test_callable_from_config", _AgentGoodFromConfig)
        inst = AgentRegistry.build_from_config(
            "_test_callable_from_config", {}, run_id="cfg_run"
        )
        assert isinstance(inst, _AgentGoodFromConfig)
        assert inst.source == "from_config"
        assert inst.run_id == "cfg_run"

    def test_unknown_agent_name_raises_key_error(self) -> None:
        from spectraquant_v3.strategies.agents.registry import AgentRegistry

        with pytest.raises(KeyError, match="_no_such_agent"):
            AgentRegistry.build_from_config("_no_such_agent", {})
