"""Strategy registry for SpectraQuant-AI-V3.

The :class:`StrategyRegistry` is the central catalog of all known strategies.
Strategies are registered at module import time (or programmatically).

Usage::

    from spectraquant_v3.strategies.registry import StrategyRegistry
    from spectraquant_v3.strategies.strategy_definition import StrategyDefinition

    # List registered strategies
    for sid in StrategyRegistry.list():
        print(sid)

    # Look up a single strategy
    defn = StrategyRegistry.get("crypto_momentum_v1")

    # Register a custom strategy
    StrategyRegistry.register(StrategyDefinition(
        strategy_id="my_alpha_v1",
        asset_class="crypto",
        agents=["momentum"],
        policy="confidence_filter_v1",
        allocator="rank_vol_target_allocator",
    ))
"""

from __future__ import annotations

from spectraquant_v3.strategies.strategy_definition import RiskConfig, StrategyDefinition


class StrategyRegistry:
    """Central registry of :class:`~spectraquant_v3.strategies.strategy_definition.StrategyDefinition` objects.

    All methods are class-methods; the registry is a process-wide singleton backed
    by a class-level ``dict``.

    Methods
    -------
    register(defn)
        Add or replace a strategy definition.
    get(strategy_id)
        Return the definition for *strategy_id*, raising :class:`KeyError` if absent.
    list()
        Return a sorted list of registered strategy IDs.
    list_all()
        Return all :class:`StrategyDefinition` objects as a list.
    unregister(strategy_id)
        Remove a strategy from the registry (mainly useful in tests).
    clear()
        Wipe the entire registry (mainly useful in tests).
    """

    _strategies: dict[str, StrategyDefinition] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, defn: StrategyDefinition) -> None:
        """Register (or replace) a strategy definition.

        Args:
            defn: A :class:`StrategyDefinition` instance.  ``defn.validate()``
                  is called before registration.

        Raises:
            ValueError: If the definition fails validation.
        """
        defn.validate()
        cls._strategies[defn.strategy_id] = defn

    @classmethod
    def unregister(cls, strategy_id: str) -> None:
        """Remove a strategy from the registry.

        Silently ignores unknown ``strategy_id`` values.
        """
        cls._strategies.pop(strategy_id, None)

    @classmethod
    def clear(cls) -> None:
        """Wipe the entire registry.  Use only in tests."""
        cls._strategies.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, strategy_id: str) -> StrategyDefinition:
        """Return the strategy definition for *strategy_id*.

        Raises:
            KeyError: If no strategy with that ID has been registered.
        """
        try:
            return cls._strategies[strategy_id]
        except KeyError:
            registered = sorted(cls._strategies)
            raise KeyError(
                f"Strategy '{strategy_id}' is not registered. "
                f"Registered strategies: {registered}"
            ) from None

    @classmethod
    def list(cls) -> list[str]:
        """Return a sorted list of registered strategy IDs."""
        return sorted(cls._strategies)

    @classmethod
    def list_all(cls) -> list[StrategyDefinition]:
        """Return all registered strategy definitions."""
        return [cls._strategies[sid] for sid in sorted(cls._strategies)]

    @classmethod
    def list_enabled(cls) -> list[StrategyDefinition]:
        """Return strategy definitions where ``enabled=True``."""
        return [d for d in cls.list_all() if d.enabled]


# ---------------------------------------------------------------------------
# Built-in strategy definitions (registered at import time)
# ---------------------------------------------------------------------------

_BUILTIN_STRATEGIES: list[StrategyDefinition] = [
    StrategyDefinition(
        strategy_id="crypto_momentum_v1",
        asset_class="crypto",
        universe_profile="static",
        feature_profile="default",
        agents=["crypto_momentum_v1"],
        policy="confidence_filter_v1",
        allocator="rank_vol_target_allocator",
        rebalance_freq="W",
        tags=["crypto", "momentum", "baseline"],
    ),
    StrategyDefinition(
        strategy_id="crypto_momentum_v2",
        asset_class="crypto",
        universe_profile="dynamic_topN",
        feature_profile="default",
        agents=["crypto_momentum_v1"],
        policy="confidence_filter_v1",
        allocator="vol_target_v1",
        rebalance_freq="W",
        risk_config=RiskConfig(max_weight=0.25, target_vol=0.15),
        tags=["crypto", "momentum", "vol_target"],
    ),
    StrategyDefinition(
        strategy_id="crypto_momentum_news_hybrid_v1",
        asset_class="crypto",
        universe_profile="dynamic_topN",
        feature_profile="crypto_with_news",
        agents=["crypto_momentum_news_hybrid_v1"],
        policy="confidence_filter_v1",
        allocator="rank_vol_target_allocator",
        rebalance_freq="D",
        tags=["crypto", "momentum", "news", "hybrid"],
    ),
    StrategyDefinition(
        strategy_id="crypto_cross_sectional_momentum_v1",
        asset_class="crypto",
        universe_profile="dynamic_topN",
        feature_profile="default",
        agents=["crypto_cross_sectional_momentum_v1"],
        policy="confidence_filter_v1",
        allocator="rank_vol_target_allocator",
        rebalance_freq="W",
        tags=["crypto", "momentum", "cross-sectional"],
    ),
    StrategyDefinition(
        strategy_id="equity_momentum_v1",
        asset_class="equity",
        universe_profile="static",
        feature_profile="default",
        agents=["equity_momentum_v1"],
        policy="confidence_filter_v1",
        allocator="equal_weight",
        rebalance_freq="ME",
        tags=["equity", "momentum", "baseline"],
    ),
    StrategyDefinition(
        strategy_id="equity_breakout_v1",
        asset_class="equity",
        universe_profile="static",
        feature_profile="default",
        agents=["equity_breakout_v1"],
        policy="confidence_filter_v1",
        allocator="equal_weight",
        rebalance_freq="ME",
        tags=["equity", "breakout"],
    ),
    StrategyDefinition(
        strategy_id="equity_mean_reversion_v1",
        asset_class="equity",
        universe_profile="static",
        feature_profile="default",
        agents=["equity_mean_reversion_v1"],
        policy="confidence_filter_v1",
        allocator="equal_weight",
        rebalance_freq="W",
        tags=["equity", "mean_reversion"],
    ),
    StrategyDefinition(
        strategy_id="equity_volatility_v1",
        asset_class="equity",
        universe_profile="static",
        feature_profile="default",
        agents=["equity_volatility_v1"],
        policy="confidence_filter_v1",
        allocator="equal_weight",
        rebalance_freq="ME",
        tags=["equity", "volatility", "risk_off"],
    ),
    StrategyDefinition(
        strategy_id="equity_quality_v1",
        asset_class="equity",
        universe_profile="static",
        feature_profile="default",
        agents=["equity_quality_v1"],
        policy="confidence_filter_v1",
        allocator="equal_weight",
        rebalance_freq="ME",
        tags=["equity", "quality", "filter"],
    ),
]

for _defn in _BUILTIN_STRATEGIES:
    StrategyRegistry.register(_defn)
