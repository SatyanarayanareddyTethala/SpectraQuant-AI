"""Strategy loader for SpectraQuant-AI-V3.

The :class:`StrategyLoader` resolves a strategy ID into a fully validated
:class:`~spectraquant_v3.strategies.strategy_definition.StrategyDefinition`
and verifies that all referenced agents, policies, and allocators are
present in their respective registries.

Usage::

    from spectraquant_v3.strategies.loader import StrategyLoader

    strategy = StrategyLoader.load("crypto_momentum_v1")
    print(strategy.agents)       # ['crypto_momentum_v1']
    print(strategy.policy)       # 'confidence_filter_v1'
    print(strategy.allocator)    # 'vol_target_v1'

After loading, the caller can look up the concrete classes via the
individual registries::

    from spectraquant_v3.strategies.agents.registry import AgentRegistry
    from spectraquant_v3.strategies.policies.registry import PolicyRegistry
    from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry

    agent_cls   = AgentRegistry.get(strategy.agents[0])
    policy_cls  = PolicyRegistry.get(strategy.policy)
    alloc_cls   = AllocatorRegistry.get(strategy.allocator)
"""

from __future__ import annotations

from typing import Any

from spectraquant_v3.strategies.agents.registry import AgentRegistry
from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry
from spectraquant_v3.strategies.policies.registry import PolicyRegistry
from spectraquant_v3.strategies.registry import StrategyRegistry
from spectraquant_v3.strategies.strategy_definition import StrategyDefinition


class StrategyLoader:
    """Resolve and validate a strategy by ID.

    This class is stateless; all methods are class-methods.
    """

    @classmethod
    def load(cls, strategy_id: str) -> StrategyDefinition:
        """Load and validate a strategy definition.

        Steps performed:

        1. Look up *strategy_id* in :class:`~spectraquant_v3.strategies.registry.StrategyRegistry`.
        2. Assert the strategy is ``enabled``.
        3. Validate the definition (``defn.validate()``).
        4. Assert every agent name is registered in :class:`~spectraquant_v3.strategies.agents.registry.AgentRegistry`.
        5. Assert the policy name is registered in :class:`~spectraquant_v3.strategies.policies.registry.PolicyRegistry`.
        6. Assert the allocator name is registered in :class:`~spectraquant_v3.strategies.allocators.registry.AllocatorRegistry`.

        Args:
            strategy_id: The strategy to load (must be registered).

        Returns:
            A validated :class:`StrategyDefinition`.

        Raises:
            KeyError:   If the strategy is not registered.
            ValueError: If the strategy is disabled or any component is missing.
        """
        defn = StrategyRegistry.get(strategy_id)

        if not defn.enabled:
            raise ValueError(
                f"Strategy '{strategy_id}' is disabled (enabled=False). "
                "Re-enable it in the registry or pass a different strategy."
            )

        # Re-validate structure
        defn.validate()

        # Validate agents
        missing_agents = [a for a in defn.agents if a not in AgentRegistry.list()]
        if missing_agents:
            raise ValueError(
                f"Strategy '{strategy_id}' references unregistered agent(s): "
                f"{missing_agents}. "
                f"Registered agents: {AgentRegistry.list()}"
            )

        # Validate policy
        if defn.policy not in PolicyRegistry.list():
            raise ValueError(
                f"Strategy '{strategy_id}' references unregistered policy "
                f"'{defn.policy}'. "
                f"Registered policies: {PolicyRegistry.list()}"
            )

        # Validate allocator
        if defn.allocator not in AllocatorRegistry.list():
            raise ValueError(
                f"Strategy '{strategy_id}' references unregistered allocator "
                f"'{defn.allocator}'. "
                f"Registered allocators: {AllocatorRegistry.list()}"
            )

        return defn

    @classmethod
    def build_pipeline_config(
        cls,
        strategy_id: str,
        base_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge strategy risk limits into a pipeline config dict.

        This allows :func:`~spectraquant_v3.pipeline.crypto_pipeline.run_crypto_pipeline`
        and :func:`~spectraquant_v3.pipeline.equity_pipeline.run_equity_pipeline`
        to consume the strategy's :class:`~spectraquant_v3.strategies.strategy_definition.RiskConfig`
        without requiring a pipeline redesign.

        The strategy's risk limits are written into the ``portfolio`` key of
        the returned config dict, overriding any values that were present in
        *base_cfg*.

        Four private keys are also injected so that pipelines can consume
        strategy metadata without re-querying the registry:

        - ``_strategy_id``    – the resolved strategy identifier.
        - ``_asset_class``    – ``"crypto"`` or ``"equity"``.
        - ``_agents``         – list of validated agent names from the registry.
        - ``_rebalance_freq`` – pandas offset alias for rebalance cadence.

        Args:
            strategy_id: Strategy to load.
            base_cfg:    Existing merged pipeline config (not modified in-place).

        Returns:
            A new config dict with the strategy's risk limits applied and
            ``_strategy_id``, ``_asset_class``, ``_agents``, and
            ``_rebalance_freq`` injected.
        """
        defn = cls.load(strategy_id)
        cfg = dict(base_cfg)
        portfolio = dict(cfg.get("portfolio", {}))
        risk = defn.risk_config

        portfolio["max_weight"] = risk.max_weight
        portfolio["max_gross_leverage"] = risk.max_gross_leverage
        portfolio["target_vol"] = risk.target_vol
        portfolio["min_confidence"] = risk.min_confidence
        portfolio["min_signal_threshold"] = risk.min_signal_threshold

        # Write allocator mode so AllocatorConfig.from_config picks it up.
        # Strategy allocators that are volatility-targeted map to pipeline mode
        # "vol_target" for backwards-compatible pipeline execution.
        # "equal_weight_v1" is treated as an alias for "equal_weight".
        _ALLOCATOR_MODE_MAP: dict[str, str] = {
            "vol_target_v1": "vol_target",
            "rank_vol_target_allocator": "vol_target",
            "equal_weight_v1": "equal_weight",
        }
        portfolio["allocator"] = _ALLOCATOR_MODE_MAP.get(defn.allocator, defn.allocator)

        cfg["portfolio"] = portfolio
        cfg["_strategy_id"] = defn.strategy_id
        cfg["_asset_class"] = defn.asset_class
        cfg["_agents"] = list(defn.agents)
        cfg["_rebalance_freq"] = defn.rebalance_freq
        return cfg
