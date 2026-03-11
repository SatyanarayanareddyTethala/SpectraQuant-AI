"""Agent registry for SpectraQuant-AI-V3.

Maps agent name strings (used in :class:`~spectraquant_v3.strategies.strategy_definition.StrategyDefinition`)
to the concrete agent classes that implement them.

Usage::

    from spectraquant_v3.strategies.agents.registry import AgentRegistry

    cls = AgentRegistry.get("crypto_momentum_v1")
    agent_instance = cls(run_id="my_run")

    AgentRegistry.list()  # ['crypto_momentum_v1', 'equity_momentum_v1']
"""

from __future__ import annotations

from typing import Any


class AgentRegistry:
    """Registry mapping agent name → agent class.

    Methods
    -------
    register(name, cls)  Add or replace an agent class.
    get(name)            Return the agent class for *name*.
    list()               Sorted list of registered agent names.
    """

    _agents: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, agent_cls: type) -> None:
        """Register an agent class under *name*.

        Args:
            name:      Unique string identifier (e.g. ``"crypto_momentum_v1"``).
            agent_cls: The agent class (must be callable with at least ``run_id``).
        """
        if not name:
            raise ValueError("Agent name must be a non-empty string.")
        cls._agents[name] = agent_cls

    @classmethod
    def get(cls, name: str) -> type:
        """Return the agent class for *name*.

        Raises:
            KeyError: If no agent with that name has been registered.
        """
        try:
            return cls._agents[name]
        except KeyError:
            registered = sorted(cls._agents)
            raise KeyError(
                f"Agent '{name}' is not registered. "
                f"Registered agents: {registered}"
            ) from None

    @classmethod
    def list(cls) -> list[str]:
        """Return a sorted list of registered agent names."""
        return sorted(cls._agents)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove an agent from the registry (useful in tests)."""
        cls._agents.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Wipe the registry (useful in tests)."""
        cls._agents.clear()

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> Any:
        """Instantiate the agent class for *name* with the given keyword arguments.

        Args:
            name:   Registered agent name.
            **kwargs: Passed through to the agent constructor.

        Returns:
            An agent instance.
        """
        agent_cls = cls.get(name)
        return agent_cls(**kwargs)

    @classmethod
    def build_from_config(
        cls,
        name: str,
        cfg: dict[str, Any],
        run_id: str = "unknown",
    ) -> Any:
        """Instantiate the agent for *name* using ``from_config`` when available.

        If the agent class exposes a ``from_config(cfg, run_id=...)`` factory
        class-method it is preferred over the bare constructor so that agents
        can apply their own config extraction logic.  Falls back to
        ``agent_cls(run_id=run_id)`` for agents without ``from_config``.

        This helper centralises the ``hasattr(cls, "from_config")`` pattern
        that would otherwise be copy-pasted across pipeline and backtest code.

        Args:
            name:   Registered agent name.
            cfg:    Merged pipeline configuration dict.
            run_id: Run identifier forwarded to the agent constructor.

        Returns:
            An agent instance.
        """
        agent_cls = cls.get(name)
        if hasattr(agent_cls, "from_config") and callable(agent_cls.from_config):
            return agent_cls.from_config(cfg, run_id=run_id)
        return agent_cls(run_id=run_id)


# ---------------------------------------------------------------------------
# Built-in agent registrations
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    from spectraquant_v3.crypto.signals.cross_sectional_momentum import (
        CryptoCrossSectionalMomentumAgent,
    )
    from spectraquant_v3.crypto.signals.hybrid import CryptoMomentumNewsHybridAgent
    from spectraquant_v3.crypto.signals.momentum import CryptoMomentumAgent
    from spectraquant_v3.equities.signals.breakout import EquityBreakoutAgent
    from spectraquant_v3.equities.signals.hybrid import EquityMomentumNewsHybridAgent
    from spectraquant_v3.equities.signals.mean_reversion import EquityMeanReversionAgent
    from spectraquant_v3.equities.signals.momentum import EquityMomentumAgent
    from spectraquant_v3.equities.signals.news_sentiment import EquityNewsSentimentAgent
    from spectraquant_v3.equities.signals.quality import EquityQualityAgent
    from spectraquant_v3.equities.signals.volatility import EquityVolatilityAgent

    AgentRegistry.register("crypto_momentum_v1", CryptoMomentumAgent)
    AgentRegistry.register("crypto_cross_sectional_momentum_v1", CryptoCrossSectionalMomentumAgent)
    AgentRegistry.register("crypto_momentum_news_hybrid_v1", CryptoMomentumNewsHybridAgent)
    AgentRegistry.register("equity_momentum_v1", EquityMomentumAgent)
    AgentRegistry.register("equity_breakout_v1", EquityBreakoutAgent)
    AgentRegistry.register("equity_mean_reversion_v1", EquityMeanReversionAgent)
    AgentRegistry.register("equity_volatility_v1", EquityVolatilityAgent)
    AgentRegistry.register("equity_quality_v1", EquityQualityAgent)
    AgentRegistry.register("equity_news_sentiment_v1", EquityNewsSentimentAgent)
    AgentRegistry.register("equity_momentum_news_hybrid_v1", EquityMomentumNewsHybridAgent)


_register_builtins()
