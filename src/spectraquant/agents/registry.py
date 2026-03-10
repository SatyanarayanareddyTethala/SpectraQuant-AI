"""Agent registry that discovers and manages trading agents."""
from __future__ import annotations

import importlib
import logging
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AgentSignal:
    """Standardised output schema for every trading agent."""

    symbol: str
    score: float  # [-1, 1]: negative = short, positive = long
    confidence: float  # [0, 1]
    horizon: str  # e.g. "1d", "4h"
    rationale_tags: list[str] = field(default_factory=list)
    asof_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseAgent(ABC):
    """Abstract base class that every trading agent must implement."""

    @abstractmethod
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[AgentSignal]:
        """Return a list of :class:`AgentSignal` for the given market data."""


class AgentRegistry:
    """Discovers, registers, and orchestrates trading agents."""

    def __init__(self) -> None:
        self._agents: dict[str, type[BaseAgent]] = {}

    def register(self, name: str, agent_cls: type[BaseAgent]) -> None:
        """Register *agent_cls* under *name*."""
        if not (isinstance(agent_cls, type) and issubclass(agent_cls, BaseAgent)):
            raise TypeError(f"{agent_cls!r} is not a BaseAgent subclass")
        self._agents[name] = agent_cls
        logger.info("Registered agent %s → %s", name, agent_cls.__name__)

    def get(self, name: str) -> type[BaseAgent]:
        """Return the agent class registered under *name*."""
        try:
            return self._agents[name]
        except KeyError:
            raise KeyError(f"No agent registered under '{name}'") from None

    def list_agents(self) -> list[str]:
        """Return sorted list of registered agent names."""
        return sorted(self._agents)

    def run_all(
        self,
        market_data: pd.DataFrame,
        **kwargs: Any,
    ) -> dict[str, list[AgentSignal]]:
        """Instantiate every registered agent and collect their signals."""
        results: dict[str, list[AgentSignal]] = {}
        for name, cls in self._agents.items():
            try:
                agent = cls()
                signals = agent.generate_signals(market_data, **kwargs)
                results[name] = signals
                logger.debug(
                    "Agent %s produced %d signal(s)", name, len(signals),
                )
            except Exception:
                logger.exception("Agent %s raised an error", name)
                results[name] = []
        return results


def _auto_register(registry: AgentRegistry) -> None:
    """Import every module in the ``agents`` sub-package and register agents."""
    package = importlib.import_module("spectraquant.agents.agents")
    for _importer, modname, _ispkg in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"spectraquant.agents.agents.{modname}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseAgent)
                and attr is not BaseAgent
            ):
                registry_name = modname
                registry.register(registry_name, attr)


default_registry = AgentRegistry()
_auto_register(default_registry)

__all__ = ["AgentSignal", "BaseAgent", "AgentRegistry", "default_registry"]
