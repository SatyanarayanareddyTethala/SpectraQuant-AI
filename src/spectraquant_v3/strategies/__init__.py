"""SpectraQuant-AI-V3 Strategy package.

Exports the strategy registry, definition dataclass, and loader.
"""

from spectraquant_v3.strategies.loader import StrategyLoader
from spectraquant_v3.strategies.registry import StrategyRegistry
from spectraquant_v3.strategies.strategy_definition import RiskConfig, StrategyDefinition

__all__ = [
    "StrategyDefinition",
    "RiskConfig",
    "StrategyRegistry",
    "StrategyLoader",
]
