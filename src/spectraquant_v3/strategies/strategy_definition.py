"""Strategy definition dataclass for SpectraQuant-AI-V3.

A :class:`StrategyDefinition` describes a complete research/trading strategy
in a declarative way so that strategies can be composed from registered
components rather than hardcoded pipelines.

Each definition specifies:
- ``asset_class``       – "crypto" or "equity"
- ``universe_profile``  – how to build the investment universe
- ``feature_profile``   – which feature engine to use
- ``agents``            – list of registered agent names
- ``policy``            – registered meta-policy name
- ``allocator``         – registered allocator name
- ``rebalance_freq``    – pandas offset alias (e.g. "W", "ME", "D")
- ``risk_config``       – per-strategy risk limits
- ``tags``              – arbitrary labels for filtering / discovery
- ``enabled``           – soft-disable without deleting the definition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskConfig:
    """Per-strategy risk limits passed to the allocator."""

    max_weight: float = 0.20
    """Maximum weight per symbol (fraction of portfolio)."""

    max_gross_leverage: float = 1.0
    """Maximum sum of absolute weights."""

    target_vol: float = 0.15
    """Target annualised portfolio volatility (used by vol_target allocator)."""

    min_confidence: float = 0.10
    """Minimum signal confidence required to pass the meta-policy."""

    min_signal_threshold: float = 0.05
    """Minimum |score| to consider a signal actionable."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RiskConfig":
        """Build from a plain dict (e.g. loaded from YAML)."""
        return cls(
            max_weight=float(d.get("max_weight", 0.20)),
            max_gross_leverage=float(d.get("max_gross_leverage", 1.0)),
            target_vol=float(d.get("target_vol", 0.15)),
            min_confidence=float(d.get("min_confidence", 0.10)),
            min_signal_threshold=float(d.get("min_signal_threshold", 0.05)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_weight": self.max_weight,
            "max_gross_leverage": self.max_gross_leverage,
            "target_vol": self.target_vol,
            "min_confidence": self.min_confidence,
            "min_signal_threshold": self.min_signal_threshold,
        }


@dataclass
class StrategyDefinition:
    """Declarative description of a complete research / trading strategy.

    Instances are registered in :class:`~spectraquant_v3.strategies.registry.StrategyRegistry`
    and resolved by :class:`~spectraquant_v3.strategies.loader.StrategyLoader`.

    Args:
        strategy_id:      Unique identifier (e.g. ``"crypto_momentum_v1"``).
        asset_class:      ``"crypto"`` or ``"equity"``.
        universe_profile: Universe selection mode (e.g. ``"static"``, ``"dynamic_topN"``).
        feature_profile:  Feature engine label (e.g. ``"default"``).
        agents:           Ordered list of registered agent names.
        policy:           Registered meta-policy name.
        allocator:        Registered allocator name.
        rebalance_freq:   Pandas offset alias for rebalance cadence.
        risk_config:      :class:`RiskConfig` controlling risk limits.
        tags:             Arbitrary string labels for filtering.
        enabled:          When ``False`` the strategy is skipped in bulk runs.
    """

    strategy_id: str
    asset_class: str  # "crypto" | "equity"
    universe_profile: str = "static"
    feature_profile: str = "default"
    agents: list[str] = field(default_factory=list)
    policy: str = "confidence_filter_v1"
    allocator: str = "equal_weight"
    rebalance_freq: str = "W"
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            "strategy_id": self.strategy_id,
            "asset_class": self.asset_class,
            "universe_profile": self.universe_profile,
            "feature_profile": self.feature_profile,
            "agents": list(self.agents),
            "policy": self.policy,
            "allocator": self.allocator,
            "rebalance_freq": self.rebalance_freq,
            "risk_config": self.risk_config.to_dict(),
            "tags": list(self.tags),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyDefinition":
        """Deserialise from a plain dict (e.g. loaded from YAML or JSON)."""
        risk_raw = d.get("risk_config", {})
        return cls(
            strategy_id=d["strategy_id"],
            asset_class=d["asset_class"],
            universe_profile=d.get("universe_profile", "static"),
            feature_profile=d.get("feature_profile", "default"),
            agents=list(d.get("agents", [])),
            policy=d.get("policy", "confidence_filter_v1"),
            allocator=d.get("allocator", "equal_weight"),
            rebalance_freq=d.get("rebalance_freq", "W"),
            risk_config=RiskConfig.from_dict(risk_raw),
            tags=list(d.get("tags", [])),
            enabled=bool(d.get("enabled", True)),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ValueError` if the definition is incomplete or inconsistent."""
        if not self.strategy_id:
            raise ValueError("strategy_id must be a non-empty string.")
        if self.asset_class not in ("crypto", "equity"):
            raise ValueError(
                f"asset_class must be 'crypto' or 'equity', got {self.asset_class!r}."
            )
        if not self.agents:
            raise ValueError(
                f"Strategy '{self.strategy_id}' must declare at least one agent."
            )
        if not self.policy:
            raise ValueError(
                f"Strategy '{self.strategy_id}' must declare a policy."
            )
        if not self.allocator:
            raise ValueError(
                f"Strategy '{self.strategy_id}' must declare an allocator."
            )
        if self.risk_config.max_weight <= 0 or self.risk_config.max_weight > 1:
            raise ValueError(
                f"risk_config.max_weight must be in (0, 1], "
                f"got {self.risk_config.max_weight}."
            )
