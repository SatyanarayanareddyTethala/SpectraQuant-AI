"""Meta-policy skeleton for SpectraQuant-AI-V3.

The meta-policy sits between signal aggregation and allocation.  It:
1. Aggregates signals from multiple agents into a composite score.
2. Applies portfolio-level risk filters (e.g. block symbols below min confidence).
3. Returns a filtered list of :class:`~spectraquant_v3.core.schema.SignalRow` objects
   that the allocator can consume.

This is a skeleton implementation – the composite scoring logic is intentionally
simple (equal-weight average of agent scores) and can be replaced without
changing the allocator interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spectraquant_v3.core.enums import SignalStatus
from spectraquant_v3.core.schema import SignalRow


@dataclass
class MetaPolicyConfig:
    """Configuration for the meta-policy filter."""

    min_confidence: float = 0.10
    """Minimum absolute confidence required to pass the policy."""

    min_signal_threshold: float = 0.05
    """Minimum |score| to consider a signal actionable."""

    block_error_signals: bool = True
    """When True, ERROR-status signals are always blocked."""

    agents_required: list[str] = field(default_factory=list)
    """Require signal from all listed agent IDs before passing a symbol.
    Empty list = no requirement."""

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MetaPolicyConfig":
        """Build from merged pipeline config."""
        portfolio_cfg = cfg.get("portfolio", {})
        return cls(
            min_confidence=float(portfolio_cfg.get("min_confidence", 0.10)),
            min_signal_threshold=float(
                portfolio_cfg.get("min_signal_threshold", 0.05)
            ),
        )


@dataclass
class PolicyDecision:
    """Per-symbol meta-policy decision."""

    canonical_symbol: str
    asset_class: str
    composite_score: float
    composite_confidence: float
    passed: bool
    reason: str
    contributing_agents: list[str] = field(default_factory=list)


class MetaPolicy:
    """Aggregates multi-agent signals and filters them for the allocator.

    Args:
        config: :class:`MetaPolicyConfig` controlling filter thresholds.
    """

    def __init__(self, config: MetaPolicyConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MetaPolicy":
        """Build from merged pipeline config."""
        return cls(config=MetaPolicyConfig.from_config(cfg))

    def run(self, signals: list[SignalRow]) -> list[PolicyDecision]:
        """Apply the meta-policy to a list of signals.

        Groups signals by symbol, computes a composite score, and decides
        whether each symbol passes the policy filter.

        Args:
            signals: All :class:`~spectraquant_v3.core.schema.SignalRow` objects
                     from one or more signal agents.

        Returns:
            List of :class:`PolicyDecision` objects, one per unique symbol.
        """
        # Group by symbol
        by_symbol: dict[str, list[SignalRow]] = {}
        for row in signals:
            by_symbol.setdefault(row.canonical_symbol, []).append(row)

        decisions: list[PolicyDecision] = []
        for sym, rows in sorted(by_symbol.items()):
            decisions.append(self._decide(sym, rows))
        return decisions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decide(self, symbol: str, rows: list[SignalRow]) -> PolicyDecision:
        asset_class = rows[0].asset_class if rows else ""

        # Check required agents
        if self.config.agents_required:
            present = {r.agent_id for r in rows}
            missing = set(self.config.agents_required) - present
            if missing:
                return PolicyDecision(
                    canonical_symbol=symbol,
                    asset_class=asset_class,
                    composite_score=0.0,
                    composite_confidence=0.0,
                    passed=False,
                    reason=f"missing_required_agents={sorted(missing)}",
                )

        # Separate ERROR rows from NO_SIGNAL rows so block_error_signals
        # can be applied independently of the NO_SIGNAL filter.
        has_error = any(r.status == SignalStatus.ERROR.value for r in rows)
        if self.config.block_error_signals and has_error:
            return PolicyDecision(
                canonical_symbol=symbol,
                asset_class=asset_class,
                composite_score=0.0,
                composite_confidence=0.0,
                passed=False,
                reason="blocked_due_to_error_signal",
            )

        # Keep only OK rows for scoring; NO_SIGNAL and ERROR rows are excluded.
        ok_rows = [r for r in rows if r.status == SignalStatus.OK.value]

        if not ok_rows:
            return PolicyDecision(
                canonical_symbol=symbol,
                asset_class=asset_class,
                composite_score=0.0,
                composite_confidence=0.0,
                passed=False,
                reason="all_no_signal",
            )

        # Equal-weight composite score
        composite_score = sum(r.signal_score for r in ok_rows) / len(ok_rows)
        composite_confidence = sum(r.confidence for r in ok_rows) / len(ok_rows)
        contributing = [r.agent_id for r in ok_rows]

        # Apply filters
        if composite_confidence < self.config.min_confidence:
            return PolicyDecision(
                canonical_symbol=symbol,
                asset_class=asset_class,
                composite_score=composite_score,
                composite_confidence=composite_confidence,
                passed=False,
                reason=(
                    f"confidence={composite_confidence:.3f} "
                    f"< min={self.config.min_confidence}"
                ),
                contributing_agents=contributing,
            )

        if abs(composite_score) < self.config.min_signal_threshold:
            return PolicyDecision(
                canonical_symbol=symbol,
                asset_class=asset_class,
                composite_score=composite_score,
                composite_confidence=composite_confidence,
                passed=False,
                reason=(
                    f"|score|={abs(composite_score):.3f} "
                    f"< threshold={self.config.min_signal_threshold}"
                ),
                contributing_agents=contributing,
            )

        return PolicyDecision(
            canonical_symbol=symbol,
            asset_class=asset_class,
            composite_score=composite_score,
            composite_confidence=composite_confidence,
            passed=True,
            reason="passed_all_filters",
            contributing_agents=contributing,
        )
