"""Backtest results dataclass for SpectraQuant-AI-V3.

:class:`BacktestResults` is the structured output of a
:class:`~spectraquant_v3.backtest.engine.BacktestEngine` run.  It holds
all per-rebalance snapshots, aggregated performance metrics, and provides
JSON serialisation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Per-rebalance snapshot
# ---------------------------------------------------------------------------


@dataclass
class RebalanceSnapshot:
    """Portfolio snapshot produced at one rebalance step.

    Attributes:
        date:           ISO-8601 date string of the rebalance step.
        universe:       Canonical symbols eligible at this step.
        signals_ok:     Count of symbols with ``OK`` signal status.
        signals_nosig:  Count of symbols with ``NO_SIGNAL`` status.
        policy_passed:  Count of symbols that passed the meta-policy.
        policy_blocked: Count of symbols blocked by the meta-policy.
        allocations:    Dict mapping canonical symbol → target weight.
        portfolio_value: Normalised portfolio value at end of this step
                         (starts at 1.0).
        step_return:    Net return over this rebalance period (decimal).
        gross_return:   Gross return before transaction costs (decimal).
        net_return:     Net return after transaction costs (decimal).
        turnover:       One-way turnover at rebalance (sum(|w_t - w_{t-1}|)).
        positions_count: Number of active non-zero positions.
        exposure:       Gross portfolio exposure (sum(|weights|)).
    """

    date: str
    universe: list[str] = field(default_factory=list)
    signals_ok: int = 0
    signals_nosig: int = 0
    policy_passed: int = 0
    policy_blocked: int = 0
    allocations: dict[str, float] = field(default_factory=dict)
    portfolio_value: float = 1.0
    step_return: float = 0.0
    gross_return: float = 0.0
    net_return: float = 0.0
    turnover: float = 0.0
    positions_count: int = 0
    exposure: float = 0.0
    blocked_reasons: dict[str, int] = field(default_factory=dict)
    no_signal_reasons: dict[str, int] = field(default_factory=dict)
    selected_symbols: list[str] = field(default_factory=list)
    composite_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "universe": self.universe,
            "signals_ok": self.signals_ok,
            "signals_nosig": self.signals_nosig,
            "policy_passed": self.policy_passed,
            "policy_blocked": self.policy_blocked,
            "allocations": self.allocations,
            "portfolio_value": self.portfolio_value,
            "step_return": self.step_return,
            "gross_return": self.gross_return,
            "net_return": self.net_return,
            "turnover": self.turnover,
            "positions_count": self.positions_count,
            "exposure": self.exposure,
            "blocked_reasons": self.blocked_reasons,
            "no_signal_reasons": self.no_signal_reasons,
            "selected_symbols": self.selected_symbols,
            "composite_scores": self.composite_scores,
        }


# ---------------------------------------------------------------------------
# Aggregated results
# ---------------------------------------------------------------------------


@dataclass
class BacktestResults:
    """Aggregated results of a completed backtest run.

    Attributes:
        run_id:               Identifier for the backtest run.
        asset_class:          ``"crypto"`` or ``"equity"``.
        start_date:           First rebalance date (ISO-8601 string).
        end_date:             Last rebalance date (ISO-8601 string).
        n_steps:              Total number of rebalance steps executed.
        symbols:              All symbols that appeared in the universe.
        total_return:         Cumulative portfolio return over the full period.
        annualised_return:    CAGR (annualised).
        annualised_volatility: Annualised portfolio volatility.
        sharpe_ratio:         Annualised Sharpe ratio (risk-free rate = 0).
        max_drawdown:         Maximum peak-to-trough drawdown (negative value).
        calmar_ratio:         |annualised_return / max_drawdown|, or 0 if max_drawdown=0.
        win_rate:             Fraction of rebalance steps with positive return.
        turnover:             Average one-way turnover per rebalance step.
        avg_positions:        Average number of active positions.
        exposure:             Average gross exposure per rebalance step.
        snapshots:            Ordered list of per-step :class:`RebalanceSnapshot`.
        generated_at:         ISO-8601 timestamp of when results were created.
        extra:                Optional metadata dict.
    """

    run_id: str
    asset_class: str
    start_date: str
    end_date: str
    n_steps: int
    symbols: list[str] = field(default_factory=list)
    total_return: float = 0.0
    annualised_return: float = 0.0
    annualised_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    turnover: float = 0.0
    avg_positions: float = 0.0
    exposure: float = 0.0
    blocked_reasons: dict[str, int] = field(default_factory=dict)
    no_signal_reasons: dict[str, int] = field(default_factory=dict)
    selected_symbols: list[str] = field(default_factory=list)
    composite_scores: dict[str, float] = field(default_factory=dict)
    snapshots: list[RebalanceSnapshot] = field(default_factory=list)
    generated_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "run_id": self.run_id,
            "asset_class": self.asset_class,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_steps": self.n_steps,
            "symbols": self.symbols,
            "total_return": round(self.total_return, 6),
            "annualised_return": round(self.annualised_return, 6),
            "annualised_volatility": round(self.annualised_volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "win_rate": round(self.win_rate, 4),
            "turnover": round(self.turnover, 6),
            "avg_positions": round(self.avg_positions, 4),
            "exposure": round(self.exposure, 6),
            "generated_at": self.generated_at,
            "extra": self.extra,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

    def write(self, output_dir: str | Path) -> Path:
        """Write results to ``backtest_results_<run_id>.json``.

        Args:
            output_dir: Directory to write the file.

        Returns:
            Path of the written file.
        """
        path = Path(output_dir) / f"backtest_results_{self.run_id}.json"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def summary(self) -> str:
        """Return a human-readable one-paragraph summary."""
        return (
            f"Backtest {self.run_id}  [{self.start_date} → {self.end_date}]  "
            f"asset_class={self.asset_class}\n"
            f"  steps={self.n_steps}  symbols={len(self.symbols)}\n"
            f"  total_return={self.total_return:+.2%}  "
            f"annualised={self.annualised_return:+.2%}\n"
            f"  volatility={self.annualised_volatility:.2%}  "
            f"sharpe={self.sharpe_ratio:.2f}  "
            f"max_drawdown={self.max_drawdown:.2%}\n"
            f"  calmar={self.calmar_ratio:.2f}  win_rate={self.win_rate:.1%}\n"
            f"  turnover={self.turnover:.4f}  avg_positions={self.avg_positions:.2f}  "
            f"exposure={self.exposure:.4f}"
        )
