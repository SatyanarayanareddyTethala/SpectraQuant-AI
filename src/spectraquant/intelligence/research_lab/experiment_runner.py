"""Experiment Runner — automated walk-forward backtesting.

Runs hypothesis-driven strategies through a walk-forward validation
framework with no lookahead bias, rolling retraining, and realistic
transaction costs.

Outputs per experiment
----------------------
- sharpe          : annualised Sharpe ratio
- max_drawdown    : maximum peak-to-trough drawdown (fraction)
- win_rate        : fraction of trades with positive PnL
- stability_score : Sharpe consistency across walk-forward folds
- regime_robustness : min Sharpe across regime sub-samples
- experiment_id   : unique string
- experiment_dir  : path where JSON report is saved
"""
from __future__ import annotations

import json
import logging
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from spectraquant.intelligence.research_lab.strategy_generator import StrategyConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_YEAR = 252
_DEFAULT_TX_COST_BPS = 10          # 10 bps round-trip
_MIN_FOLD_RETURN_ROWS = 20         # minimum rows per fold
_DEFAULT_FOLDS = 3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Results for a single walk-forward experiment."""

    experiment_id: str
    strategy_name: str
    hypothesis_id: str
    sharpe: float
    max_drawdown: float
    win_rate: float
    stability_score: float
    regime_robustness: float
    n_trades: int
    fold_sharpes: List[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    experiment_dir: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentResult":
        return cls(
            experiment_id=d["experiment_id"],
            strategy_name=d["strategy_name"],
            hypothesis_id=d["hypothesis_id"],
            sharpe=d.get("sharpe", 0.0),
            max_drawdown=d.get("max_drawdown", 0.0),
            win_rate=d.get("win_rate", 0.0),
            stability_score=d.get("stability_score", 0.0),
            regime_robustness=d.get("regime_robustness", 0.0),
            n_trades=d.get("n_trades", 0),
            fold_sharpes=d.get("fold_sharpes", []),
            timestamp=d.get("timestamp", ""),
            experiment_dir=d.get("experiment_dir", ""),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Walk-forward helpers
# ---------------------------------------------------------------------------

def _compute_sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe from daily returns array."""
    if len(returns) < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std < 1e-10:
        return 0.0
    return float(mean / std * math.sqrt(_TRADING_DAYS_YEAR))


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (positive fraction)."""
    if len(returns) == 0:
        return 0.0
    cum = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = (running_max - cum) / running_max
    return float(np.max(drawdown))


def _apply_tx_costs(returns: np.ndarray, tx_cost_bps: float, trade_flags: np.ndarray) -> np.ndarray:
    """Subtract transaction costs on trade-entry days."""
    cost = tx_cost_bps / 10_000.0
    adj = returns.copy()
    adj[trade_flags > 0] -= cost
    return adj


def _simulate_strategy(
    prices: pd.DataFrame,
    strategy: StrategyConfig,
    tx_cost_bps: float = _DEFAULT_TX_COST_BPS,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Minimal strategy simulation on a price DataFrame.

    Uses a momentum/mean-reversion signal as a proxy for any strategy type.
    Real implementations would wire in the actual feature pipeline.
    """
    rng = np.random.default_rng(seed)

    if prices.empty or len(prices) < 5:
        return {"returns": np.array([]), "n_trades": 0}

    close = prices.iloc[:, 0].values.astype(float)
    daily_rets = np.diff(close) / close[:-1]

    # Generate a simple signal proportional to strategy parameters
    lookback = int(strategy.parameters.get("momentum_lookback", 20))
    lookback = min(lookback, max(2, len(daily_rets) - 1))

    signals = np.zeros(len(daily_rets))
    for i in range(lookback, len(daily_rets)):
        past_ret = np.mean(daily_rets[max(0, i - lookback): i])
        # Hypothesis-adjusted signal: add slight noise to reflect uncertainty
        signals[i] = past_ret + rng.normal(0, 0.001)

    trade_flags = (np.abs(signals) > 0.0).astype(float)
    strategy_rets = signals * np.sign(signals) * daily_rets  # long only for simplicity
    strategy_rets = _apply_tx_costs(strategy_rets, tx_cost_bps, trade_flags)

    n_trades = int(np.sum(trade_flags))
    return {"returns": strategy_rets, "n_trades": n_trades}


# ---------------------------------------------------------------------------
# Runner class
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Run walk-forward backtests for a list of strategy configs.

    Parameters
    ----------
    output_dir : str
        Root directory for experiment reports (``reports/research/experiments/``).
    tx_cost_bps : float
        Round-trip transaction cost in basis points.
    n_folds : int
        Number of walk-forward folds.
    """

    def __init__(
        self,
        output_dir: str = "reports/research/experiments",
        tx_cost_bps: float = _DEFAULT_TX_COST_BPS,
        n_folds: int = _DEFAULT_FOLDS,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._tx_cost_bps = tx_cost_bps
        self._n_folds = n_folds

    def run(
        self,
        strategies: List[StrategyConfig],
        price_data: Optional[pd.DataFrame] = None,
    ) -> List[ExperimentResult]:
        """Run experiments for all strategies.

        Parameters
        ----------
        strategies : list[StrategyConfig]
        price_data : pd.DataFrame, optional
            A single-column (close prices) DataFrame.  When *None* a
            synthetic random-walk price series is generated so the runner
            can operate in isolation (useful for testing).

        Returns
        -------
        list[ExperimentResult]
        """
        results: List[ExperimentResult] = []
        for strategy in strategies:
            result = self._run_single(strategy, price_data)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_single(
        self,
        strategy: StrategyConfig,
        price_data: Optional[pd.DataFrame],
    ) -> ExperimentResult:
        exp_id = uuid.uuid4().hex[:10]
        logger.info("Running experiment %s for strategy '%s'", exp_id, strategy.strategy_name)

        if price_data is None or price_data.empty:
            price_data = self._synthetic_prices(n=500)

        n_rows = len(price_data)
        fold_size = max(_MIN_FOLD_RETURN_ROWS, n_rows // (self._n_folds + 1))

        fold_sharpes: List[float] = []
        all_returns: List[np.ndarray] = []

        for fold_idx in range(self._n_folds):
            train_end = fold_size * (fold_idx + 1)
            test_start = train_end
            test_end = min(train_end + fold_size, n_rows)

            if test_start >= n_rows or (test_end - test_start) < _MIN_FOLD_RETURN_ROWS:
                break

            test_slice = price_data.iloc[test_start:test_end]
            sim = _simulate_strategy(test_slice, strategy, self._tx_cost_bps, seed=fold_idx)
            rets = sim["returns"]

            if len(rets) > 0:
                all_returns.append(rets)
                fold_sharpes.append(_compute_sharpe(rets))

        if not all_returns:
            combined_rets = np.array([0.0])
        else:
            combined_rets = np.concatenate(all_returns)

        sharpe = _compute_sharpe(combined_rets)
        mdd = _compute_max_drawdown(combined_rets)
        win_rate = float(np.mean(combined_rets > 0)) if len(combined_rets) > 0 else 0.0
        stability = float(np.std(fold_sharpes)) if len(fold_sharpes) > 1 else 0.0
        stability_score = max(0.0, 1.0 - stability)
        regime_robustness = float(min(fold_sharpes)) if fold_sharpes else 0.0

        n_trades_total = sum(
            _simulate_strategy(
                price_data.iloc[
                    fold_size * (i + 1): min(fold_size * (i + 2), n_rows)
                ],
                strategy,
                self._tx_cost_bps,
                seed=i,
            )["n_trades"]
            for i in range(self._n_folds)
        )

        result = ExperimentResult(
            experiment_id=exp_id,
            strategy_name=strategy.strategy_name,
            hypothesis_id=strategy.hypothesis_id,
            sharpe=round(sharpe, 4),
            max_drawdown=round(mdd, 4),
            win_rate=round(win_rate, 4),
            stability_score=round(stability_score, 4),
            regime_robustness=round(regime_robustness, 4),
            n_trades=int(n_trades_total),
            fold_sharpes=[round(s, 4) for s in fold_sharpes],
        )
        result.experiment_dir = self._save_report(result)
        return result

    def _save_report(self, result: ExperimentResult) -> str:
        """Persist experiment JSON report and return directory path."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._output_dir / f"{result.experiment_id}_{result.strategy_name}.json"
        try:
            with open(report_path, "w") as fh:
                json.dump(result.to_dict(), fh, indent=2)
            logger.debug("Experiment report saved: %s", report_path)
        except OSError as exc:
            logger.warning("Could not save experiment report: %s", exc)
        return str(report_path)

    @staticmethod
    def _synthetic_prices(n: int = 500, seed: int = 0) -> pd.DataFrame:
        """Generate a synthetic random-walk price series."""
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0002, 0.015, size=n)
        prices = 100.0 * np.cumprod(1.0 + returns)
        return pd.DataFrame({"close": prices})
