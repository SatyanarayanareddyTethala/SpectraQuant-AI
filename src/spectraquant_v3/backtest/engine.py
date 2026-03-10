"""Backtest orchestration engine for SpectraQuant-AI-V3.

Runs a walk-forward simulation by replaying the production
signal → meta-policy → allocation pipeline over a historical OHLCV
dataset.

Walk-forward modes
------------------
``expanding``
    The training window grows at every step: the model always sees all
    data from the start up to the current rebalance date.

``rolling``
    A fixed-length lookback window slides forward with each step.
    Set ``lookback_periods`` to control the window size.

For each rebalance step the engine:

1. Slices each symbol's OHLCV DataFrame to the in-sample window.
2. Computes features (via the supplied feature engine).
3. Runs the signal agent(s) on the latest available row.
4. Applies the meta-policy filter.
5. Runs the allocator to obtain target weights.
6. Computes the step's portfolio return by applying weights to the *next*
   period's actual returns (``close[t+1] / close[t] - 1``).
7. Accumulates the NAV and records a :class:`RebalanceSnapshot`.

After all steps the engine computes aggregate performance metrics and
returns a :class:`~spectraquant_v3.backtest.results.BacktestResults`.

Usage example::

    from spectraquant_v3.backtest.engine import BacktestEngine

    engine = BacktestEngine(
        cfg=cfg,
        asset_class="crypto",
        price_data={"BTC": df_btc, "ETH": df_eth},
        rebalance_freq="W",          # weekly rebalance
        window_type="expanding",
        min_in_sample_periods=60,    # 60 bars minimum
        run_id="bt_demo",
    )
    results = engine.run()
    print(results.summary())
    results.write("/tmp/backtest_output")
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant_v3.backtest.results import BacktestResults, RebalanceSnapshot
from spectraquant_v3.core.enums import AssetClass
from spectraquant_v3.core.errors import DataSchemaError, EmptyPriceDataError
from spectraquant_v3.execution.simulator import ExecutionSimulator
from spectraquant_v3.pipeline.allocator import Allocator
from spectraquant_v3.pipeline.meta_policy import MetaPolicy
from spectraquant_v3.strategies.agents.runner import run_signal_agent

logger = logging.getLogger(__name__)

# Minimum in-sample rows before the first rebalance step is attempted
_DEFAULT_MIN_PERIODS = 30


class BacktestEngine:
    """Walk-forward backtesting engine for crypto and equity pipelines.

    Args:
        cfg:                   Merged pipeline configuration dict (same as
                               used by the production pipeline).
        asset_class:           ``"crypto"`` or ``"equity"``.
        price_data:            Dict mapping canonical symbol → daily OHLCV
                               DataFrame with a DatetimeIndex.
        rebalance_freq:        Pandas offset alias controlling rebalance
                               cadence, e.g. ``"D"`` (daily), ``"W"``
                               (weekly), ``"ME"`` (month-end).
                               Defaults to ``"W"``.
        window_type:           ``"expanding"`` or ``"rolling"``.
        lookback_periods:      Number of periods in a rolling window (only
                               used when ``window_type="rolling"``).
        min_in_sample_periods: Minimum bars required in the in-sample slice
                               before a rebalance step is executed.
        commission:            Commission cost in basis points (bps).
        slippage:              Slippage cost in basis points (bps).
        spread:                Spread cost in basis points (bps).
        run_id:                Identifier for this backtest run.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        asset_class: str,
        price_data: dict[str, pd.DataFrame],
        strategy_id: str | None = None,
        rebalance_freq: str = "W",
        window_type: str = "expanding",
        lookback_periods: int = 252,
        min_in_sample_periods: int = _DEFAULT_MIN_PERIODS,
        commission: float = 0.0,
        slippage: float = 0.0,
        spread: float = 0.0,
        run_id: str = "backtest",
    ) -> None:
        if not price_data:
            raise EmptyPriceDataError(
                "BacktestEngine: price_data must contain at least one symbol."
            )
        _valid_asset = {a.value for a in AssetClass}
        if asset_class not in _valid_asset:
            raise ValueError(
                f"BacktestEngine: asset_class must be one of {sorted(_valid_asset)}, "
                f"got {asset_class!r}."
            )
        if window_type not in ("expanding", "rolling"):
            raise ValueError(
                f"BacktestEngine: window_type must be 'expanding' or 'rolling', "
                f"got {window_type!r}."
            )

        self._strategy_id = strategy_id
        if strategy_id:
            from spectraquant_v3.strategies.loader import StrategyLoader  # noqa: PLC0415

            self._cfg = StrategyLoader.build_pipeline_config(strategy_id, cfg)
        else:
            self._cfg = cfg
        self._asset_class = asset_class
        self._price_data = price_data
        self._rebalance_freq = rebalance_freq
        self._window_type = window_type
        self._lookback_periods = lookback_periods
        self._min_in_sample_periods = min_in_sample_periods
        self._commission_bps = float(commission)
        self._slippage_bps = float(slippage)
        self._spread_bps = float(spread)
        self._run_id = run_id

        # Cost model uses the same execution simulator semantics as paper mode.
        # We encode commission/slippage/spread (all in bps) as an aggregate
        # turnover penalty in bps so backtests remain deterministic and aligned.
        self._cost_simulator = ExecutionSimulator(
            slippage_bps=0.0,
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            turnover_penalty_bps=(
                self._commission_bps + self._slippage_bps + self._spread_bps
            ),
            mode="paper",
        )

        # Instantiate shared pipeline components once
        self._meta_policy = MetaPolicy.from_config(self._cfg)
        self._allocator = Allocator.from_config(self._cfg, run_id=run_id)
        self._feature_engine = self._build_feature_engine()
        self._signal_agent = self._build_signal_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResults:
        """Execute the walk-forward backtest.

        Returns:
            :class:`~spectraquant_v3.backtest.results.BacktestResults`
            containing per-step snapshots and aggregated metrics.
        """
        rebalance_dates = self._rebalance_dates()
        if not rebalance_dates:
            raise EmptyPriceDataError(
                "BacktestEngine: no rebalance dates generated. "
                "Check that price_data spans at least min_in_sample_periods bars."
            )

        snapshots: list[RebalanceSnapshot] = []
        nav = 1.0
        step_returns: list[float] = []
        prev_weights: dict[str, float] = {}

        for date in rebalance_dates:
            snapshot = self._step(date, nav, prev_weights)
            if snapshot is None:
                continue
            nav = snapshot.portfolio_value
            step_returns.append(snapshot.step_return)
            prev_weights = snapshot.allocations.copy()
            snapshots.append(snapshot)

        return self._compile_results(snapshots, step_returns)

    # ------------------------------------------------------------------
    # Internal – walk-forward step
    # ------------------------------------------------------------------

    def _step(
        self,
        date: pd.Timestamp,
        prev_nav: float,
        prev_weights: dict[str, float],
    ) -> RebalanceSnapshot | None:
        """Execute one rebalance step at *date*.

        Returns ``None`` when there is insufficient data to compute signals.
        """
        as_of_str = date.isoformat()

        # Slice in-sample data for each symbol
        in_sample: dict[str, pd.DataFrame] = {}
        for sym, df in self._price_data.items():
            sliced = self._slice_in_sample(df, date)
            if sliced is not None and len(sliced) >= self._min_in_sample_periods:
                in_sample[sym] = sliced

        if not in_sample:
            logger.debug("BacktestEngine: no symbols with sufficient data at %s", date)
            return None

        # Compute features
        feature_map: dict[str, pd.DataFrame] = {}
        for sym, df in in_sample.items():
            try:
                feature_map[sym] = self._feature_engine.transform(df, symbol=sym)
            except (DataSchemaError, EmptyPriceDataError, ValueError) as exc:
                logger.debug(
                    "BacktestEngine: feature error for %s at %s: %s",
                    sym,
                    date,
                    exc,
                )

        if not feature_map:
            return None

        # Generate signals
        signals = run_signal_agent(self._signal_agent, feature_map, as_of=as_of_str)
        signals_ok = sum(1 for s in signals if s.status == "OK")
        signals_nosig = sum(1 for s in signals if s.status == "NO_SIGNAL")

        # Meta-policy
        decisions = self._meta_policy.run(signals)
        policy_passed = sum(1 for d in decisions if d.passed)
        policy_blocked = sum(1 for d in decisions if not d.passed)

        blocked_reasons: dict[str, int] = {}
        for d in decisions:
            if not d.passed:
                blocked_reasons[d.reason] = blocked_reasons.get(d.reason, 0) + 1

        no_signal_reasons: dict[str, int] = {}
        for s in signals:
            if s.status == "NO_SIGNAL":
                reason = s.rationale or "no_signal"
                no_signal_reasons[reason] = no_signal_reasons.get(reason, 0) + 1

        # Allocate weights
        vol_map = self._extract_vol_map(feature_map)
        allocations = self._allocate_for_decisions(decisions, vol_map)
        weight_map = {
            a.canonical_symbol: a.target_weight for a in allocations if not a.blocked
        }

        # Compute gross step return using next-period actual returns
        gross_return = self._compute_step_return(weight_map, date)

        turnover, trading_cost = self._compute_turnover_and_cost(
            prev_weights=prev_weights,
            target_weights=weight_map,
        )
        net_return = gross_return - trading_cost

        positions_count = sum(1 for w in weight_map.values() if abs(w) > 0.0)
        exposure = float(sum(abs(w) for w in weight_map.values()))

        nav = prev_nav * (1.0 + net_return)

        return RebalanceSnapshot(
            date=as_of_str,
            universe=list(in_sample.keys()),
            signals_ok=signals_ok,
            signals_nosig=signals_nosig,
            policy_passed=policy_passed,
            policy_blocked=policy_blocked,
            allocations=weight_map,
            portfolio_value=nav,
            step_return=net_return,
            gross_return=gross_return,
            net_return=net_return,
            turnover=turnover,
            positions_count=positions_count,
            exposure=exposure,
            blocked_reasons=blocked_reasons,
            no_signal_reasons=no_signal_reasons,
            selected_symbols=sorted(weight_map.keys()),
            composite_scores = {
                 str(getattr(d, "canonical_symbol")): float(getattr(d, "composite_score"))
                 for d in decisions
                 if getattr(d, "canonical_symbol", None) is not None
                  and getattr(d, "composite_score", None) is not None 
               }  
      )


    def _allocate_for_decisions(self, decisions, vol_map: dict[str, float]):
        from spectraquant_v3.core.schema import AllocationRow  # noqa: PLC0415

        if self._strategy_id:
            from spectraquant_v3.strategies.loader import StrategyLoader  # noqa: PLC0415

            defn = StrategyLoader.load(self._strategy_id)
            if defn.allocator == "rank_vol_target_allocator":
                from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry  # noqa: PLC0415

                rank_alloc = AllocatorRegistry.get(defn.allocator).from_config(self._cfg, run_id=self._run_id)
                ranked_input = {
                    d.canonical_symbol: {
                        "rank": i + 1,
                        "confidence": d.composite_confidence,
                        "vol": vol_map.get(d.canonical_symbol, 0.0),
                    }
                    for i, d in enumerate(sorted((x for x in decisions if x.passed), key=lambda x: abs(x.composite_score), reverse=True))
                }
                weights, _diag = rank_alloc.allocate(ranked_input)
                return [
                    AllocationRow(
                        run_id=self._run_id,
                        canonical_symbol=d.canonical_symbol,
                        asset_class=d.asset_class,
                        target_weight=float(weights.get(d.canonical_symbol, 0.0)),
                        blocked=not d.passed,
                        blocked_reason="" if d.passed else d.reason,
                    )
                    for d in decisions
                ]

        return self._allocator.allocate(decisions, vol_map=vol_map or None)

    # ------------------------------------------------------------------
    # Internal – helpers
    # ------------------------------------------------------------------

    def _rebalance_dates(self) -> list[pd.Timestamp]:
        """Compute rebalance dates that fall within the price data range."""
        all_indices = [df.index for df in self._price_data.values() if not df.empty]
        if not all_indices:
            return []

        global_start = max(idx.min() for idx in all_indices)
        global_end = min(idx.max() for idx in all_indices)

        if global_start >= global_end:
            return []

        all_dates_union = sorted(set().union(*[set(idx.date.tolist()) for idx in all_indices]))
        if len(all_dates_union) <= self._min_in_sample_periods:
            return []

        warmup_end = all_dates_union[self._min_in_sample_periods - 1]
        warmup_end_ts = pd.Timestamp(warmup_end, tz="UTC")
        effective_start = max(global_start, warmup_end_ts)

        try:
            dates = pd.date_range(
                start=effective_start,
                end=global_end,
                freq=self._rebalance_freq,
            )
        except ValueError:
            logger.warning(
                "BacktestEngine: invalid rebalance_freq=%r; no dates generated.",
                self._rebalance_freq,
            )
            return []

        all_ts = set().union(*[set(idx.normalize()) for idx in all_indices])
        valid = [d for d in dates if d.normalize() in all_ts]
        return list(valid)

    def _slice_in_sample(
        self, df: pd.DataFrame, date: pd.Timestamp
    ) -> pd.DataFrame | None:
        """Return the in-sample slice of *df* up to and including *date*."""
        try:
            mask = df.index <= date
        except (TypeError, AttributeError):
            return None

        sliced = df.loc[mask]
        if sliced.empty:
            return None

        if self._window_type == "rolling":
            sliced = sliced.iloc[-self._lookback_periods :]

        return sliced

    def _extract_vol_map(
        self, feature_map: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """Extract last realised-vol value from feature DataFrames."""
        vol_map: dict[str, float] = {}
        for sym, df in feature_map.items():
            if "vol_realised" in df.columns and not df["vol_realised"].empty:
                v = float(df["vol_realised"].iloc[-1])
                if math.isfinite(v) and v > 0:
                    vol_map[sym] = v
        return vol_map

    def _compute_step_return(
        self, weight_map: dict[str, float], date: pd.Timestamp
    ) -> float:
        """Compute the one-period portfolio return for the given weights.

        Uses the *next* available close-to-close return after *date* for
        each symbol in the portfolio. Symbols without a next-period return
        are treated as flat (0 return).

        Applies a simple portfolio-level volatility target using the latest
        realised vol estimates from the feature engine when available.
        """
        if not weight_map:
            return 0.0

        # Estimate current portfolio volatility from latest in-sample realised vols
        vol_estimates: list[float] = []
        for sym, weight in weight_map.items():
            df = self._price_data.get(sym)
            if df is None or df.empty:
                continue

            hist = self._slice_in_sample(df, date)
            if hist is None or hist.empty:
                continue

            try:
                feat = self._feature_engine.transform(hist, symbol=sym)
            except Exception:
                continue

            if "vol_realised" not in feat.columns or feat["vol_realised"].empty:
                continue

            try:
                vol_val = float(feat["vol_realised"].iloc[-1])
            except Exception:
                continue

            if math.isfinite(vol_val) and vol_val > 0:
                vol_estimates.append(abs(weight) * vol_val)

        # Read target vol from config
        target_vol = float(self._cfg.get("backtest", {}).get("target_vol", 0.20))

        # Simple portfolio vol estimate
        portfolio_vol = float(sum(vol_estimates)) if vol_estimates else 0.0

        # Scale down only; do not lever up
        scale = 1.0
        if portfolio_vol > 0:
            scale = min(1.0, target_vol / portfolio_vol)

        total = 0.0
        for sym, weight in weight_map.items():
            df = self._price_data.get(sym)
            if df is None or df.empty:
                continue

            future = df.loc[df.index > date]
            current = df.loc[df.index <= date]

            if future.empty or current.empty:
                continue

            try:
                r = float(future["close"].iloc[0]) / float(current["close"].iloc[-1]) - 1.0
                if math.isfinite(r):
                    total += (weight * scale) * r
            except (KeyError, ZeroDivisionError, ValueError):
                continue

        return total

    def _compute_turnover_and_cost(
        self,
        prev_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> tuple[float, float]:
        """Return (turnover, trading_cost) for a rebalance.

        Turnover is one-way portfolio turnover ``sum(|w_t - w_{t-1}|)``.
        Costs are modelled using execution simulator semantics and returned
        as decimal return drag (e.g., 0.001 = 10 bps).
        """
        all_symbols = set(prev_weights) | set(target_weights)
        changed_symbols = {
            sym
            for sym in all_symbols
            if not math.isclose(
                float(target_weights.get(sym, 0.0)),
                float(prev_weights.get(sym, 0.0)),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        }
        if not changed_symbols:
            return 0.0, 0.0

        execution_targets = {sym: float(target_weights.get(sym, 0.0)) for sym in changed_symbols}
        execution_prev = {sym: float(prev_weights.get(sym, 0.0)) for sym in changed_symbols}

        execution_results = self._cost_simulator.execute_weights(
            target_weights=execution_targets,
            prev_weights=execution_prev,
            prices={sym: 1.0 for sym in changed_symbols},
        )
        turnover = float(sum(r.metadata.get("turnover", 0.0) for r in execution_results))
        trading_cost = float(-sum(r.net_return_impact for r in execution_results))
        return turnover, trading_cost

    def _build_feature_engine(self):
        """Instantiate the correct feature engine for this asset class."""
        if self._asset_class == AssetClass.CRYPTO.value:
            from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine  # noqa: PLC0415

            return CryptoFeatureEngine.from_config(self._cfg)
        else:
            from spectraquant_v3.equities.features.engine import EquityFeatureEngine  # noqa: PLC0415

            return EquityFeatureEngine.from_config(self._cfg)

    def _build_signal_agent(self):
        """Instantiate the correct signal agent for this asset class."""
        if self._asset_class == AssetClass.CRYPTO.value:
            if self._strategy_id:
                from spectraquant_v3.strategies.loader import StrategyLoader  # noqa: PLC0415
                from spectraquant_v3.strategies.agents.registry import AgentRegistry  # noqa: PLC0415

                defn = StrategyLoader.load(self._strategy_id)
                agent_cls = AgentRegistry.get(defn.agents[0])
                if hasattr(agent_cls, "from_config"):
                    return agent_cls.from_config(self._cfg, run_id=self._run_id)
                return agent_cls(run_id=self._run_id)

            from spectraquant_v3.crypto.signals.momentum import CryptoMomentumAgent  # noqa: PLC0415

            return CryptoMomentumAgent.from_config(self._cfg, run_id=self._run_id)
        else:
            from spectraquant_v3.equities.signals.momentum import EquityMomentumAgent  # noqa: PLC0415

            return EquityMomentumAgent.from_config(self._cfg, run_id=self._run_id)

    # ------------------------------------------------------------------
    # Results compilation
    # ------------------------------------------------------------------

    def _compile_results(
        self,
        snapshots: list[RebalanceSnapshot],
        step_returns: list[float],
    ) -> BacktestResults:
        """Compute aggregate performance metrics from step data."""
        generated_at = datetime.now(timezone.utc).isoformat()

        if not snapshots:
            return BacktestResults(
                run_id=self._run_id,
                asset_class=self._asset_class,
                start_date="",
                end_date="",
                n_steps=0,
                generated_at=generated_at,
            )

        start_date = snapshots[0].date
        end_date = snapshots[-1].date

        all_symbols = sorted({sym for s in snapshots for sym in s.universe})

        nav_series = np.array([s.portfolio_value for s in snapshots])
        returns_arr = np.array(step_returns)

        total_return = float(nav_series[-1] - 1.0)

        periods_per_year = self._infer_periods_per_year(
            snapshots[0].date, snapshots[-1].date, len(snapshots)
        )

        if len(nav_series) > 1:
            annualised_return = float(
                (1.0 + total_return) ** (periods_per_year / len(nav_series)) - 1.0
            )
        else:
            annualised_return = total_return

        ann_vol = (
            float(np.std(returns_arr, ddof=1) * math.sqrt(periods_per_year))
            if len(returns_arr) > 1
            else 0.0
        )

        sharpe = annualised_return / ann_vol if ann_vol > 0 else 0.0

        running_max = np.maximum.accumulate(nav_series)
        drawdowns = (nav_series - running_max) / running_max
        max_dd = float(np.min(drawdowns))

        calmar = abs(annualised_return / max_dd) if max_dd < 0 else 0.0
        win_rate = float(np.mean(returns_arr > 0)) if len(returns_arr) > 0 else 0.0
        avg_turnover = (
            float(np.mean([s.turnover for s in snapshots])) if snapshots else 0.0
        )
        avg_positions = (
            float(np.mean([s.positions_count for s in snapshots])) if snapshots else 0.0
        )
        avg_exposure = (
            float(np.mean([s.exposure for s in snapshots])) if snapshots else 0.0
        )

        return BacktestResults(
            run_id=self._run_id,
            asset_class=self._asset_class,
            start_date=start_date,
            end_date=end_date,
            n_steps=len(snapshots),
            symbols=all_symbols,
            total_return=total_return,
            annualised_return=annualised_return,
            annualised_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            turnover=avg_turnover,
            avg_positions=avg_positions,
            exposure=avg_exposure,
            snapshots=snapshots,
            generated_at=generated_at,
        )

    @staticmethod
    def _infer_periods_per_year(
        start_date_str: str, end_date_str: str, n_steps: int
    ) -> float:
        """Estimate how many steps occur in one calendar year."""
        try:
            start = datetime.fromisoformat(start_date_str)
            end = datetime.fromisoformat(end_date_str)
            days = max(1.0, (end - start).days)
            return n_steps * 365.25 / days
        except Exception:  # noqa: BLE001
            return 52.0
