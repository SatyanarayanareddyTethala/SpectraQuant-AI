"""Experiment manager for SpectraQuant-AI-V3.

The :class:`ExperimentManager` orchestrates experiment runs:

1. Creates an :class:`~spectraquant_v3.experiments.run_tracker.RunTracker` for the run.
2. Executes the strategy via :func:`~spectraquant_v3.pipeline.run_strategy`.
3. Extracts performance metrics from the pipeline result.
4. Persists everything through the :class:`~spectraquant_v3.experiments.result_store.ResultStore`.

It also provides :meth:`compare_experiments` to produce a comparison table
across multiple experiment IDs.

Usage::

    from spectraquant_v3.experiments.experiment_manager import ExperimentManager
    from spectraquant_v3.core.config import get_crypto_config

    cfg = get_crypto_config()
    manager = ExperimentManager()
    result = manager.run_experiment(
        experiment_id="exp_001",
        strategy_id="crypto_momentum_v1",
        cfg=cfg,
        dry_run=True,
    )
    print(result["experiment_id"])   # "exp_001"
    print(result["metrics"])         # {"sharpe": ..., ...}

    table = manager.compare_experiments(["exp_001", "exp_002"])
    # Returns a list of dicts with keys:
    #   experiment_id, strategy_id, sharpe, cagr, max_drawdown, volatility, win_rate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectraquant_v3.core.enums import RunMode
from spectraquant_v3.experiments.result_store import ResultStore
from spectraquant_v3.experiments.run_tracker import RunTracker


class ExperimentManager:
    """Coordinate experiment execution, tracking, and comparison.

    Args:
        base_dir: Root directory for experiment results.  Defaults to
                  ``reports/experiments``.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.store = ResultStore(base_dir)

    # ------------------------------------------------------------------
    # Running experiments
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        experiment_id: str,
        strategy_id: str,
        cfg: dict[str, Any],
        run_mode: RunMode = RunMode.NORMAL,
        dry_run: bool = False,
        price_data: dict | None = None,
        market_data: dict | None = None,
        dataset_version: str = "",
        run_id: str | None = None,
        project_root: str | None = None,
        backtest_results: Any | None = None,
    ) -> dict[str, Any]:
        """Run a strategy experiment and persist results.

        Args:
            experiment_id:    Unique experiment identifier.
            strategy_id:      Registered strategy to run.
            cfg:              Merged pipeline config.
            run_mode:         Cache behaviour.
            dry_run:          When True skip all writes and network calls.
            price_data:       Pre-loaded OHLCV frames for the pipeline.
            market_data:      Optional market metrics for universe gating.
            dataset_version:  Optional label identifying the dataset snapshot.
            run_id:           Override the auto-generated pipeline run ID.
            project_root:     Override repo-root discovery.
            backtest_results: Optional pre-computed
                              :class:`~spectraquant_v3.backtest.results.BacktestResults`
                              object.  When supplied, its metrics are merged in.

        Returns:
            Dict with keys: ``experiment_id``, ``strategy_id``, ``run_id``,
            ``status``, ``metrics``, ``artefact_paths``, ``universe``,
            ``signals``, ``decisions``, ``allocations``.
        """
        from spectraquant_v3.pipeline import run_strategy

        tracker = RunTracker(
            experiment_id=experiment_id,
            strategy_id=strategy_id,
            dataset_version=dataset_version,
            config=cfg,
        )

        pipeline_result = run_strategy(
            strategy_id=strategy_id,
            cfg=cfg,
            run_mode=run_mode,
            dry_run=dry_run,
            price_data=price_data,
            market_data=market_data,
            run_id=run_id,
            project_root=project_root,
        )

        # Extract lightweight metrics from backtest results (if supplied)
        metrics: dict[str, Any] = {}
        if backtest_results is not None:
            metrics.update(self._metrics_from_backtest(backtest_results))

        tracker.record_metrics(metrics)

        # Record artefact paths from the pipeline result
        for key, path in pipeline_result.get("artefact_paths", {}).items():
            tracker.record_artefact(key, path)

        # Persist to disk (config.json + metrics.json)
        if not dry_run:
            paths = tracker.save(self.store)

            # Also write a dataset manifest if price_data is supplied
            if price_data:
                manifest = self._build_dataset_manifest(
                    strategy_id, dataset_version, price_data
                )
                self.store.write_dataset_manifest(experiment_id, manifest)
                paths["dataset_manifest"] = str(
                    self.store.experiment_dir(experiment_id) / "dataset_manifest.json"
                )

            # Write backtest summary if results are available
            if backtest_results is not None:
                summary = self._backtest_summary(backtest_results)
                self.store.write_backtest_summary(experiment_id, summary)
                paths["backtest_summary"] = str(
                    self.store.experiment_dir(experiment_id) / "backtest_summary.json"
                )

        return {
            "experiment_id": experiment_id,
            "strategy_id": strategy_id,
            "run_id": pipeline_result.get("run_id"),
            "status": pipeline_result.get("status"),
            "metrics": tracker.metrics,
            "artefact_paths": tracker.artefact_paths,
            "universe": pipeline_result.get("universe", []),
            "signals": pipeline_result.get("signals", []),
            "decisions": pipeline_result.get("decisions", []),
            "allocations": pipeline_result.get("allocations", []),
        }

    # ------------------------------------------------------------------
    # Experiment comparison
    # ------------------------------------------------------------------

    def compare_experiments(
        self, exp_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Return a comparison table for the given experiment IDs.

        Each row contains:
        ``experiment_id``, ``strategy_id``, ``sharpe``, ``cagr``,
        ``max_drawdown``, ``volatility``, ``win_rate``, ``turnover``.

        Missing metric fields default to ``None``.

        Args:
            exp_ids: List of experiment identifiers to compare.

        Returns:
            List of dicts, one per experiment.  Suitable for printing as
            a table or converting to a ``pandas.DataFrame``.
        """
        rows = []
        for eid in exp_ids:
            try:
                config_doc = self.store.read_config(eid)
            except FileNotFoundError:
                rows.append({"experiment_id": eid, "error": "not_found"})
                continue

            try:
                metrics = self.store.read_metrics(eid)
            except FileNotFoundError:
                metrics = {}

            effective_metrics = dict(config_doc.get("metrics_payload") or {})
            if isinstance(metrics, dict):
                effective_metrics.update(metrics)

            rows.append(
                {
                    "experiment_id": eid,
                    "strategy_id": config_doc.get("strategy_id"),
                    "dataset_version": config_doc.get("dataset_version"),
                    "config_hash": config_doc.get("config_hash"),
                    "run_timestamp": config_doc.get("run_timestamp"),
                    "sharpe": effective_metrics.get("sharpe"),
                    "cagr": effective_metrics.get("cagr"),
                    "max_drawdown": effective_metrics.get("max_drawdown"),
                    "volatility": effective_metrics.get("volatility"),
                    "win_rate": effective_metrics.get("win_rate"),
                    "turnover": effective_metrics.get("turnover"),
                    "total_return": effective_metrics.get("total_return"),
                    "calmar": effective_metrics.get("calmar"),
                    "n_steps": effective_metrics.get("n_steps"),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Hybrid strategy variant support
    # ------------------------------------------------------------------

    def run_hybrid_backtest_experiment(
        self,
        experiment_id: str,
        params: "HybridStrategyParams",  # noqa: F821 – imported lazily below
        price_data: "dict[str, Any]",
        base_cfg: "dict[str, Any]",
        news_feature_map: "dict[str, Any] | None" = None,
        rebalance_freq: str = "ME",
        window_type: str = "expanding",
        min_in_sample_periods: int = 30,
        commission: float = 0.0,
        slippage: float = 0.0,
        spread: float = 0.0,
        dataset_version: str = "",
        dry_run: bool = False,
    ) -> "dict[str, Any]":
        """Run a hybrid strategy variant backtest and record the experiment.

        This is the primary entry point for hybrid strategy research.
        It:

        1. Injects ``params`` into a copy of ``base_cfg``, so that the hybrid
           agent receives the correct blend weights and thresholds.
        2. Runs a :class:`~spectraquant_v3.backtest.engine.BacktestEngine`
           walk-forward simulation using the configured strategy.
        3. Records all blend parameters as ``hybrid_params`` in the
           experiment's ``config.json`` for full reproducibility.
        4. Persists standard metrics plus the blend params so that
           :meth:`compare_hybrid_variants` can produce a clean side-by-side
           table.

        Args:
            experiment_id:        Unique experiment identifier.
            params:               :class:`~spectraquant_v3.experiments.hybrid_params.HybridStrategyParams`
                                  controlling the blend weights/thresholds.
            price_data:           Dict of canonical symbol → OHLCV DataFrame.
            base_cfg:             Base pipeline config (not mutated).
            news_feature_map:     Optional dict of symbol → news DataFrame.
            rebalance_freq:       Pandas offset alias for rebalance cadence.
            window_type:          ``"expanding"`` or ``"rolling"``.
            min_in_sample_periods: Minimum bars before first rebalance.
            commission:           Commission cost in bps.
            slippage:             Slippage cost in bps.
            spread:               Spread cost in bps.
            dataset_version:      Dataset label for reproducibility.
            dry_run:              When ``True`` skip all file writes.

        Returns:
            Dict with keys: ``experiment_id``, ``strategy_id``, ``run_id``,
            ``hybrid_params``, ``metrics``, ``artefact_paths``,
            ``backtest_run_id``, ``n_steps``.
        """
        from spectraquant_v3.backtest.engine import BacktestEngine  # noqa: PLC0415
        from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams  # noqa: PLC0415, F401

        # Build cfg with hybrid params applied
        cfg = params.inject_into_cfg(base_cfg)

        # Determine asset_class from strategy_id
        from spectraquant_v3.strategies.registry import StrategyRegistry  # noqa: PLC0415

        try:
            defn = StrategyRegistry.get(params.strategy_id)
            asset_class = defn.asset_class
        except KeyError:
            # Fallback: infer from strategy_id naming convention
            asset_class = (
                "crypto" if "crypto" in params.strategy_id else "equity"
            )

        run_id = params.run_id()

        engine = BacktestEngine(
            cfg=cfg,
            asset_class=asset_class,
            price_data=price_data,
            strategy_id=params.strategy_id,
            rebalance_freq=rebalance_freq,
            window_type=window_type,
            min_in_sample_periods=min_in_sample_periods,
            commission=commission,
            slippage=slippage,
            spread=spread,
            run_id=run_id,
            news_feature_map=news_feature_map or {},
        )

        bt_results = engine.run()
        metrics = self._metrics_from_backtest(bt_results)
        params_dict = params.to_dict()

        tracker = RunTracker(
            experiment_id=experiment_id,
            strategy_id=params.strategy_id,
            dataset_version=dataset_version,
            config={
                **cfg,
                "hybrid_params": params_dict,
            },
        )
        tracker.record_metrics(metrics)

        if not dry_run:
            paths = tracker.save(self.store)

            # Write hybrid_params.json for quick lookup
            self.store.write_hybrid_params(experiment_id, params_dict)

            # Write backtest summary
            summary = self._backtest_summary(bt_results)
            self.store.write_backtest_summary(experiment_id, summary)

            if price_data:
                manifest = self._build_dataset_manifest(
                    params.strategy_id, dataset_version, price_data
                )
                self.store.write_dataset_manifest(experiment_id, manifest)
        else:
            paths: dict[str, Any] = {}

        return {
            "experiment_id": experiment_id,
            "strategy_id": params.strategy_id,
            "run_id": run_id,
            "hybrid_params": params_dict,
            "metrics": tracker.metrics,
            "artefact_paths": paths,
            "backtest_run_id": bt_results.run_id,
            "n_steps": bt_results.n_steps,
        }

    def compare_hybrid_variants(
        self, exp_ids: "list[str]"
    ) -> "list[dict[str, Any]]":
        """Return a comparison table for hybrid strategy experiment variants.

        Extends :meth:`compare_experiments` by additionally surfacing the
        blend parameters (``momentum_weight``, ``news_weight``,
        ``vol_gate_threshold``, ``min_confidence``) from each experiment's
        stored ``hybrid_params.json`` (if present).

        Args:
            exp_ids: Ordered list of experiment IDs to compare.

        Returns:
            List of dicts, one per experiment, with keys:

            - ``experiment_id``
            - ``strategy_id``
            - ``momentum_weight``
            - ``news_weight``
            - ``vol_gate_threshold``
            - ``min_confidence``
            - ``sharpe``
            - ``cagr``
            - ``max_drawdown``
            - ``volatility``
            - ``win_rate``
            - ``turnover``
            - ``total_return``
            - ``n_steps``
            - ``config_hash``
            - ``run_timestamp``
        """
        base_rows = {r["experiment_id"]: r for r in self.compare_experiments(exp_ids)}

        rows = []
        for eid in exp_ids:
            base = base_rows.get(eid, {"experiment_id": eid})

            # Try to read hybrid_params.json; gracefully fall back to config doc
            hybrid_params: dict[str, Any] = {}
            try:
                hybrid_params = self.store.read_hybrid_params(eid)
            except FileNotFoundError:
                # Try extracting from config doc hybrid_params key
                try:
                    config_doc = self.store.read_config(eid)
                    hybrid_params = config_doc.get("metrics_payload", {}).get(
                        "hybrid_params", {}
                    ) or config_doc.get("hybrid_params", {})
                except FileNotFoundError:
                    hybrid_params = {}

            row: dict[str, Any] = {
                "experiment_id": eid,
                "strategy_id": base.get("strategy_id"),
                "momentum_weight": hybrid_params.get("momentum_weight"),
                "news_weight": hybrid_params.get("news_weight"),
                "vol_gate_threshold": hybrid_params.get("vol_gate_threshold"),
                "min_confidence": hybrid_params.get("min_confidence"),
                "sharpe": base.get("sharpe"),
                "cagr": base.get("cagr"),
                "max_drawdown": base.get("max_drawdown"),
                "volatility": base.get("volatility"),
                "win_rate": base.get("win_rate"),
                "turnover": base.get("turnover"),
                "total_return": base.get("total_return"),
                "n_steps": base.get("n_steps"),
                "config_hash": base.get("config_hash"),
                "run_timestamp": base.get("run_timestamp"),
            }
            if base.get("error"):
                row["error"] = base["error"]
            rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_from_backtest(results: Any) -> dict[str, Any]:
        """Extract standard metrics from a BacktestResults object.

        Supports both:
        - attribute-style results objects
        - dict / to_dict()-style backtest payloads

        Normalises common aliases so experiment views can always use:
        sharpe, cagr, volatility, max_drawdown, turnover, win_rate
        """
        payload: dict[str, Any] = {}

        # 1) Try summary_dict()
        summary_fn = getattr(results, "summary_dict", None)
        if callable(summary_fn):
            try:
                summary = summary_fn()
                if isinstance(summary, dict):
                    payload.update(summary)
            except Exception:
                pass

        # 2) Try to_dict()
        to_dict_fn = getattr(results, "to_dict", None)
        if callable(to_dict_fn):
            try:
                data = to_dict_fn()
                if isinstance(data, dict):
                    payload.update(data)
            except Exception:
                pass

        # 3) Fallback to common attributes
        for attr in (
            "sharpe",
            "sharpe_ratio",
            "cagr",
            "annualised_return",
            "volatility",
            "annualised_volatility",
            "max_drawdown",
            "turnover",
            "win_rate",
            "total_return",
            "calmar_ratio",
            "n_steps",
            "start_date",
            "end_date",
            "asset_class",
            "run_id",
            "symbols",
        ):
            val = getattr(results, attr, None)
            if val is not None and attr not in payload:
                payload[attr] = val

        metrics: dict[str, Any] = {}

        # Canonical fields used by experiment list/show/compare
        metrics["sharpe"] = payload.get("sharpe", payload.get("sharpe_ratio"))
        metrics["cagr"] = payload.get("cagr", payload.get("annualised_return"))
        metrics["volatility"] = payload.get(
            "volatility", payload.get("annualised_volatility")
        )
        metrics["max_drawdown"] = payload.get("max_drawdown")
        metrics["turnover"] = payload.get("turnover")
        metrics["win_rate"] = payload.get("win_rate")

        # Extra useful fields
        if payload.get("total_return") is not None:
            metrics["total_return"] = payload.get("total_return")
        if payload.get("calmar_ratio") is not None:
            metrics["calmar"] = payload.get("calmar_ratio")
        elif payload.get("calmar") is not None:
            metrics["calmar"] = payload.get("calmar")
        if payload.get("n_steps") is not None:
            metrics["n_steps"] = payload.get("n_steps")

        return {k: v for k, v in metrics.items() if v is not None}

    @staticmethod
    def _backtest_summary(results: Any) -> dict[str, Any]:
        """Convert a BacktestResults object to a plain dict."""
        # 1) Prefer summary_dict()
        summary_fn = getattr(results, "summary_dict", None)
        if callable(summary_fn):
            try:
                summary = summary_fn()
                if isinstance(summary, dict) and summary:
                    return summary
            except Exception:
                pass

        # 2) Fallback to to_dict()
        to_dict_fn = getattr(results, "to_dict", None)
        if callable(to_dict_fn):
            try:
                data = to_dict_fn()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        # 3) Final fallback: build dict from common attrs
        summary: dict[str, Any] = {}
        for attr in (
            "run_id",
            "asset_class",
            "start_date",
            "end_date",
            "n_steps",
            "symbols",
            "total_return",
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "extra",
            "snapshots",
        ):
            val = getattr(results, attr, None)
            if val is not None:
                summary[attr] = val
        return summary

    @staticmethod
    def _build_dataset_manifest(
        strategy_id: str,
        dataset_version: str,
        price_data: dict,
    ) -> dict[str, Any]:
        """Build a minimal dataset manifest from *price_data*."""
        symbol_info = {}
        for sym, df in price_data.items():
            try:
                import pandas as pd

                symbol_info[sym] = {
                    "rows": len(df),
                    "start": str(df.index.min()) if len(df) > 0 else None,
                    "end": str(df.index.max()) if len(df) > 0 else None,
                }
            except Exception:  # noqa: BLE001
                symbol_info[sym] = {"rows": 0}

        return {
            "strategy_id": strategy_id,
            "dataset_version": dataset_version,
            "symbols": list(price_data.keys()),
            "symbol_info": symbol_info,
        }
