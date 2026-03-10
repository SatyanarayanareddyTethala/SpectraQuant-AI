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
