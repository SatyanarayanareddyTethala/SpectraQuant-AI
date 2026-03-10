"""Equity pipeline orchestrator for SpectraQuant-AI-V3.

Stages:
  universe → features → signals → meta_policy → allocation → reporting

This module must never import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from spectraquant_v3.core.enums import AssetClass, RunMode, RunStage, RunStatus
from spectraquant_v3.core.context import RunContext
from spectraquant_v3.core.schema import QARow
from spectraquant_v3.equities.features.engine import EquityFeatureEngine
from spectraquant_v3.equities.signals.momentum import EquityMomentumAgent
from spectraquant_v3.equities.symbols.registry import build_registry_from_config
from spectraquant_v3.equities.universe.builder import EquityUniverseBuilder
from spectraquant_v3.pipeline.allocator import Allocator
from spectraquant_v3.pipeline.meta_policy import MetaPolicy
from spectraquant_v3.pipeline.reporter import PipelineReporter


def run_equity_pipeline(
    cfg: dict[str, Any],
    run_mode: RunMode = RunMode.NORMAL,
    dry_run: bool = False,
    price_data: dict[str, pd.DataFrame] | None = None,
    market_data: dict[str, dict[str, Any]] | None = None,
    dataset: dict[str, pd.DataFrame] | None = None,
    run_id: str | None = None,
    project_root: str | None = None,
) -> dict[str, Any]:
    """Execute the full equity research pipeline.

    Args:
        cfg:          Merged equity configuration dict (``get_equity_config()``).
        run_mode:     Cache behaviour (NORMAL / TEST / REFRESH).
        dry_run:      When True, skip all writes and network calls.
        price_data:   Pre-loaded OHLCV DataFrames keyed by canonical symbol.
        market_data:  Optional quality-gate metrics per symbol.
        dataset:      Optional pre-computed per-symbol feature dataset.
                      When provided, it is used directly instead of deriving
                      features from ``price_data``.
        run_id:       Override the auto-generated run ID.
        project_root: Override repo-root discovery (useful in tests).

    Returns:
        Dict with keys: ``run_id``, ``status``, ``universe``, ``signals``,
        ``allocations``, ``artefact_paths``.
    """
    with RunContext.create(
        asset_class=AssetClass.EQUITY,
        run_mode=run_mode,
        config=cfg,
        run_id=run_id,
        project_root=project_root,
    ) as ctx:
        # ------------------------------------------------------------------
        # Stage 1: Symbol registry
        # ------------------------------------------------------------------
        registry = build_registry_from_config(cfg)

        # ------------------------------------------------------------------
        # Stage 2: Universe
        # ------------------------------------------------------------------
        universe_builder = EquityUniverseBuilder(
            cfg=cfg,
            registry=registry,
            run_id=ctx.run_id,
        )
        universe_artifact = universe_builder.build(price_data=market_data)
        universe_symbols = universe_artifact.included_symbols

        if not dry_run:
            universe_artifact.write(ctx.paths.run_dir)

        ctx.mark_stage_ok(
            RunStage.UNIVERSE.value,
            included=len(universe_symbols),
            excluded=len(universe_artifact.excluded_symbols),
        )

        # ------------------------------------------------------------------
        # Stage 3: QA matrix population
        # ------------------------------------------------------------------
        from datetime import datetime, timezone
        as_of = datetime.now(timezone.utc).isoformat()

        price_data = price_data or {}
        for sym in universe_symbols:
            df = price_data.get(sym)
            row = QARow(
                run_id=ctx.run_id,
                as_of=as_of,
                canonical_symbol=sym,
                asset_class=AssetClass.EQUITY.value,
                has_ohlcv=df is not None and not df.empty,
                rows_loaded=len(df) if df is not None else 0,
                provider_used=cfg.get("equities", {}).get(
                    "primary_ohlcv_provider", "yfinance"
                ),
            )
            ctx.qa_matrix.add(row)

        # Hard guard: if price_data was supplied but ALL symbols are missing
        # OHLCV, abort – this indicates a provider or cache failure.
        if price_data:
            ctx.qa_matrix.assert_ohlcv_available()

        if not dry_run:
            ctx.write_qa_matrix()

        ctx.mark_stage_ok(RunStage.INGESTION.value)

        # ------------------------------------------------------------------
        # Stage 4: Feature engineering
        # ------------------------------------------------------------------
        feature_engine = EquityFeatureEngine.from_config(cfg)
        feature_map: dict[str, pd.DataFrame] = {}
        if dataset:
            feature_map = dict(dataset)
        elif price_data:
            feature_map = feature_engine.transform_many(price_data)

        ctx.mark_stage_ok(
            RunStage.FEATURES.value,
            symbols_with_features=len(feature_map),
        )

        # ------------------------------------------------------------------
        # Stage 5: Signals
        # ------------------------------------------------------------------
        signal_agent = EquityMomentumAgent.from_config(cfg, run_id=ctx.run_id)

        if feature_map:
            signals = signal_agent.evaluate_many(feature_map, as_of=as_of)
        else:
            import pandas as _pd
            signals = [
                signal_agent.evaluate(sym, _pd.DataFrame(), as_of=as_of)
                for sym in universe_symbols
            ]

        ctx.mark_stage_ok(
            RunStage.SIGNALS.value,
            ok=sum(1 for s in signals if s.status == "OK"),
            no_signal=sum(1 for s in signals if s.status == "NO_SIGNAL"),
            error=sum(1 for s in signals if s.status == "ERROR"),
        )

        # ------------------------------------------------------------------
        # Stage 6: Meta-policy
        # ------------------------------------------------------------------
        meta_policy = MetaPolicy.from_config(cfg)
        decisions = meta_policy.run(signals)

        ctx.mark_stage_ok(
            RunStage.META_POLICY.value,
            passed=sum(1 for d in decisions if d.passed),
            blocked=sum(1 for d in decisions if not d.passed),
        )

        # ------------------------------------------------------------------
        # Stage 7: Allocation
        # ------------------------------------------------------------------
        allocator = Allocator.from_config(cfg, run_id=ctx.run_id)
        import math
        vol_map = {
            sym: float(v)
            for sym in feature_map
            if "vol_realised" in feature_map[sym].columns
            and not feature_map[sym]["vol_realised"].empty
            for v in [feature_map[sym]["vol_realised"].iloc[-1]]
            if math.isfinite(float(v)) and float(v) > 0
        }
        allocations = allocator.allocate(decisions, vol_map=vol_map or None)

        ctx.mark_stage_ok(
            RunStage.ALLOCATION.value,
            active=sum(1 for a in allocations if not a.blocked),
        )

        # ------------------------------------------------------------------
        # Stage 8: Reporting
        # ------------------------------------------------------------------
        artefact_paths: dict[str, str] = {}
        if not dry_run:
            reporter = PipelineReporter(
                run_id=ctx.run_id,
                output_dir=ctx.paths.run_dir,
                asset_class=AssetClass.EQUITY.value,
            )
            written = reporter.write_all(
                universe_symbols=universe_symbols,
                signals=signals,
                decisions=decisions,
                allocations=allocations,
            )
            artefact_paths = {k: str(v) for k, v in written.items()}

        ctx.mark_stage_ok(RunStage.REPORTING.value)
        ctx.manifest.mark_complete(RunStatus.SUCCESS)

        return {
            "run_id": ctx.run_id,
            "status": RunStatus.SUCCESS.value,
            "universe": universe_symbols,
            "signals": signals,
            "decisions": decisions,
            "allocations": allocations,
            "artefact_paths": artefact_paths,
        }

