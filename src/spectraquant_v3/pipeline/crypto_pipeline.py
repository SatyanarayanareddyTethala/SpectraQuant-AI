"""Crypto pipeline orchestrator for SpectraQuant-AI-V3.

Stages:
  universe → features → signals → meta_policy → allocation → reporting

Ingestion (live network calls) is intentionally out-of-scope here; the
pipeline expects pre-cached OHLCV DataFrames to be supplied by the caller
or loaded from the cache via :class:`~spectraquant_v3.core.cache.CacheManager`.

This module must never import from ``spectraquant_v3.equities``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from spectraquant_v3.core.enums import AssetClass, RunMode, RunStage, RunStatus
from spectraquant_v3.core.context import RunContext
from spectraquant_v3.core.schema import QARow
from spectraquant_v3.core.config import validate_config, validate_crypto_config
from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine
from spectraquant_v3.strategies.agents.registry import AgentRegistry
from spectraquant_v3.crypto.symbols.registry import build_registry_from_config
from spectraquant_v3.crypto.universe.builder import CryptoUniverseBuilder
from spectraquant_v3.strategies.agents.runner import run_signal_agent
from spectraquant_v3.strategies.loader import StrategyLoader
from spectraquant_v3.pipeline.meta_policy import MetaPolicy
from spectraquant_v3.pipeline.reporter import PipelineReporter


def run_crypto_pipeline(
    cfg: dict[str, Any],
    run_mode: RunMode = RunMode.NORMAL,
    dry_run: bool = False,
    price_data: dict[str, pd.DataFrame] | None = None,
    market_data: dict[str, dict[str, Any]] | None = None,
    dataset: dict[str, pd.DataFrame] | None = None,
    run_id: str | None = None,
    project_root: str | None = None,
) -> dict[str, Any]:
    """Execute the full crypto research pipeline.

    Args:
        cfg:          Merged crypto configuration dict (``get_crypto_config()``).
        run_mode:     Cache behaviour (NORMAL / TEST / REFRESH).
        dry_run:      When True, skip all writes and network calls.
        price_data:   Pre-loaded OHLCV DataFrames keyed by canonical symbol.
                      When *None* the pipeline still builds the universe and
                      emits NO_SIGNAL rows for all symbols.
        market_data:  Optional market metrics (market_cap_usd, volume_24h_usd,
                      age_days) per symbol for universe quality gates.
        dataset:      Optional pre-computed per-symbol feature dataset. When
                      provided, this is consumed directly by signal agents and
                      supersedes feature-engine output.
        run_id:       Override the auto-generated run ID.
        project_root: Override repo-root discovery (useful in tests).

    Returns:
        Dict with keys: ``run_id``, ``status``, ``universe``, ``signals``,
        ``allocations``, ``artefact_paths``.
    """
    validate_config(cfg)
    validate_crypto_config(cfg)

    with RunContext.create(
        asset_class=AssetClass.CRYPTO,
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
        universe_builder = CryptoUniverseBuilder(
            cfg=cfg,
            registry=registry,
            run_id=ctx.run_id,
        )
        universe_artifact = universe_builder.build(market_data=market_data)
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
                asset_class=AssetClass.CRYPTO.value,
                has_ohlcv=df is not None and not df.empty,
                rows_loaded=len(df) if df is not None else 0,
                provider_used=cfg.get("crypto", {}).get(
                    "primary_ohlcv_provider", "ccxt"
                ),
            )
            ctx.qa_matrix.add(row)

        # Hard guard: if price_data was supplied but ALL symbols are missing
        # OHLCV, abort – this indicates a provider or cache failure.
        if price_data:
            ctx.qa_matrix.assert_ohlcv_available()

        qa_path: str | None = None
        if not dry_run:
            qa_path = str(ctx.write_qa_matrix())

        ctx.mark_stage_ok(RunStage.INGESTION.value)

        # ------------------------------------------------------------------
        # Stage 4: Feature engineering
        # ------------------------------------------------------------------
        feature_engine = CryptoFeatureEngine.from_config(cfg)
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
        strategy_id = str(cfg.get("_strategy_id", "crypto_momentum_v1"))
        primary_agent = strategy_id
        # Prefer _agents injected by StrategyLoader.build_pipeline_config (registry-validated).
        # Fall back to cfg["strategies"][strategy_id]["agents"] for backwards compatibility
        # when the pipeline is called directly without going through StrategyLoader.
        agents = (
            cfg.get("_agents")
            or cfg.get("strategies", {}).get(strategy_id, {}).get("agents", [])
        )
        if agents:
            primary_agent = str(agents[0])

        signal_agent = AgentRegistry.build_from_config(
            primary_agent, cfg, run_id=ctx.run_id
        )

        if feature_map:
            signals = run_signal_agent(signal_agent, feature_map, as_of=as_of)
        else:
            # No price data – emit NO_SIGNAL for every universe symbol
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
        from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry

        defn = StrategyLoader.load(strategy_id)
        import math
        vol_map = {
            sym: float(v)
            for sym in feature_map
            if "vol_realised" in feature_map[sym].columns
            and not feature_map[sym]["vol_realised"].empty
            for v in [feature_map[sym]["vol_realised"].iloc[-1]]
            if math.isfinite(float(v)) and float(v) > 0
        }
        allocator_cls = AllocatorRegistry.get(defn.allocator)
        allocator = allocator_cls.from_config(cfg, run_id=ctx.run_id)
        allocations = allocator.allocate_decisions(decisions, vol_map=vol_map or None)

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
                asset_class=AssetClass.CRYPTO.value,
            )
            written = reporter.write_all(
                universe_symbols=universe_symbols,
                signals=signals,
                decisions=decisions,
                allocations=allocations,
            )
            artefact_paths = {k: str(v) for k, v in written.items()}
            if qa_path is not None:
                artefact_paths["qa_matrix"] = qa_path

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
