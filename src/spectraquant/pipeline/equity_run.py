"""Equity pipeline runner.

Orchestrates the full equity research pipeline:
  universe → ingestion → signals → meta-policy → allocation → reporting

Key guarantees:
- Crypto symbols are rejected at entry (AssetClassLeakError).
- Zero OHLCV rows causes fail-loud EmptyOHLCVError via
  ``ohlcv_result.assert_ohlcv_available()`` after QA population.
- Every run writes a QA matrix and run manifest.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectraquant.core.enums import AssetClass, RunMode
from spectraquant.core.errors import EmptyOHLCVError
from spectraquant.equities.execution.paper_executor import EquityPaperExecutor
from spectraquant.equities.ingestion.price_downloader import EquityPriceDownloader
from spectraquant.equities.policy.allocator import EquityAllocator
from spectraquant.equities.policy.meta_policy import EquityMetaPolicy
from spectraquant.equities.reporting.reporter import EquityReporter, EquityRunReport
from spectraquant.equities.signals._base_agent import AgentOutput
from spectraquant.equities.signals.breakout_agent import BreakoutAgent
from spectraquant.equities.signals.mean_reversion_agent import MeanReversionAgent
from spectraquant.equities.signals.momentum_agent import MomentumAgent
from spectraquant.equities.signals.news_sentiment_agent import NewsSentimentAgent
from spectraquant.equities.signals.quality_agent import QualityAgent
from spectraquant.equities.signals.regime_agent import RegimeAgent
from spectraquant.equities.signals.volatility_agent import VolatilityAgent
from spectraquant.equities.universe.equity_universe_builder import (
    EquityUniverseBuilder,
)

logger = logging.getLogger(__name__)


def run_equity_pipeline(
    cfg: dict[str, Any],
    run_mode: RunMode = RunMode.NORMAL,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full equity pipeline.

    Args:
        cfg: Full pipeline configuration dict (``equities`` section is used).
        run_mode: NORMAL | TEST | REFRESH.
        dry_run: If True, skip execution and return early after allocation.

    Returns:
        Dict with keys: ``run_id``, ``weights``, ``signals``, ``qa``,
        ``blocked``, ``report_path``.

    Raises:
        AssetClassLeakError: If crypto symbols are in the equity universe.
        EmptyOHLCVError: If zero symbols return OHLCV data
            (raised via ``ohlcv_result.assert_ohlcv_available()``).
    """
    run_id = str(uuid.uuid4())[:8]
    logger.info("Starting equity pipeline run_id=%s mode=%s", run_id, run_mode)

    equity_cfg: dict[str, Any] = cfg.get("equities", cfg.get("equity", {}))
    data_cfg: dict[str, Any] = cfg.get("data", {})
    reports_dir = equity_cfg.get("reports_dir", "reports/equities")
    cache_dir = equity_cfg.get("prices_dir", data_cfg.get("prices_dir", "data/equities/prices"))

    # ------------------------------------------------------------------ 1. Universe
    universe_cfg = equity_cfg.get("universe", equity_cfg)
    builder = EquityUniverseBuilder(config=universe_cfg)
    yf_symbols = builder.build()
    logger.info("Equity universe: %d symbols", len(yf_symbols))

    # ------------------------------------------------------------------ 2. Ingestion
    downloader = EquityPriceDownloader(
        config=data_cfg,
        run_mode=run_mode,
        cache_dir=cache_dir,
    )
    ohlcv_result = downloader.download(yf_symbols)
    # Hard guard: abort if no OHLCV data is available after QA population.
    ohlcv_result.assert_ohlcv_available()

    # ------------------------------------------------------------------ 3. Signals
    agents = [
        MomentumAgent(),
        MeanReversionAgent(),
        BreakoutAgent(),
        VolatilityAgent(),
        RegimeAgent(),
        QualityAgent(),
        NewsSentimentAgent(),  # gracefully degrades when no news
    ]

    signals_by_symbol: dict[str, list[AgentOutput]] = {}
    for sym in yf_symbols:
        df = ohlcv_result.prices.get(sym)
        if df is None:
            continue
        outputs: list[AgentOutput] = []
        for agent in agents:
            out = agent.run(df, symbol=sym)
            outputs.append(out)
        signals_by_symbol[sym] = outputs

    # ------------------------------------------------------------------ 4. Policy
    policy = EquityMetaPolicy(
        min_confidence=equity_cfg.get("min_confidence", 0.15),
    )
    decisions = policy.run(signals_by_symbol)

    # ------------------------------------------------------------------ 5. Allocation
    allocator = EquityAllocator(
        max_weight=equity_cfg.get("max_weight", 0.20),
        min_signal_threshold=equity_cfg.get("min_signal_threshold", 0.05),
    )
    allocation = allocator.allocate(decisions)

    # ------------------------------------------------------------------ 6. Execution (paper)
    orders = []
    if not dry_run:
        executor = EquityPaperExecutor(
            slippage_bps=equity_cfg.get("slippage_bps", 5.0),
            transaction_cost_bps=equity_cfg.get("transaction_cost_bps", 10.0),
        )
        orders = executor.execute(allocation.target_weights)

    # ------------------------------------------------------------------ 7. Reporting
    report = EquityRunReport(
        run_id=run_id,
        symbols_requested=ohlcv_result.symbols_requested,
        symbols_loaded=ohlcv_result.symbols_loaded,
        symbols_failed=ohlcv_result.symbols_failed,
        qa_matrix=list(ohlcv_result.qa.values()),
        allocation_weights=allocation.target_weights,
        blocked_assets=allocation.blocked_assets,
        metadata={
            "mode": str(run_mode),
            "dry_run": dry_run,
            "n_agents": len(agents),
            "n_orders": len(orders),
        },
    )
    reporter = EquityReporter(reports_dir=reports_dir)
    report_path = reporter.write(report)

    logger.info(
        "Equity pipeline complete: %d weights, %d blocked, report=%s",
        len(allocation.target_weights),
        len(allocation.blocked_assets),
        report_path,
    )

    return {
        "run_id": run_id,
        "weights": allocation.target_weights,
        "blocked": allocation.blocked_assets,
        "signals": {sym: [o.__dict__ for o in outs] for sym, outs in signals_by_symbol.items()},
        "qa": ohlcv_result.qa,
        "report_path": str(report_path),
        "orders": [o.__dict__ for o in orders],
    }
