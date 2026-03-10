"""Equity-related CLI commands.

These commands are STRICTLY SEPARATE from crypto commands.
A single invocation must never run both crypto and equity pipelines.

Available commands:
  equity-run        Run the full equity research pipeline end-to-end.
  equity-download   Download equity OHLCV data to local cache.
  equity-universe   Show equity universe statistics.
  equity-signals    Run equity signal agents on cached data.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _load_equity_cfg() -> dict[str, Any]:
    """Load and validate config, ensuring equities section is present.

    In test mode, a minimal equities config is injected so pipelines
    can be exercised without requiring config.yaml edits.
    """
    from spectraquant.config import get_config

    cfg = get_config()
    test_mode = os.getenv("SPECTRAQUANT_TEST_MODE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    equity_section = cfg.get("equities", cfg.get("equity", {}))
    if test_mode and not equity_section:
        cfg["equities"] = {
            "tickers": [],
            "prices_dir": "data/equities/prices",
            "reports_dir": "reports/equities",
        }
    return cfg


def register_equity_commands(commands: dict[str, Any]) -> None:
    """Register equity-related commands.

    Args:
        commands: Dictionary to register commands into.
    """
    commands["equity-run"] = lambda: cmd_equity_run(_load_equity_cfg())
    commands["equity-download"] = lambda: cmd_equity_download(_load_equity_cfg())
    commands["equity-universe"] = lambda: cmd_equity_universe(_load_equity_cfg())
    commands["equity-signals"] = lambda: cmd_equity_signals(_load_equity_cfg())


def cmd_equity_run(cfg: dict[str, Any]) -> None:
    """Run the full equity pipeline end-to-end."""
    from spectraquant.core.enums import RunMode
    from spectraquant.pipeline.equity_run import run_equity_pipeline

    mode_str = os.getenv("SPECTRAQUANT_RUN_MODE", "normal").lower()
    run_mode_map = {
        "normal": RunMode.NORMAL,
        "test": RunMode.TEST,
        "refresh": RunMode.REFRESH,
    }
    run_mode = run_mode_map.get(mode_str, RunMode.NORMAL)
    dry_run = os.getenv("SPECTRAQUANT_DRY_RUN", "").strip().lower() in {
        "1", "true", "yes", "on",
    }

    logger.info("Running equity pipeline mode=%s dry_run=%s", run_mode, dry_run)
    result = run_equity_pipeline(cfg=cfg, run_mode=run_mode, dry_run=dry_run)
    logger.info(
        "Equity pipeline complete: %d weights, %d blocked",
        len(result.get("weights", {})),
        len(result.get("blocked", [])),
    )


def cmd_equity_download(cfg: dict[str, Any]) -> None:
    """Download equity OHLCV data to local cache."""
    import sys

    from spectraquant.core.enums import RunMode
    from spectraquant.equities.ingestion.price_downloader import EquityPriceDownloader
    from spectraquant.equities.universe.equity_universe_builder import (
        EquityUniverseBuilder,
    )

    equity_cfg = cfg.get("equities", cfg.get("equity", {}))
    argv = sys.argv[2:]

    # Parse --symbols override from CLI args
    symbols: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--symbols" and i + 1 < len(argv):
            symbols = [s.strip() for s in argv[i + 1].split(",") if s.strip()]
            i += 2
        else:
            i += 1

    if not symbols:
        builder = EquityUniverseBuilder(config=equity_cfg.get("universe", equity_cfg))
        symbols = builder.build()

    logger.info("Downloading equity OHLCV for %d symbols", len(symbols))
    downloader = EquityPriceDownloader(
        config=cfg.get("data", {}),
        run_mode=RunMode.REFRESH,
        cache_dir=equity_cfg.get("prices_dir", "data/equities/prices"),
    )
    result = downloader.download(symbols)
    logger.info(
        "Download complete: %d/%d symbols loaded",
        len(result.symbols_loaded),
        len(result.symbols_requested),
    )


def cmd_equity_universe(cfg: dict[str, Any]) -> None:
    """Print equity universe statistics."""
    import json

    from spectraquant.equities.universe.equity_universe_builder import (
        EquityUniverseBuilder,
    )

    equity_cfg = cfg.get("equities", cfg.get("equity", {}))
    builder = EquityUniverseBuilder(config=equity_cfg.get("universe", equity_cfg))
    symbols = builder.build()

    stats = {
        "n_symbols": len(symbols),
        "symbols": symbols,
    }
    print(json.dumps(stats, indent=2))


def cmd_equity_signals(cfg: dict[str, Any]) -> None:
    """Run equity signal agents on cached OHLCV data."""
    import json
    from pathlib import Path

    import pandas as pd

    from spectraquant.equities.signals.breakout_agent import BreakoutAgent
    from spectraquant.equities.signals.mean_reversion_agent import MeanReversionAgent
    from spectraquant.equities.signals.momentum_agent import MomentumAgent
    from spectraquant.equities.signals.regime_agent import RegimeAgent
    from spectraquant.equities.signals.volatility_agent import VolatilityAgent
    from spectraquant.equities.universe.equity_universe_builder import (
        EquityUniverseBuilder,
    )

    equity_cfg = cfg.get("equities", cfg.get("equity", {}))
    cache_dir = Path(equity_cfg.get("prices_dir", "data/equities/prices"))

    builder = EquityUniverseBuilder(config=equity_cfg.get("universe", equity_cfg))
    symbols = builder.build()

    agents = [
        MomentumAgent(), MeanReversionAgent(), BreakoutAgent(),
        VolatilityAgent(), RegimeAgent(),
    ]

    results: dict[str, list] = {}
    for sym in symbols:
        safe = sym.replace("/", "_")
        parquet = cache_dir / f"{safe}.parquet"
        if not parquet.exists():
            logger.warning("No cached data for %r – skipping", sym)
            continue
        df = pd.read_parquet(parquet)
        outputs = [agent.run(df, symbol=sym).__dict__ for agent in agents]
        results[sym] = outputs

    print(json.dumps(results, indent=2, default=str))
