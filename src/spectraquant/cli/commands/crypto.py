"""Crypto-related CLI commands."""
from __future__ import annotations

from typing import Any


def _load_crypto_cfg() -> dict[str, Any]:
    """Load and validate config, ensuring crypto is enabled.

    When ``--test-mode`` is active (env ``SPECTRAQUANT_TEST_MODE=true``),
    crypto is implicitly enabled so the pipeline can be exercised without
    requiring the user to edit ``config.yaml``.

    Note: ``validate_runtime_defaults`` is intentionally NOT called here because
    it is an equity-focused validator that would log NSE tickers (a side-effect
    unrelated to crypto commands).
    """
    import os
    from spectraquant.config import get_config

    cfg = get_config()

    # In test-mode, auto-enable crypto so dry-run pipelines work
    test_mode = os.getenv("SPECTRAQUANT_TEST_MODE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    crypto_section = cfg.get("crypto", {})
    if test_mode and not crypto_section.get("enabled", False):
        cfg.setdefault("crypto", {})["enabled"] = True

    if not crypto_section.get("enabled", False):
        raise RuntimeError("Enable crypto.enabled in config.yaml")
    return cfg


def register_crypto_commands(commands: dict[str, Any]) -> None:
    """Register crypto-related commands.

    All registered handlers are zero-arg callables that load config
    internally so the CLI dispatcher can call ``commands[name]()``.

    Args:
        commands: Dictionary to register commands into.
    """
    commands["crypto-run"] = lambda: cmd_crypto_run(_load_crypto_cfg())
    commands["crypto-stream"] = lambda: cmd_crypto_stream(_load_crypto_cfg())
    commands["onchain-scan"] = lambda: cmd_onchain_scan(_load_crypto_cfg())
    commands["agents-run"] = lambda: cmd_agents_run(_load_crypto_cfg())
    commands["allocate"] = lambda: cmd_allocate(_load_crypto_cfg())
    commands["crypto-ingest-dataset"] = lambda: cmd_crypto_ingest_dataset(_load_crypto_cfg())


def cmd_crypto_stream(cfg: dict[str, Any]) -> None:
    """Start WebSocket stream and write candles to parquet."""
    import asyncio
    import logging

    logger = logging.getLogger(__name__)
    crypto_cfg = cfg.get("crypto", {})

    from spectraquant.crypto.exchange.coinbase_ws import CoinbaseWSClient

    client = CoinbaseWSClient(
        endpoint=crypto_cfg.get("endpoint", "ws-feed"),
        environment=crypto_cfg.get("environment", "production"),
        output_dir=crypto_cfg.get("prices_dir", "data/prices/crypto"),
    )
    symbols = crypto_cfg.get("symbols", ["BTC-USD", "ETH-USD"])
    client.subscribe(symbols, ["ticker", "matches"])

    logger.info("Starting crypto stream for %s", symbols)
    try:
        asyncio.run(client.start())
    except KeyboardInterrupt:
        logger.info("Stream stopped by user")


def cmd_onchain_scan(cfg: dict[str, Any]) -> None:
    """Collect on-chain data and compute features."""
    import logging
    from datetime import datetime, timezone
    from pathlib import Path

    logger = logging.getLogger(__name__)
    onchain_cfg = cfg.get("onchain_ai", {})

    from spectraquant.onchain.collectors.free_sources import collect_all
    from spectraquant.onchain.features import compute_onchain_features
    from spectraquant.onchain.anomaly import compute_anomaly_scores

    crypto_cfg = cfg.get("crypto", {})
    symbols = crypto_cfg.get("symbols", ["BTC-USD", "ETH-USD"])
    symbols = [s.split("-")[0] for s in symbols]

    raw = collect_all(symbols)
    if raw.empty:
        logger.warning("No on-chain data collected")
        return

    features = compute_onchain_features(raw)
    features = compute_anomaly_scores(
        features,
        threshold=onchain_cfg.get("anomaly_threshold", 2.5),
    )

    out_dir = Path(onchain_cfg.get("output_dir", "reports/onchain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"onchain_features_{ts}.parquet"
    features.to_parquet(out_path, engine="pyarrow")
    logger.info("Wrote %s (%d rows)", out_path, len(features))


def cmd_agents_run(cfg: dict[str, Any]) -> None:
    """Run all trading agents on latest data."""
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    from spectraquant.agents.registry import AgentRegistry

    crypto_cfg = cfg.get("crypto", {})
    prices_dir = Path(crypto_cfg.get("prices_dir", "data/prices/crypto"))

    import pandas as pd

    registry = AgentRegistry()
    all_signals: dict[str, list] = {}

    symbols = crypto_cfg.get("symbols", ["BTC-USD"])
    for sym_pair in symbols:
        for suffix in [f"{sym_pair}_1m", f"{sym_pair}_5m", sym_pair]:
            parquet = prices_dir / f"{suffix}.parquet"
            if parquet.exists():
                df = pd.read_parquet(parquet)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                sigs = registry.run_all(df, symbol_override=sym_pair.split("-")[0])
                for agent_name, sig_list in sigs.items():
                    all_signals.setdefault(agent_name, []).extend(sig_list)
                break

    total = sum(len(v) for v in all_signals.values())
    logger.info("Generated %d signals from %d agents", total, len(all_signals))


def cmd_allocate(cfg: dict[str, Any]) -> None:
    """Run portfolio allocation with latest signals."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Allocation command — use 'crypto-run' for full pipeline")


def cmd_crypto_run(cfg: dict[str, Any]) -> None:
    """Run the full crypto pipeline end-to-end."""
    import logging
    import os

    logger = logging.getLogger(__name__)
    logger.info("Running full crypto pipeline")

    dry_run = os.getenv("SPECTRAQUANT_DRY_RUN", "").strip().lower() in {
        "1", "true", "yes", "on",
    }

    from spectraquant.pipeline.crypto_run import run_crypto_pipeline

    result = run_crypto_pipeline(cfg=cfg, dry_run=dry_run)
    logger.info(
        "Pipeline complete: %d weights, %d agent signal groups",
        len(result.get("weights", [])),
        len(result.get("agent_signals", {})),
    )


def cmd_crypto_ingest_dataset(cfg: dict[str, Any]) -> None:
    """Ingest cryptocurrency_dataset.csv into parquet artefacts.

    Reads CLI arguments from sys.argv:
      --path <csv>          path to CSV file (required)
      --as-of <iso>         UTC timestamp for snapshot (default: now)
      --append-snapshot     append to existing snapshot (default: True)
      --no-append-snapshot  overwrite existing snapshot
    """
    import logging
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    logger = logging.getLogger(__name__)

    # ---- parse sys.argv ---------------------------------------------------
    argv = sys.argv[2:]  # skip command name
    csv_path: str | None = None
    as_of: datetime | None = None
    append_snapshot = True

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--path" and i + 1 < len(argv):
            csv_path = argv[i + 1]
            i += 2
        elif arg == "--as-of" and i + 1 < len(argv):
            try:
                as_of = datetime.fromisoformat(argv[i + 1].replace("Z", "+00:00"))
            except ValueError as exc:
                logger.error(
                    "Invalid --as-of value %r: %s. Expected ISO 8601 format, e.g. '2024-01-01T00:00:00+00:00'.",
                    argv[i + 1],
                    exc,
                )
                raise SystemExit(1) from exc
            i += 2
        elif arg == "--no-append-snapshot":
            append_snapshot = False
            i += 1
        elif arg == "--append-snapshot":
            append_snapshot = True
            i += 1
        else:
            i += 1

    if csv_path is None:
        # Try config default
        dataset_cfg = cfg.get("crypto_dataset", {})
        csv_path = dataset_cfg.get("path", "data/crypto/cryptocurrency_dataset.csv")

    dataset_cfg = cfg.get("crypto_dataset", {})
    data_dir = dataset_cfg.get("data_dir", "data/crypto")
    yfinance_overrides = dataset_cfg.get("yfinance_overrides", {})

    from spectraquant.crypto.dataset.ingest import ingest_crypto_dataset

    try:
        summary = ingest_crypto_dataset(
            csv_path=csv_path,
            as_of=as_of,
            append_snapshot=append_snapshot,
            data_dir=data_dir,
            yfinance_overrides=yfinance_overrides,
        )
        logger.info(
            "Ingestion complete — rows_read=%d, rows_kept=%d, "
            "duplicates_removed=%d, nulls_filled=%d",
            summary["rows_read"],
            summary["rows_kept"],
            summary["duplicates_removed"],
            summary["nulls_filled"],
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Provide the CSV path via --path or set crypto_dataset.path in config.yaml"
        )
        raise SystemExit(1) from exc
