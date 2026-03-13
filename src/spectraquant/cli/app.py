"""Command-line application for SpectraQuant-AI.

This module provides the main entry point for the CLI application.
Commands are organized into separate modules under cli/commands/.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from spectraquant.cli.commands.data import register_data_commands
from spectraquant.cli.commands.model import register_model_commands
from spectraquant.cli.commands.portfolio import register_portfolio_commands
from spectraquant.cli.commands.analysis import register_analysis_commands
from spectraquant.cli.commands.universe import register_universe_commands
from spectraquant.cli.commands.crypto import register_crypto_commands
from spectraquant.cli.commands.equities import register_equity_commands
from spectraquant.core.diagnostics import run_summary
from spectraquant.core.perf import enforce_stage_budget

logger = logging.getLogger(__name__)

USAGE = (
    "Usage: spectraquant "
    "[download|news-scan|features|build-dataset|train|predict|signals|score|portfolio|execute|eval|retrain|refresh|doctor|health-check|"
    "release-check|promote-model|list-models|universe-stats|universe update-nse|universe stats|feature-pruning|model-compare|stress-test|regime-stress|"
    "explain-portfolio|compare-runs|"
    "crypto-run|crypto-stream|onchain-scan|agents-run|allocate|"
    "equity-run|equity-download|equity-universe|equity-signals] [--research] [--use-sentiment] [--test-mode] "
    "[--force-pass-tests] [--dry-run] [--universe \"nifty50,ftse100\"] [--verbose]"
)


def _print_usage() -> None:
    """Print CLI usage information."""
    print(USAGE)
    print("\nAvailable commands:")
    print("  Data commands:")
    print("    download         - Fetch market data from providers")
    print("    news-scan        - Scan for news articles")
    print("    features         - Compute OHLCV features")
    print("    build-dataset    - Build ML dataset")
    print("    refresh          - Refresh cached data")
    print("\n  Model commands:")
    print("    train            - Train prediction models")
    print("    predict          - Generate predictions")
    print("    retrain          - Auto-retrain models")
    print("    promote-model    - Promote model to production")
    print("    list-models      - List available models")
    print("\n  Portfolio commands:")
    print("    signals          - Generate trading signals")
    print("    score            - Compute alpha scores")
    print("    portfolio        - Construct portfolio")
    print("    execute          - Execute paper trading")
    print("\n  Analysis commands:")
    print("    eval             - Evaluate performance")
    print("    feature-pruning  - Analyze feature importance")
    print("    model-compare    - Compare model performance")
    print("    stress-test      - Run parameter sensitivity")
    print("    regime-stress    - Analyze regime performance")
    print("    explain-portfolio - Generate portfolio explanations")
    print("    compare-runs     - Compare multiple runs")
    print("\n  Universe commands:")
    print("    universe update-nse  - Update NSE universe")
    print("    universe stats       - Show universe statistics")
    print("\n  Crypto commands:")
    print("    crypto-run              - Run full crypto pipeline end-to-end")
    print("    crypto-stream           - Start WebSocket crypto data stream")
    print("    onchain-scan            - Collect on-chain data and features")
    print("    agents-run              - Run trading agents on latest data")
    print("    allocate                - Run portfolio allocation")
    print("    crypto-ingest-dataset   - Ingest cryptocurrency_dataset.csv into parquet")
    print("\n  Equity commands:")
    print("    equity-run              - Run full equity research pipeline end-to-end")
    print("    equity-download         - Download equity OHLCV data to cache")
    print("    equity-universe         - Show equity universe statistics")
    print("    equity-signals          - Run equity signal agents on cached data")
    print("\n  System commands:")
    print("    doctor           - Run diagnostic checks")
    print("    health-check     - Check system health")
    print("    release-check    - Verify release readiness")


def _parse_cli_overrides(args: list[str]) -> tuple[list[str], bool, bool, bool, bool, str | None, bool, bool]:
    """Parse CLI override flags."""
    use_sentiment = False
    test_mode = False
    force_pass_tests = False
    dry_run = False
    universe: str | None = None
    from_news = False
    cleaned = []
    verbose = False
    
    it = iter(args)
    for arg in it:
        if arg == "--use-sentiment":
            use_sentiment = True
            continue
        if arg == "--test-mode":
            test_mode = True
            continue
        if arg == "--force-pass-tests":
            force_pass_tests = True
            continue
        if arg == "--dry-run":
            dry_run = True
            continue
        if arg == "--from-news":
            from_news = True
            continue
        if arg == "--verbose":
            verbose = True
            continue
        if arg.startswith("--universe"):
            value = None
            if arg == "--universe":
                value = next(it, None)
            else:
                _, _, value = arg.partition("=")
            if value:
                universe = value
            continue
        cleaned.append(arg)
    
    return cleaned, use_sentiment, test_mode, force_pass_tests, dry_run, universe, from_news, verbose


def _load_config() -> dict:
    """Load configuration from config.yaml."""
    from spectraquant.config import get_config, validate_runtime_defaults
    
    config = get_config()
    validate_runtime_defaults(config)
    return config


def _write_dashboard_manifest(command: str, config: dict) -> Path:
    """Write dashboard manifest for command execution."""
    from datetime import datetime
    from spectraquant.cli.main import _update_latest_manifest
    
    manifest_entries = {
        "command": command,
        "timestamp": datetime.now().isoformat(),
        "version": config.get("version", "unknown"),
    }
    return _update_latest_manifest(command, manifest_entries)


def main() -> None:
    """Main CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    commands: dict[str, callable] = {}
    
    register_data_commands(commands)
    register_model_commands(commands)
    register_portfolio_commands(commands)
    register_analysis_commands(commands)
    register_universe_commands(commands)
    register_crypto_commands(commands)
    register_equity_commands(commands)
    
    args = sys.argv[1:]
    args, use_sentiment, test_mode, force_pass_tests, dry_run, universe, from_news, verbose = _parse_cli_overrides(args)
    
    if "-h" in args or "--help" in args:
        _print_usage()
        return
    
    if "--research" in args:
        os.environ["SPECTRAQUANT_RESEARCH_MODE"] = "true"
        args = [arg for arg in args if arg != "--research"]
    
    if use_sentiment:
        os.environ["SPECTRAQUANT_USE_SENTIMENT"] = "true"
    if test_mode:
        os.environ["SPECTRAQUANT_TEST_MODE"] = "true"
    if force_pass_tests:
        os.environ["SPECTRAQUANT_FORCE_PASS_TESTS"] = "true"
    if dry_run:
        os.environ["SPECTRAQUANT_DRY_RUN"] = "true"
    if universe:
        os.environ["SPECTRAQUANT_UNIVERSE"] = universe
    if from_news:
        os.environ["SPECTRAQUANT_FROM_NEWS"] = "true"
    if verbose:
        os.environ["SPECTRAQUANT_VERBOSE"] = "true"
        logging.getLogger().setLevel(logging.DEBUG)
    
    if len(args) < 1:
        logger.error(USAGE)
        _print_usage()
        return
    
    command = args[0]
    
    if command == "universe":
        if len(args) < 2:
            logger.error("Usage: spectraquant universe [update-nse|stats]")
            return
        sub = args[1].strip().lower()
        if sub == "update-nse":
            command = "universe-update-nse"
        elif sub == "stats":
            command = "universe-nse-stats"
        else:
            logger.error("Unknown universe subcommand: %s", sub)
            return
    
    if command not in commands:
        logger.error(USAGE)
        _print_usage()
        return
    
    logger.info("Running command: %s", command)
    
    pipeline_commands = {
        "download",
        "news-scan",
        "universe-stats",
        "universe-update-nse",
        "universe-nse-stats",
        "features",
        "build-dataset",
        "train",
        "predict",
        "signals",
        "score",
        "portfolio",
        "execute",
        "refresh",
        "eval",
        "retrain",
        "health-check",
        "release-check",
        "crypto-run",
        "crypto-stream",
        "onchain-scan",
        "agents-run",
        "allocate",
        "crypto-ingest-dataset",
        "equity-run",
        "equity-download",
        "equity-universe",
        "equity-signals",
    }

    # Crypto commands manage their own lifecycle and do not use the equity
    # run-manifest infrastructure (which expects reports/run/*/manifest.json).
    _CRYPTO_COMMANDS = {
        "crypto-run",
        "crypto-stream",
        "onchain-scan",
        "agents-run",
        "allocate",
        "crypto-ingest-dataset",
    }

    # Equity commands manage their own lifecycle too.
    _EQUITY_COMMANDS = {
        "equity-run",
        "equity-download",
        "equity-universe",
        "equity-signals",
    }

    # Commands that run indefinitely must not be killed by enforce_stage_budget
    _NO_BUDGET_COMMANDS = {"crypto-stream"}

    with run_summary(command):
        try:
            manifest_config: dict | None = None
            if command in pipeline_commands:
                if command == "release-check" and os.getenv("SPECTRAQUANT_RESEARCH_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
                    commands[command]()
                elif command in _NO_BUDGET_COMMANDS:
                    config = _load_config()
                    commands[command]()
                    manifest_config = config
                else:
                    config = _load_config()
                    with enforce_stage_budget(command, config):
                        commands[command]()
                    manifest_config = config
            else:
                commands[command]()

            # Crypto and equity commands manage their own manifests.
            if (
                command in pipeline_commands
                and command not in _CRYPTO_COMMANDS
                and command not in _EQUITY_COMMANDS
                and command != "release-check"
                and manifest_config is not None
            ):
                manifest_path = _write_dashboard_manifest(command, manifest_config)
                from spectraquant.core.diagnostics import record_output
                record_output(str(manifest_path))
                logger.info("Run manifest written to %s", manifest_path)
        except BaseException as exc:
            logger.error("Command %s failed: %s", command, exc)
            raise


if __name__ == "__main__":
    main()
