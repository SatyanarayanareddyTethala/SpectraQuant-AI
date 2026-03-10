"""Backtest CLI commands for SpectraQuant-AI-V3.

Available commands (registered as sub-commands of the ``backtest`` group):
  run   Run a walk-forward backtest over historical cached OHLCV data.
"""

from __future__ import annotations

from pathlib import Path

import typer

backtest_app = typer.Typer(
    name="backtest",
    help="Walk-forward backtesting commands.",
    no_args_is_help=True,
)


@backtest_app.command("run")
def backtest_run(
    asset_class: str = typer.Option(
        "crypto",
        "--asset-class",
        "-a",
        help="Asset class: crypto | equity",
    ),
    symbols: str = typer.Option(
        "", "--symbols", help="Comma-separated symbols (default: from config)"
    ),
    rebalance_freq: str = typer.Option(
        "W",
        "--rebalance-freq",
        "-f",
        help="Pandas offset alias: D (daily), W (weekly), ME (month-end), etc.",
    ),
    window_type: str = typer.Option(
        "expanding",
        "--window-type",
        "-w",
        help="Walk-forward window: expanding | rolling",
    ),
    lookback_periods: int = typer.Option(
        252,
        "--lookback-periods",
        help="Number of periods in rolling window (ignored for expanding)",
    ),
    min_periods: int = typer.Option(
        30,
        "--min-periods",
        help="Minimum in-sample bars required before executing a step",
    ),
    output_dir: str = typer.Option(
        "reports/backtest",
        "--output-dir",
        "-o",
        help="Directory to write backtest_results_<run_id>.json",
    ),
    run_id: str = typer.Option("bt", "--run-id", help="Run identifier"),
    config_dir: str = typer.Option(
        "", "--config-dir", help="Override config/v3/ directory"
    ),
    experiment_id: str = typer.Option(
        "",
        "--experiment-id",
        help="Experiment identifier (default: same as --run-id)",
    ),
    experiments_dir: str = typer.Option(
        "reports/experiments",
        "--experiments-dir",
        help="Directory to store experiment artifacts",
    ),
    persist_experiment: bool = typer.Option(
        True,
        "--persist-experiment/--no-persist-experiment",
        help="Persist this backtest into the experiment store",
    ),
    strategy: str = typer.Option(
        "",
        "--strategy",
        help="Registered strategy id to run in backtest (default: asset-class baseline)",
    ),
) -> None:
    """Run a walk-forward backtest over historical cached OHLCV data.

    Loads OHLCV data from the local cache, replays the production
    signal → meta-policy → allocation pipeline over the history, and
    writes a JSON results file with performance metrics.
    """
    from spectraquant_v3.backtest.engine import BacktestEngine
    from spectraquant_v3.core.cache import CacheManager
    from spectraquant_v3.core.config import get_crypto_config, get_equity_config
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import EmptyPriceDataError, SpectraQuantError
    from spectraquant_v3.experiments.experiment_manager import ExperimentManager

    # Validate asset class
    if asset_class not in ("crypto", "equity"):
        typer.echo(
            f"[backtest run] ERROR: --asset-class must be 'crypto' or 'equity', "
            f"got {asset_class!r}.",
            err=True,
        )
        raise typer.Exit(1)

    # Validate window type
    if window_type not in ("expanding", "rolling"):
        typer.echo(
            f"[backtest run] ERROR: --window-type must be 'expanding' or 'rolling', "
            f"got {window_type!r}.",
            err=True,
        )
        raise typer.Exit(1)

    # Validate optional strategy selection
    if strategy:
        from spectraquant_v3.strategies.loader import StrategyLoader  # noqa: PLC0415

        try:
            selected = StrategyLoader.load(strategy)
        except (KeyError, ValueError) as exc:
            typer.echo(f"[backtest run] ERROR: invalid --strategy: {exc}", err=True)
            raise typer.Exit(1)
        if selected.asset_class != asset_class:
            typer.echo(
                f"[backtest run] ERROR: strategy '{strategy}' is asset_class={selected.asset_class!r} "
                f"but --asset-class={asset_class!r}.",
                err=True,
            )
            raise typer.Exit(1)

    # Load config
    try:
        if asset_class == "crypto":
            cfg = get_crypto_config(config_dir or None)
        else:
            cfg = get_equity_config(config_dir or None)
    except FileNotFoundError as exc:
        typer.echo(f"[backtest run] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    # Resolve symbol list
    if symbols:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif asset_class == "crypto":
        sym_list = [s.upper() for s in cfg.get("crypto", {}).get("symbols", [])]
    else:
        sym_list = cfg.get("equities", {}).get("universe", {}).get("tickers", [])

    if not sym_list:
        typer.echo(
            "[backtest run] No symbols found. Pass --symbols or configure in YAML.",
            err=True,
        )
        raise typer.Exit(1)

    # Load OHLCV from cache
    if asset_class == "crypto":
        cache_dir = cfg.get("crypto", {}).get("prices_dir", "data/cache/crypto/prices")
    else:
        cache_dir = cfg.get(
            "equities", {}
        ).get("prices_dir", "data/cache/equities/prices")

    cache = CacheManager(cache_dir, run_mode=RunMode.TEST)
    price_data: dict = {}
    for sym in sym_list:
        try:
            df = cache.read_parquet(sym)
            if df is not None and not df.empty:
                price_data[sym] = df
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[backtest run] WARNING: could not load cache for '{sym}': {exc}")

    if not price_data:
        typer.echo(
            f"[backtest run] No cached OHLCV data found in '{cache_dir}'. "
            "Run the download command first.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"[backtest run] Running {window_type} backtest over "
        f"{len(price_data)} symbol(s) "
        f"(freq={rebalance_freq} min_periods={min_periods}) …"
    )

    try:
        resolved_strategy = strategy or (
            "crypto_momentum_v1" if asset_class == "crypto" else "equity_momentum_v1"
        )
        engine = BacktestEngine(
            cfg=cfg,
            asset_class=asset_class,
            price_data=price_data,
            strategy_id=resolved_strategy,
            rebalance_freq=rebalance_freq,
            window_type=window_type,
            lookback_periods=lookback_periods,
            min_in_sample_periods=min_periods,
            run_id=run_id,
        )
        results = engine.run()
    except EmptyPriceDataError as exc:
        typer.echo(f"[backtest run] ERROR: {exc}", err=True)
        raise typer.Exit(1)
    except SpectraQuantError as exc:
        typer.echo(f"[backtest run] PIPELINE ERROR: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(results.summary())

    written_path = results.write(output_dir)
    typer.echo(f"[backtest run] Results written to {written_path}")

    if persist_experiment:
        exp_id = experiment_id or run_id
        strategy_id = strategy or (
            "crypto_momentum_v1" if asset_class == "crypto" else "equity_momentum_v1"
        )

        try:
            manager = ExperimentManager(Path(experiments_dir))
            manager.run_experiment(
                experiment_id=exp_id,
                strategy_id=strategy_id,
                cfg=cfg,
                run_mode=RunMode.NORMAL,
                dry_run=False,
                price_data=price_data,
                market_data=None,
                dataset_version=run_id,
                run_id=run_id,
                project_root=str(Path.cwd()),
                backtest_results=results,
            )
            typer.echo(
                f"[backtest run] Experiment registered at "
                f"{Path(experiments_dir) / exp_id}"
            )
        except Exception as exc:  # noqa: BLE001
            typer.echo(
                "[backtest run] WARNING: backtest succeeded but experiment "
                f"registration failed: {exc}",
                err=True,
            )

    raise typer.Exit(0)