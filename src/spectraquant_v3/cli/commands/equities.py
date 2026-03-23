"""Equity CLI commands for SpectraQuant-AI-V3.

Available commands (registered as sub-commands of the ``equity`` group):
  run        Run the full equity research pipeline end-to-end.
  download   Download equity OHLCV data to local cache.
  universe   Show the equity universe after quality gating.
  signals    Run equity signal agents on cached data.
"""

from __future__ import annotations

import typer

equity_app = typer.Typer(
    name="equity",
    help=(
        "Equity pipeline commands (India/NSE-first). "
        "Use canonical NSE tickers like INFY.NS,TCS.NS or "
        "configure equities.universe.tickers_file for NSE CSV universes."
    ),
    no_args_is_help=True,
)


@equity_app.command("run")
def equity_run(
    mode: str = typer.Option("normal", "--mode", "-m", help="Run mode: normal | test | refresh"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip writes and network calls"),
    config_dir: str = typer.Option("", "--config-dir", help="Override config/v3/ directory"),
) -> None:
    """Run the full equity research pipeline end-to-end.

    Stages: universe → ingestion → features → signals →
            meta_policy → allocation → execution → reporting

    NSE-first happy path:
    - configure ``equities.universe.tickers`` with ``.NS`` tickers, or
    - configure ``equities.universe.tickers_file`` pointing to an NSE CSV
      with a ``SYMBOL`` column; bare NSE symbols are canonicalized to ``.NS``.
    """
    from spectraquant_v3.core.config import get_equity_config
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import SpectraQuantError, UniverseValidationError
    from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

    try:
        run_mode = RunMode(mode.lower())
    except ValueError:
        typer.echo(f"[equity run] ERROR: invalid --mode={mode!r}. Use normal|test|refresh.", err=True)
        raise typer.Exit(1)

    try:
        cfg = get_equity_config(config_dir or None)
    except FileNotFoundError as exc:
        typer.echo(f"[equity run] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    # Inject hybrid universe symbols when a universe file is configured.
    universe_file = cfg.get("universe", {}).get("file", "")
    if universe_file:
        from spectraquant_v3.core.universe_loader import inject_universe_into_config

        try:
            cfg, _universe = inject_universe_into_config(cfg, universe_file)
            equity_count = len(_universe.get("equities", []))
            typer.echo(
                f"[equity run] Loaded hybrid universe from {universe_file!r} "
                f"({equity_count} equity symbols)"
            )
        except (FileNotFoundError, UniverseValidationError) as exc:
            typer.echo(f"[equity run] Universe ERROR: {exc}", err=True)
            raise typer.Exit(1)

    typer.echo(f"[equity run] mode={mode} dry_run={dry_run} – starting pipeline …")
    try:
        result = run_equity_pipeline(cfg, run_mode=run_mode, dry_run=dry_run)
        typer.echo(
            f"[equity run] completed  status={result['status']} "
            f"universe={len(result['universe'])} symbols"
        )
    except SpectraQuantError as exc:
        typer.echo(f"[equity run] PIPELINE ERROR: {exc}", err=True)
        raise typer.Exit(1)


@equity_app.command("download")
def equity_download(
    symbols: str = typer.Option("", "--symbols", help="Comma-separated ticker symbols (e.g. INFY.NS,TCS.NS)"),
    mode: str = typer.Option("refresh", "--mode", "-m", help="Run mode: normal | refresh"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Max parallel downloads"),
    config_dir: str = typer.Option("", "--config-dir", help="Override config/v3/ directory"),
) -> None:
    """Download equity OHLCV data to the local parquet cache (async batch).

    Symbols can be provided via ``--symbols`` or default to the universe
    defined in ``equities.yaml``.  Downloads run concurrently using the
    async ingestion engine with bounded concurrency and per-symbol retry.
    """
    from spectraquant_v3.core.cache import CacheManager
    from spectraquant_v3.core.config import get_equity_config
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import SpectraQuantError
    from spectraquant_v3.equities.ingestion.ohlcv_loader import EquityOHLCVLoader
    from spectraquant_v3.equities.symbols.registry import build_registry_from_config

    try:
        run_mode = RunMode(mode.lower())
    except ValueError:
        typer.echo(
            f"[equity download] ERROR: invalid --mode={mode!r}. Use normal|refresh.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        cfg = get_equity_config(config_dir or None)
    except FileNotFoundError as exc:
        typer.echo(f"[equity download] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    if symbols:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        sym_list = cfg.get("equities", {}).get("universe", {}).get("tickers", [])

    if not sym_list:
        typer.echo(
            "[equity download] No symbols found – pass --symbols or add tickers to equities.yaml.",
            err=True,
        )
        raise typer.Exit(1)

    cache_dir = cfg.get("equities", {}).get("prices_dir", "data/cache/equities/prices")
    cache = CacheManager(cache_dir, run_mode=run_mode)

    try:
        registry = build_registry_from_config(cfg)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[equity download] ERROR building registry: {exc}", err=True)
        raise typer.Exit(1)

    loader = EquityOHLCVLoader.from_config(cfg, cache=cache, registry=registry, run_mode=run_mode)

    typer.echo(
        f"[equity download] Downloading {len(sym_list)} symbol(s) "
        f"(mode={mode}, concurrency={concurrency}) …"
    )
    try:
        import asyncio  # noqa: PLC0415

        results = asyncio.run(
            loader.load_many_async(sym_list, concurrency=concurrency)
        )
    except SpectraQuantError as exc:
        typer.echo(f"[equity download] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    ok = [s for s in sym_list if s in results]
    failed = [s for s in sym_list if s not in results]

    typer.echo(
        f"[equity download] Done: {len(ok)} succeeded, {len(failed)} failed."
    )
    if failed:
        typer.echo(f"[equity download] Failed symbols: {', '.join(failed)}", err=True)
        raise typer.Exit(2)
    raise typer.Exit(0)


@equity_app.command("universe")
def equity_universe(
    config_dir: str = typer.Option("", "--config-dir", help="Override config/v3/ directory"),
) -> None:
    """Print the equity universe after applying quality gates."""
    from spectraquant_v3.core.config import get_equity_config
    from spectraquant_v3.core.errors import SpectraQuantError
    from spectraquant_v3.equities.symbols.registry import resolve_equity_tickers_from_config

    try:
        cfg = get_equity_config(config_dir or None)
        tickers = resolve_equity_tickers_from_config(cfg)
    except SpectraQuantError as exc:
        typer.echo(f"[equity universe] ERROR: {exc}", err=True)
        raise typer.Exit(1)
    except FileNotFoundError as exc:
        typer.echo(f"[equity universe] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    if not tickers:
        typer.echo(
            "[equity universe] No tickers resolved. "
            "Configure equities.universe.tickers or equities.universe.tickers_file.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"[equity universe] Resolved {len(tickers)} canonical equity tickers:")
    for ticker in tickers:
        typer.echo(f"  - {ticker}")
    raise typer.Exit(0)


@equity_app.command("signals")
def equity_signals(
    symbols: str = typer.Option("", "--symbols", help="Comma-separated ticker symbols"),
    config_dir: str = typer.Option("", "--config-dir", help="Override config/v3/ directory"),
) -> None:
    """Run equity signal agents on cached OHLCV data."""
    typer.echo(
        f"[equity signals] symbols={symbols or 'from config'}  "
        "⚠  Not yet implemented – scaffold only."
    )
    raise typer.Exit(0)
