"""Crypto CLI commands for SpectraQuant-AI-V3.

Available commands (registered as sub-commands of the ``crypto`` group):
  run        Run the full crypto research pipeline end-to-end.
  download   Download crypto OHLCV data to local cache.
  universe   Show the crypto universe after quality gating.
  signals    Run crypto signal agents on cached data.
"""

from __future__ import annotations

import typer

crypto_app = typer.Typer(
    name="crypto",
    help="Crypto pipeline commands (CCXT / Binance / Coinbase / Kraken).",
    no_args_is_help=True,
)


@crypto_app.command("run")
def crypto_run(
    mode: str = typer.Option("normal", "--mode", "-m", help="Run mode: normal | test | refresh"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip writes and network calls"),
    config_dir: str = typer.Option("", "--config-dir", help="Override config/v3/ directory"),
) -> None:
    """Run the full crypto research pipeline end-to-end.

    Stages: universe → ingestion → features → signals →
            meta_policy → allocation → execution → reporting
    """
    from spectraquant_v3.core import config as config_mod
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import SpectraQuantError, UniverseValidationError
    from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

    try:
        run_mode = RunMode(mode.lower())
    except ValueError:
        typer.echo(f"[crypto run] ERROR: invalid --mode={mode!r}. Use normal|test|refresh.", err=True)
        raise typer.Exit(1)

    try:
        cfg = config_mod.get_crypto_config(config_dir or None)
    except FileNotFoundError as exc:
        typer.echo(f"[crypto run] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    # Inject hybrid universe symbols when a universe file is configured.
    universe_file = cfg.get("universe", {}).get("file", "")
    if universe_file:
        from spectraquant_v3.core.universe_loader import inject_universe_into_config

        try:
            cfg, _universe = inject_universe_into_config(cfg, universe_file)
            crypto_count = len(_universe.get("crypto", []))
            typer.echo(
                f"[crypto run] Loaded hybrid universe from {universe_file!r} "
                f"({crypto_count} crypto symbols)"
            )
        except (FileNotFoundError, UniverseValidationError) as exc:
            typer.echo(f"[crypto run] Universe ERROR: {exc}", err=True)
            raise typer.Exit(1)

    typer.echo(f"[crypto run] mode={mode} dry_run={dry_run} – starting pipeline …")
    try:
        result = run_crypto_pipeline(cfg, run_mode=run_mode, dry_run=dry_run)
        typer.echo(
            f"[crypto run] completed  status={result['status']} "
            f"universe={len(result['universe'])} symbols"
        )
    except SpectraQuantError as exc:
        typer.echo(f"[crypto run] PIPELINE ERROR: {exc}", err=True)
        raise typer.Exit(1)


@crypto_app.command("download")
def crypto_download(
    symbols: str = typer.Option("", "--symbols", help="Comma-separated canonical symbols (e.g. BTC,ETH,SOL)"),
    mode: str = typer.Option("refresh", "--mode", "-m", help="Run mode: normal | refresh"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Max parallel downloads"),
    config_dir: str = typer.Option("", "--config-dir", help="Override config/v3/ directory"),
) -> None:
    """Download crypto OHLCV data to the local parquet cache (async batch).

    Symbols can be provided via ``--symbols`` or default to the universe
    defined in ``crypto.yaml``.  Downloads run concurrently using the
    async ingestion engine with bounded concurrency and per-symbol retry.
    """
    from spectraquant_v3.core.cache import CacheManager
    from spectraquant_v3.core import config as config_mod
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import SpectraQuantError
    from spectraquant_v3.crypto.ingestion.ohlcv_loader import CryptoOHLCVLoader
    from spectraquant_v3.crypto.symbols.registry import build_registry_from_config

    try:
        run_mode = RunMode(mode.lower())
    except ValueError:
        typer.echo(
            f"[crypto download] ERROR: invalid --mode={mode!r}. Use normal|refresh.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        cfg = config_mod.get_crypto_config(config_dir or None)
    except FileNotFoundError as exc:
        typer.echo(f"[crypto download] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    if symbols:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        sym_list = [s.upper() for s in cfg.get("crypto", {}).get("symbols", [])]

    if not sym_list:
        typer.echo("[crypto download] No symbols found – pass --symbols or add symbols to crypto.yaml.", err=True)
        raise typer.Exit(1)

    cache_dir = cfg.get("crypto", {}).get("prices_dir", "data/cache/crypto/prices")
    cache = CacheManager(cache_dir, run_mode=run_mode)

    try:
        registry = build_registry_from_config(cfg)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[crypto download] ERROR building registry: {exc}", err=True)
        raise typer.Exit(1)

    loader = CryptoOHLCVLoader.from_config(cfg, cache=cache, registry=registry, run_mode=run_mode)

    typer.echo(
        f"[crypto download] Downloading {len(sym_list)} symbol(s) "
        f"(mode={mode}, concurrency={concurrency}) …"
    )
    try:
        import asyncio  # noqa: PLC0415

        results = asyncio.run(
            loader.load_many_async(sym_list, concurrency=concurrency)
        )
    except SpectraQuantError as exc:
        typer.echo(f"[crypto download] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    ok = [s for s in sym_list if s in results]
    failed = [s for s in sym_list if s not in results]

    typer.echo(
        f"[crypto download] Done: {len(ok)} succeeded, {len(failed)} failed."
    )
    if failed:
        typer.echo(f"[crypto download] Failed symbols: {', '.join(failed)}", err=True)
        raise typer.Exit(2)
    raise typer.Exit(0)


@crypto_app.command("universe")
def crypto_universe(
    config_dir: str = typer.Option("", "--config-dir", help="Override config/v3/ directory"),
    run_id: str = typer.Option("crypto_universe", "--run-id", help="Run identifier for universe artifact"),
    output_dir: str = typer.Option("reports/universe", "--output-dir", help="Directory to persist universe artifact"),
) -> None:
    """Build and print symbol inclusion decisions for the configured crypto universe."""
    from spectraquant_v3.core import config as config_mod
    from spectraquant_v3.core.errors import ConfigValidationError, EmptyUniverseError
    from spectraquant_v3.crypto.symbols.registry import build_registry_from_config
    from spectraquant_v3.crypto.universe.builder import CryptoUniverseBuilder

    try:
        cfg = config_mod.get_crypto_config(config_dir or None)
        registry = build_registry_from_config(cfg)
        artifact = CryptoUniverseBuilder(cfg, registry, run_id=run_id).build()
    except (FileNotFoundError, ConfigValidationError, ValueError, EmptyUniverseError) as exc:
        typer.echo(f"[crypto universe] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo("symbol included reason")
    for row in artifact.entries:
        typer.echo(f"{row.canonical_symbol} {str(row.included).lower()} {row.reason}")

    written_path = artifact.write(output_dir)
    typer.echo(
        f"[crypto universe] included={len(artifact.included_symbols)} "
        f"excluded={len(artifact.excluded_symbols)} artifact={written_path}"
    )
    raise typer.Exit(0)


@crypto_app.command("signals")
def crypto_signals(
    symbols: str = typer.Option("", "--symbols", help="Comma-separated canonical symbols"),
    config_dir: str = typer.Option("", "--config-dir", help="Override config/v3/ directory"),
) -> None:
    """Run crypto signal agents on cached OHLCV data."""
    typer.echo(
        f"[crypto signals] symbols={symbols or 'from config'}  "
        "⚠  Not yet implemented – scaffold only."
    )
    raise typer.Exit(0)
