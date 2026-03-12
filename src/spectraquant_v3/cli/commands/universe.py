"""Universe CLI commands for SpectraQuant-AI-V3.

Available commands (registered as sub-commands of the ``universe`` group):
  validate   Load the hybrid universe file, show counts and check for issues.
"""

from __future__ import annotations

import typer

universe_app = typer.Typer(
    name="universe",
    help="Hybrid universe utilities (equities + crypto + FX).",
    no_args_is_help=True,
)


@universe_app.command("validate")
def universe_validate(
    universe_file: str = typer.Option(
        "",
        "--file",
        "-f",
        help="Path to the hybrid universe CSV.  Defaults to universe.file in base.yaml.",
    ),
    config_dir: str = typer.Option(
        "",
        "--config-dir",
        help="Override config/v3/ directory.",
    ),
) -> None:
    """Validate the hybrid universe CSV and print a summary.

    Checks performed:
    \\b
    - All required columns are present
    - No empty symbol fields
    - All asset_class values are valid (equity / crypto / forex)
    - Duplicate symbols are reported
    - Total asset count does not exceed 150

    Example output::

        ## Universe summary

        Equities: 10
        Crypto: 10
        FX: 4
        Total: 24

        Duplicates dropped: 0
        Status: OK
    """
    from spectraquant_v3.core.config import load_config
    from spectraquant_v3.core.errors import UniverseValidationError
    from spectraquant_v3.core.universe_loader import load_universe

    # ------------------------------------------------------------------
    # Resolve universe file path
    # ------------------------------------------------------------------
    path = universe_file.strip()
    if not path:
        try:
            cfg = load_config(config_dir or None, force_reload=True)
        except FileNotFoundError as exc:
            typer.echo(f"[universe validate] ERROR loading config: {exc}", err=True)
            raise typer.Exit(1)
        path = cfg.get("universe", {}).get("file", "")

    if not path:
        typer.echo(
            "[universe validate] ERROR: No universe file configured. "
            "Pass --file or set universe.file in config/v3/base.yaml.",
            err=True,
        )
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Load and validate
    # ------------------------------------------------------------------
    try:
        universe = load_universe(path)
    except FileNotFoundError as exc:
        typer.echo(f"[universe validate] ERROR: {exc}", err=True)
        raise typer.Exit(1)
    except UniverseValidationError as exc:
        typer.echo(f"[universe validate] VALIDATION ERROR:\n{exc}", err=True)
        raise typer.Exit(1)

    equities = universe["equities"]
    crypto = universe["crypto"]
    forex = universe["forex"]
    duplicates: list[str] = universe.get("_duplicates_dropped", [])  # type: ignore[assignment]
    total = len(equities) + len(crypto) + len(forex)

    # ------------------------------------------------------------------
    # Output summary
    # ------------------------------------------------------------------
    typer.echo("\n## Universe summary\n")
    typer.echo(f"Equities: {len(equities)}")
    typer.echo(f"Crypto:   {len(crypto)}")
    typer.echo(f"FX:       {len(forex)}")
    typer.echo(f"Total:    {total}")

    if duplicates:
        typer.echo(f"\nDuplicates dropped ({len(duplicates)}): {', '.join(duplicates)}")
    else:
        typer.echo("\nDuplicates dropped: 0")

    typer.echo("\nStatus: OK")
    raise typer.Exit(0)
