"""Strategy CLI commands for SpectraQuant-AI-V3.

Available commands (registered as sub-commands of the ``strategy`` group):

  list    List all registered strategies.
  show    Show details for a single strategy.
  run     Run a registered strategy through the appropriate pipeline.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Optional

import typer
from typing_extensions import Annotated



def _strategy_not_found_error(strategy_id: str) -> str:
    from spectraquant_v3.strategies.registry import StrategyRegistry

    valid_ids = StrategyRegistry.list()
    suggestions = get_close_matches(strategy_id, valid_ids, n=3, cutoff=0.4)
    parts = [f"invalid strategy name: '{strategy_id}'"]
    if suggestions:
        parts.append(f"Closest matches: {', '.join(suggestions)}")
    parts.append(f"Valid strategy IDs: {', '.join(valid_ids)}")
    return "\n".join(parts)


def _resolve_strategy_id(positional_strategy_id: str, option_strategy_id: Optional[str]) -> str:
    """Resolve strategy id from positional arg and optional backward-compatible flag."""
    resolved = positional_strategy_id
    if option_strategy_id:
        if positional_strategy_id and positional_strategy_id != option_strategy_id:
            typer.echo(
                "[strategy] ERROR: received both positional STRATEGY_ID and --strategy with different "
                "values. Use exactly one strategy value.",
                err=True,
            )
            raise typer.Exit(1)
        resolved = option_strategy_id

    if not resolved:
        typer.echo(
            "[strategy] ERROR: missing strategy name. Usage: sqv3 strategy run STRATEGY_ID "
            "(or sqv3 strategy run --strategy STRATEGY_ID).",
            err=True,
        )
        raise typer.Exit(1)
    return resolved

strategy_app = typer.Typer(
    name="strategy",
    help="Strategy registry commands.",
    no_args_is_help=True,
)


@strategy_app.command("list")
def strategy_list(
    asset_class: Annotated[
        str,
        typer.Option(
            "--asset-class",
            "-a",
            help="Filter by asset class: crypto | equity | all",
        ),
    ] = "all",
    enabled_only: Annotated[
        bool,
        typer.Option("--enabled-only", help="Show only enabled strategies"),
    ] = False,
) -> None:
    """List all registered strategies.

    Displays strategy ID, asset class, agents, policy, allocator, and tags.
    """
    from spectraquant_v3.strategies.registry import StrategyRegistry

    strategies = StrategyRegistry.list_all()

    if asset_class != "all":
        strategies = [s for s in strategies if s.asset_class == asset_class.lower()]

    if enabled_only:
        strategies = [s for s in strategies if s.enabled]

    if not strategies:
        typer.echo("No strategies found.")
        return

    typer.echo(f"\n{'ID':<30} {'ASSET':<8} {'AGENTS':<30} {'POLICY':<25} {'ALLOC':<20} {'ENABLED'}")
    typer.echo("-" * 120)
    for s in strategies:
        agents_str = ",".join(s.agents)
        typer.echo(
            f"{s.strategy_id:<30} {s.asset_class:<8} {agents_str:<30} "
            f"{s.policy:<25} {s.allocator:<20} {s.enabled}"
        )
    typer.echo(f"\n{len(strategies)} strategy(ies) registered.")


@strategy_app.command("show")
def strategy_show(
    strategy_id: Annotated[
        str,
        typer.Argument(help="Strategy identifier (e.g. crypto_momentum_v1)"),
    ] = "",
    strategy: Annotated[
        Optional[str],
        typer.Option(
            "--strategy",
            help="Backward-compatible alias for STRATEGY_ID.",
        ),
    ] = None,
) -> None:
    """Show full details for a single strategy.

    Displays every field of the StrategyDefinition including risk limits.
    """
    import json

    from spectraquant_v3.strategies.registry import StrategyRegistry

    resolved_strategy_id = _resolve_strategy_id(strategy_id, strategy)

    try:
        defn = StrategyRegistry.get(resolved_strategy_id)
    except KeyError:
        typer.echo(
            f"[strategy show] ERROR:\n{_strategy_not_found_error(resolved_strategy_id)}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(json.dumps(defn.to_dict(), indent=2))


@strategy_app.command("run")
def strategy_run(
    strategy_id: Annotated[
        str,
        typer.Argument(help="Strategy identifier to run (e.g. crypto_momentum_v1)"),
    ] = "",
    strategy: Annotated[
        Optional[str],
        typer.Option(
            "--strategy",
            help="Backward-compatible alias for STRATEGY_ID.",
        ),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="Run mode: normal | test | refresh"),
    ] = "normal",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Skip writes and network calls"),
    ] = False,
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", help="Override config/v3/ directory"),
    ] = "",
) -> None:
    """Run a registered strategy through the appropriate pipeline.

    The pipeline (crypto or equity) is chosen automatically based on the
    strategy's asset_class field.  Mixing asset classes raises an error.
    """
    from spectraquant_v3.core.config import get_crypto_config, get_equity_config
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import MixedAssetClassRunError, SpectraQuantError
    from spectraquant_v3.pipeline import run_strategy
    from spectraquant_v3.strategies.registry import StrategyRegistry

    resolved_strategy_id = _resolve_strategy_id(strategy_id, strategy)

    # Validate run mode
    try:
        run_mode = RunMode(mode.lower())
    except ValueError:
        typer.echo(
            f"[strategy run] ERROR: invalid --mode={mode!r}. Use normal|test|refresh.",
            err=True,
        )
        raise typer.Exit(1)

    # Look up strategy to determine asset class
    try:
        defn = StrategyRegistry.get(resolved_strategy_id)
    except KeyError:
        typer.echo(
            f"[strategy run] ERROR:\n{_strategy_not_found_error(resolved_strategy_id)}",
            err=True,
        )
        raise typer.Exit(1)

    # Load the appropriate config
    try:
        if defn.asset_class == "crypto":
            cfg = get_crypto_config(config_dir or None)
        else:
            cfg = get_equity_config(config_dir or None)
    except FileNotFoundError as exc:
        typer.echo(f"[strategy run] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"[strategy run] strategy={resolved_strategy_id!r} "
        f"asset_class={defn.asset_class!r} "
        f"mode={mode} dry_run={dry_run} – starting …"
    )

    try:
        result = run_strategy(
            strategy_id=resolved_strategy_id,
            cfg=cfg,
            run_mode=run_mode,
            dry_run=dry_run,
        )
        typer.echo(
            f"[strategy run] completed  status={result['status']} "
            f"universe={len(result.get('universe', []))} symbols"
        )
    except MixedAssetClassRunError as exc:
        typer.echo(f"[strategy run] ASSET CLASS ERROR: {exc}", err=True)
        raise typer.Exit(1)
    except SpectraQuantError as exc:
        typer.echo(f"[strategy run] PIPELINE ERROR: {exc}", err=True)
        raise typer.Exit(1)
