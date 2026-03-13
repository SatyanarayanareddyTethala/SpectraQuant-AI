"""CLI commands for SpectraQuant-AI-V3 Strategy Portfolio.

Provides:
  sqv3 strategy-portfolio run     Run a multi-strategy portfolio.
  sqv3 strategy-portfolio list    List configured strategy portfolios.
"""

from __future__ import annotations


import typer
from typing_extensions import Annotated

strategy_portfolio_app = typer.Typer(
    name="strategy-portfolio",
    help="Strategy portfolio commands — run and inspect multi-strategy portfolios.",
    no_args_is_help=True,
)


@strategy_portfolio_app.command("list")
def strategy_portfolio_list(
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", help="Override config/v3/ directory"),
    ] = "",
) -> None:
    """List all configured strategy portfolios.

    Reads portfolio definitions from config/v3/strategies.yaml.
    """
    import yaml
    from spectraquant_v3.core.config import _find_config_dir

    cfg_dir = config_dir or str(_find_config_dir())
    strategies_yaml = f"{cfg_dir}/strategies.yaml"

    try:
        with open(strategies_yaml) as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        typer.echo(f"strategies.yaml not found at {strategies_yaml}", err=True)
        raise typer.Exit(1)

    portfolios = (data or {}).get("strategy_portfolios", {})
    if not portfolios:
        typer.echo("No strategy portfolios configured.")
        return

    typer.echo(f"{'PORTFOLIO ID':<30} {'STRATEGIES':<40} {'WEIGHTING'}")
    typer.echo("-" * 80)
    for pid, pdef in portfolios.items():
        sids = ", ".join(pdef.get("strategy_ids", []))
        scheme = pdef.get("weighting_scheme", "equal")
        typer.echo(f"{pid:<30} {sids:<40} {scheme}")
    typer.echo(f"\n{len(portfolios)} portfolio(s) found.")


@strategy_portfolio_app.command("run")
def strategy_portfolio_run(
    portfolio_id: Annotated[
        str,
        typer.Argument(help="Portfolio ID from strategies.yaml"),
    ],
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", help="Override config/v3/ directory"),
    ] = "",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--no-dry-run", help="Skip writes and network calls"),
    ] = False,
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", help="Output directory for results"),
    ] = "reports/strategy_portfolio",
) -> None:
    """Run a multi-strategy portfolio simulation.

    Reads the portfolio definition from config/v3/strategies.yaml and
    runs each constituent strategy, then aggregates portfolio-level metrics.
    """
    import yaml
    from spectraquant_v3.core.config import _find_config_dir, get_crypto_config, get_equity_config
    from spectraquant_v3.strategy_portfolio import StrategyPortfolio

    cfg_dir = config_dir or str(_find_config_dir())
    strategies_yaml = f"{cfg_dir}/strategies.yaml"

    try:
        with open(strategies_yaml) as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        typer.echo(f"strategies.yaml not found at {strategies_yaml}", err=True)
        raise typer.Exit(1)

    portfolios = (data or {}).get("strategy_portfolios", {})
    if portfolio_id not in portfolios:
        typer.echo(
            f"Portfolio '{portfolio_id}' not found.  "
            f"Available: {', '.join(portfolios.keys())}",
            err=True,
        )
        raise typer.Exit(1)

    pdef = portfolios[portfolio_id]
    strategy_ids = pdef.get("strategy_ids", [])
    weighting_scheme = pdef.get("weighting_scheme", "equal")
    max_weight = pdef.get("max_strategy_weight", 1.0)
    rebalance_freq = pdef.get("rebalance_frequency", "M")

    # Build per-strategy configs
    all_strategies = (data or {}).get("strategies", {})
    cfg_by_strategy: dict = {}
    for sid in strategy_ids:
        asset_class = (all_strategies.get(sid) or {}).get("asset_class", "crypto")
        if asset_class == "crypto":
            cfg_by_strategy[sid] = get_crypto_config(config_dir=cfg_dir)
        else:
            cfg_by_strategy[sid] = get_equity_config(config_dir=cfg_dir)

    portfolio = StrategyPortfolio(
        portfolio_id=portfolio_id,
        strategy_ids=strategy_ids,
        weighting_scheme=weighting_scheme,
        max_strategy_weight=max_weight,
        rebalance_frequency=rebalance_freq,
        output_dir=output_dir,
    )

    typer.echo(f"Running strategy portfolio: {portfolio_id}")
    typer.echo(f"  Strategies : {', '.join(strategy_ids)}")
    typer.echo(f"  Weighting  : {weighting_scheme}")
    typer.echo(f"  Dry-run    : {dry_run}")
    typer.echo("")

    result = portfolio.run(
        cfg_by_strategy=cfg_by_strategy,
        dry_run=dry_run,
    )

    typer.echo(f"Portfolio run complete: {result.portfolio_id}")
    typer.echo(f"  Weights    : {result.weights}")
    typer.echo(f"  Metrics    : {result.metrics}")
    if result.artifact_paths:
        typer.echo(f"  Output     : {result.artifact_paths[0]}")
