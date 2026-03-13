"""Experiment CLI commands for SpectraQuant-AI-V3.

Available commands (registered as sub-commands of the ``experiment`` group):

  list      List all experiments found in the results directory.
  show      Show details (config + metrics) for a single experiment.
  compare   Compare metrics across multiple experiments.
"""

from __future__ import annotations

import typer
from typing_extensions import Annotated

experiment_app = typer.Typer(
    name="experiment",
    help="Experiment tracking and comparison commands. Note: no 'experiment run' command; use list/show/compare or backtest run.",
    no_args_is_help=True,
)


@experiment_app.command("list")
def experiment_list(
    base_dir: Annotated[
        str,
        typer.Option(
            "--base-dir",
            "-d",
            help="Experiments root directory (default: reports/experiments)",
        ),
    ] = "",
) -> None:
    """List all experiments found in the results directory."""
    from spectraquant_v3.experiments.result_store import ResultStore

    store = ResultStore(base_dir or None)
    exp_ids = store.list_experiments()

    if not exp_ids:
        typer.echo(f"No experiments found in '{store.base_dir}'.")
        return

    typer.echo(f"\n{'EXPERIMENT_ID':<30} {'STRATEGY_ID':<30} {'TIMESTAMP':<30} {'SHARPE':<10} {'CAGR'}")
    typer.echo("-" * 120)

    for eid in exp_ids:
        try:
            config_doc = store.read_config(eid)
            metrics = store.read_metrics(eid)
        except Exception:  # noqa: BLE001
            typer.echo(f"{eid:<30} (error reading results)")
            continue

        strategy_id = config_doc.get("strategy_id", "—")
        timestamp = config_doc.get("run_timestamp", "—")[:19]
        sharpe = metrics.get("sharpe")
        cagr = metrics.get("cagr")

        sharpe_str = f"{sharpe:.3f}" if isinstance(sharpe, (int, float)) else "—"
        cagr_str = f"{cagr:.3f}" if isinstance(cagr, (int, float)) else "—"

        typer.echo(f"{eid:<30} {strategy_id:<30} {timestamp:<30} {sharpe_str:<10} {cagr_str}")

    typer.echo(f"\n{len(exp_ids)} experiment(s) found.")


@experiment_app.command("show")
def experiment_show(
    experiment_id: Annotated[
        str,
        typer.Argument(help="Experiment identifier (e.g. exp_001)"),
    ],
    base_dir: Annotated[
        str,
        typer.Option("--base-dir", "-d", help="Experiments root directory"),
    ] = "",
) -> None:
    """Show full details for a single experiment (config + metrics)."""
    import json

    from spectraquant_v3.experiments.result_store import ResultStore

    store = ResultStore(base_dir or None)

    try:
        config_doc = store.read_config(experiment_id)
        metrics = store.read_metrics(experiment_id)
    except FileNotFoundError as exc:
        typer.echo(f"[experiment show] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    output = {
        "experiment_id": experiment_id,
        "config": config_doc,
        "metrics": metrics,
    }
    typer.echo(json.dumps(output, indent=2))


@experiment_app.command("compare")
def experiment_compare(
    exp_ids: Annotated[
        str,
        typer.Argument(
            help="Comma-separated experiment IDs to compare (e.g. exp_001,exp_002)"
        ),
    ],
    base_dir: Annotated[
        str,
        typer.Option("--base-dir", "-d", help="Experiments root directory"),
    ] = "",
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: table | json"),
    ] = "table",
) -> None:
    """Compare metrics across multiple experiments.

    Displays: experiment_id, strategy_id, sharpe, cagr, max_drawdown, volatility,
    win_rate, turnover.

    Example:
      sqv3 experiment compare exp_001,exp_002,exp_003
    """
    import json as _json

    from spectraquant_v3.experiments.experiment_manager import ExperimentManager

    ids = [x.strip() for x in exp_ids.split(",") if x.strip()]
    if not ids:
        typer.echo("[experiment compare] ERROR: no experiment IDs supplied.", err=True)
        raise typer.Exit(1)

    manager = ExperimentManager(base_dir or None)
    rows = manager.compare_experiments(ids)

    selected_rows = [
        {
            "experiment_id": row.get("experiment_id"),
            "strategy_id": row.get("strategy_id"),
            "dataset_version": row.get("dataset_version"),
            "config_hash": row.get("config_hash"),
            "sharpe": row.get("sharpe"),
            "cagr": row.get("cagr"),
            "max_drawdown": row.get("max_drawdown"),
            "volatility": row.get("volatility"),
            "win_rate": row.get("win_rate"),
            "turnover": row.get("turnover"),
            "run_timestamp": row.get("run_timestamp"),
            "total_return": row.get("total_return"),
            "calmar": row.get("calmar"),
            "n_steps": row.get("n_steps"),
            "error": row.get("error"),
        }
        for row in rows
    ]

    if output_format == "json":
        typer.echo(_json.dumps(selected_rows, indent=2, default=str))
        return

    # Table output
    typer.echo(
        f"\n{'EXPERIMENT_ID':<30} {'STRATEGY_ID':<24} "
        f"{'SHARPE':>8} {'CAGR':>8} {'MAX_DD':>8} {'VOL':>8} {'WIN_RATE':>10} {'TURNOVER':>10} {'TOTAL_RET':>10} {'CALMAR':>8} {'N_STEPS':>8}"
    )
    typer.echo("-" * 170)

    def _fmt(v: object, *, pct: bool = False) -> str:
        if isinstance(v, (int, float)):
            if pct:
                return f"{v * 100:.2f}%"
            return f"{v:.3f}"
        return str(v) if v is not None else "—"

    for row in selected_rows:
        if row.get("error"):
            typer.echo(f"{row['experiment_id']:<30} ERROR: {row['error']}")
            continue
        typer.echo(
            f"{row['experiment_id']:<30} {str(row.get('strategy_id') or '—'):<24} "
            f"{_fmt(row.get('sharpe')):>8} {_fmt(row.get('cagr'), pct=True):>8} "
            f"{_fmt(row.get('max_drawdown'), pct=True):>8} {_fmt(row.get('volatility'), pct=True):>8} "
            f"{_fmt(row.get('win_rate'), pct=True):>10} {_fmt(row.get('turnover'), pct=True):>10} "
            f"{_fmt(row.get('total_return'), pct=True):>10} {_fmt(row.get('calmar')):>8} {_fmt(row.get('n_steps')):>8}"
        )
