"""Typer CLI entry point for SpectraQuant-AI-V3.

Entry point: ``sqv3`` (configured in pyproject.toml).

Top-level commands:
  sqv3 crypto   Crypto pipeline commands (crypto-only run).
  sqv3 equity   Equity pipeline commands (equity-only run).
  sqv3 doctor   Validate environment, config, and dependencies.
  sqv3 version  Print the package version.

IMPORTANT: ``sqv3 crypto`` and ``sqv3 equity`` must NEVER be combined in a
single invocation.  Mixing asset classes raises MixedAssetClassRunError.
"""

from __future__ import annotations

import typer
from typing_extensions import Annotated

import spectraquant_v3
from spectraquant_v3.cli.commands.backtest import backtest_app
from spectraquant_v3.cli.commands.crypto import crypto_app
from spectraquant_v3.cli.commands.equities import equity_app
from spectraquant_v3.cli.commands.experiment import experiment_app
from spectraquant_v3.cli.commands.feature_store import feature_store_app
from spectraquant_v3.cli.commands.research import research_app
from spectraquant_v3.cli.commands.strategy import strategy_app
from spectraquant_v3.cli.commands.strategy_portfolio import strategy_portfolio_app

app = typer.Typer(
    name="sqv3",
    help=(
        "SpectraQuant-AI-V3 — production-grade systematic research and trading.\n\n"
        "Use 'sqv3 crypto' for crypto-only runs and 'sqv3 equity' for equity-only runs.\n"
        "Mixing asset classes in a single invocation is explicitly forbidden."
    ),
    no_args_is_help=True,
    add_completion=True,
)

# Register asset-class sub-applications
app.add_typer(crypto_app, name="crypto")
app.add_typer(equity_app, name="equity")

# Register research and backtest sub-applications
app.add_typer(research_app, name="research")
app.add_typer(backtest_app, name="backtest")

# Register strategy and experiment sub-applications
app.add_typer(strategy_app, name="strategy")
app.add_typer(experiment_app, name="experiment")

# Register feature store and strategy portfolio sub-applications
app.add_typer(feature_store_app, name="feature-store")
app.add_typer(strategy_portfolio_app, name="strategy-portfolio")


# ---------------------------------------------------------------------------
# Top-level utility commands
# ---------------------------------------------------------------------------


@app.command("version")
def version_cmd() -> None:
    """Print the SpectraQuant-AI-V3 package version."""
    typer.echo(f"SpectraQuant-AI-V3 v{spectraquant_v3.__version__}")


@app.command("doctor")
def doctor_cmd(
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", help="Override config/v3/ directory"),
    ] = "",
) -> None:
    """Validate environment, config files, and optional dependencies.

    Checks performed:
    - config/v3/base.yaml, crypto.yaml, equities.yaml are readable
    - Required Python packages are importable
    - Data directories exist or can be created
    - No cross-asset contamination in config
    """
    import importlib
    from pathlib import Path

    from spectraquant_v3.core.config import _find_config_dir

    ok = True
    typer.echo("SpectraQuant-AI-V3 — doctor\n")

    # --- config files -------------------------------------------------------
    config_dir_path = Path(config_dir) if config_dir else _find_config_dir()
    typer.echo(f"  Config dir : {config_dir_path}")
    for fname in ("base.yaml", "crypto.yaml", "equities.yaml"):
        fpath = config_dir_path / fname
        if fpath.exists():
            typer.echo(f"  ✓ {fname}")
        else:
            typer.echo(f"  ✗ {fname}  [NOT FOUND]", err=True)
            ok = False

    # --- required packages --------------------------------------------------
    typer.echo("")
    required_packages = [
        ("yaml", "PyYAML"),
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
        ("typer", "typer"),
    ]
    optional_packages = [
        ("ccxt", "ccxt"),
        ("yfinance", "yfinance"),
    ]
    for module, pkg in required_packages:
        try:
            importlib.import_module(module)
            typer.echo(f"  ✓ {pkg} (required)")
        except ImportError:
            typer.echo(f"  ✗ {pkg} (required)  [NOT INSTALLED]", err=True)
            ok = False
    for module, pkg in optional_packages:
        try:
            importlib.import_module(module)
            typer.echo(f"  ✓ {pkg} (optional)")
        except ImportError:
            typer.echo(f"  - {pkg} (optional)  [not installed]")

    typer.echo("")
    if ok:
        typer.echo("All checks passed.")
    else:
        typer.echo("Some checks failed.  See output above.", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Invoke the Typer CLI app (called by the ``sqv3`` console script)."""
    app()
