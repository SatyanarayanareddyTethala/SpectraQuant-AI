from __future__ import annotations

from typer.testing import CliRunner

from spectraquant_v3.cli.main import app


def test_strategy_run_unknown_strategy_shows_suggestions():
    runner = CliRunner()
    result = runner.invoke(app, ["strategy", "run", "crypto_momentum_v9"])

    assert result.exit_code == 1
    output = result.stdout + result.stderr
    assert "invalid strategy name: 'crypto_momentum_v9'" in output
    assert "Closest matches:" in output
    assert "Valid strategy IDs:" in output


def test_strategy_show_unknown_strategy_shows_suggestions():
    runner = CliRunner()
    result = runner.invoke(app, ["strategy", "show", "equity_momentum_v9"])

    assert result.exit_code == 1
    output = result.stdout + result.stderr
    assert "invalid strategy name: 'equity_momentum_v9'" in output
    assert "Closest matches:" in output
    assert "Valid strategy IDs:" in output


def test_strategy_run_supports_backward_compatible_strategy_flag():
    runner = CliRunner()
    result = runner.invoke(app, ["strategy", "run", "--strategy", "crypto_momentum_v9"])

    assert result.exit_code == 1
    output = result.stdout + result.stderr
    assert "invalid strategy name: 'crypto_momentum_v9'" in output


def test_strategy_show_conflicting_positional_and_strategy_flag_errors():
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["strategy", "show", "crypto_momentum_v1", "--strategy", "equity_momentum_v1"],
    )

    assert result.exit_code == 1
    output = result.stdout + result.stderr
    assert "received both positional STRATEGY_ID and --strategy" in output


def test_scaffold_commands_are_clearly_marked():
    runner = CliRunner()
    equity_signals = runner.invoke(app, ["equity", "signals"])
    crypto_signals = runner.invoke(app, ["crypto", "signals"])

    assert equity_signals.exit_code == 0
    assert "scaffold only" in equity_signals.stdout.lower()
    assert crypto_signals.exit_code == 0
    assert "scaffold only" in crypto_signals.stdout.lower()
