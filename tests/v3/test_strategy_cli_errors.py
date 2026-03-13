from __future__ import annotations

from typer.testing import CliRunner

from spectraquant_v3.cli.main import app


def test_strategy_run_unknown_strategy_shows_suggestions():
    runner = CliRunner()
    result = runner.invoke(app, ["strategy", "run", "crypto_momentum_v9"])

    assert result.exit_code == 1
    output = result.stdout + result.stderr
    assert "invalid strategy id: 'crypto_momentum_v9'" in output
    assert "Did you mean:" in output
    assert "Available strategies:" in output


def test_scaffold_commands_are_clearly_marked():
    runner = CliRunner()
    equity_signals = runner.invoke(app, ["equity", "signals"])
    crypto_signals = runner.invoke(app, ["crypto", "signals"])

    assert equity_signals.exit_code == 0
    assert "scaffold only" in equity_signals.stdout.lower()
    assert crypto_signals.exit_code == 0
    assert "scaffold only" in crypto_signals.stdout.lower()
