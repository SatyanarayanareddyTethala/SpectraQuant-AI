"""Portfolio-related CLI commands."""
from __future__ import annotations

from typing import Any


def register_portfolio_commands(commands: dict[str, callable]) -> None:
    """Register portfolio-related commands.
    
    Args:
        commands: Dictionary to register commands into
    """
    from spectraquant.cli.main import (
        cmd_signals,
        cmd_score,
        cmd_portfolio,
        cmd_execute,
    )
    
    commands["signals"] = cmd_signals
    commands["score"] = cmd_score
    commands["portfolio"] = cmd_portfolio
    commands["execute"] = cmd_execute
