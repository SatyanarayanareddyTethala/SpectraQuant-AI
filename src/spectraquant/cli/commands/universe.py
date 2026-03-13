"""Universe-related CLI commands."""
from __future__ import annotations



def register_universe_commands(commands: dict[str, callable]) -> None:
    """Register universe-related commands.
    
    Args:
        commands: Dictionary to register commands into
    """
    from spectraquant.cli.main import (
        cmd_universe_update_nse,
        cmd_universe_stats_nse,
        cmd_universe_stats,
    )
    
    commands["universe-update-nse"] = cmd_universe_update_nse
    commands["universe-nse-stats"] = cmd_universe_stats_nse
    commands["universe-stats"] = cmd_universe_stats
