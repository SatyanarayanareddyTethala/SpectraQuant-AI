"""Data-related CLI commands."""
from __future__ import annotations

from typing import Any


def register_data_commands(commands: dict[str, callable]) -> None:
    """Register data-related commands.
    
    Args:
        commands: Dictionary to register commands into
    """
    from spectraquant.cli.main import (
        cmd_download,
        cmd_news_scan,
        cmd_features,
        cmd_build_dataset,
        cmd_refresh,
    )
    
    commands["download"] = cmd_download
    commands["news-scan"] = cmd_news_scan
    commands["features"] = cmd_features
    commands["build-dataset"] = cmd_build_dataset
    commands["refresh"] = cmd_refresh
