"""Model-related CLI commands."""
from __future__ import annotations

from typing import Any


def register_model_commands(commands: dict[str, callable]) -> None:
    """Register model-related commands.
    
    Args:
        commands: Dictionary to register commands into
    """
    from spectraquant.cli.main import (
        cmd_train,
        cmd_predict,
        cmd_retrain,
        cmd_promote_model,
        cmd_list_models,
    )
    
    commands["train"] = cmd_train
    commands["predict"] = cmd_predict
    commands["retrain"] = cmd_retrain
    commands["promote-model"] = cmd_promote_model
    commands["list-models"] = cmd_list_models
