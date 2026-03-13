"""Analysis-related CLI commands."""
from __future__ import annotations



def register_analysis_commands(commands: dict[str, callable]) -> None:
    """Register analysis-related commands.
    
    Args:
        commands: Dictionary to register commands into
    """
    from spectraquant.cli.main import (
        cmd_eval,
        cmd_feature_pruning,
        cmd_model_compare,
        cmd_stress_test,
        cmd_regime_stress,
        cmd_explain_portfolio,
        cmd_compare_runs,
        cmd_doctor,
        cmd_health_check,
        cmd_release_check,
    )
    
    commands["eval"] = cmd_eval
    commands["feature-pruning"] = cmd_feature_pruning
    commands["model-compare"] = cmd_model_compare
    commands["stress-test"] = cmd_stress_test
    commands["regime-stress"] = cmd_regime_stress
    commands["explain-portfolio"] = cmd_explain_portfolio
    commands["compare-runs"] = cmd_compare_runs
    commands["doctor"] = cmd_doctor
    commands["health-check"] = cmd_health_check
    commands["release-check"] = cmd_release_check
