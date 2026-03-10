"""Signal-agent execution helpers.

Provides a small, explicit split between per-symbol (single-name) signal agents
and universe-aware cross-sectional agents.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from spectraquant_v3.core.schema import SignalRow


def is_cross_sectional_agent(agent: Any) -> bool:
    """Return True when *agent* should be executed with full-universe input."""
    mode = str(getattr(agent, "execution_mode", "")).lower()
    if mode in {"cross_sectional", "universe"}:
        return True
    return hasattr(agent, "evaluate_many") or hasattr(agent, "evaluate_cross_section")


def run_signal_agent(
    agent: Any,
    feature_map: dict[str, pd.DataFrame],
    as_of: str,
) -> list[SignalRow]:
    """Execute a signal agent against *feature_map* in the correct mode."""
    if is_cross_sectional_agent(agent):
        if hasattr(agent, "evaluate_many"):
            return list(agent.evaluate_many(feature_map, as_of=as_of))
        if hasattr(agent, "evaluate_cross_section"):
            return list(agent.evaluate_cross_section(feature_map, as_of=as_of))
        raise AttributeError(
            f"Cross-sectional agent {agent.__class__.__name__} must implement "
            "'evaluate_many' or 'evaluate_cross_section'."
        )

    rows: list[SignalRow] = []
    for symbol, frame in feature_map.items():
        rows.append(agent.evaluate(symbol, frame, as_of=as_of))
    return rows
