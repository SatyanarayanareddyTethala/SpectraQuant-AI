"""Signal-agent execution helpers.

Provides a small, explicit split between per-symbol (single-name) signal agents
and universe-aware cross-sectional agents.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from spectraquant_v3.core.enums import SignalStatus
from spectraquant_v3.core.schema import SignalRow

_runner_logger = logging.getLogger(__name__)


def is_cross_sectional_agent(agent: Any) -> bool:
    """Return True when *agent* should be executed with full-universe input."""
    mode = str(getattr(agent, "execution_mode", "")).lower()
    if mode in {"cross_sectional", "universe"}:
        return True
    return hasattr(agent, "evaluate_many") or hasattr(agent, "evaluate_cross_section")


def _error_row_for(
    agent: Any,
    symbol: str,
    as_of: str,
    exc: BaseException,
) -> SignalRow:
    """Build an ERROR :class:`SignalRow` when an agent call raises unexpectedly.

    Reads identifying attributes from *agent* with safe fallbacks so that
    the error row is always a fully-valid :class:`SignalRow` even if the
    agent is poorly implemented.
    """
    return SignalRow(
        run_id=str(getattr(agent, "run_id", "unknown")),
        timestamp=as_of,
        canonical_symbol=symbol,
        asset_class=str(getattr(agent, "asset_class", "")),
        agent_id=str(getattr(agent, "agent_id", agent.__class__.__name__)),
        horizon=str(getattr(agent, "horizon", "")),
        signal_score=0.0,
        confidence=0.0,
        status=SignalStatus.ERROR.value,
        error_reason=f"{type(exc).__name__}: {exc}",
    )


def run_signal_agent(
    agent: Any,
    feature_map: dict[str, pd.DataFrame],
    as_of: str,
) -> list[SignalRow]:
    """Execute a signal agent against *feature_map* in the correct mode.

    Contract guarantees:
    - Always returns one :class:`SignalRow` per symbol in *feature_map*
      (for single-symbol agents) or all symbols the agent handles (for
      cross-sectional agents).
    - If the agent raises an unhandled exception for a symbol, an ERROR
      :class:`SignalRow` is emitted for that symbol so the pipeline can
      continue.  The exception is logged at WARNING level.
    """
    if is_cross_sectional_agent(agent):
        if hasattr(agent, "evaluate_many"):
            try:
                return list(agent.evaluate_many(feature_map, as_of=as_of))
            except Exception as exc:  # noqa: BLE001
                _runner_logger.warning(
                    "run_signal_agent: cross-sectional agent %s.evaluate_many raised %s: %s; "
                    "emitting ERROR rows for all %d symbols.",
                    agent.__class__.__name__,
                    type(exc).__name__,
                    exc,
                    len(feature_map),
                )
                return [
                    _error_row_for(agent, sym, as_of, exc)
                    for sym in feature_map
                ]
        if hasattr(agent, "evaluate_cross_section"):
            try:
                return list(agent.evaluate_cross_section(feature_map, as_of=as_of))
            except Exception as exc:  # noqa: BLE001
                _runner_logger.warning(
                    "run_signal_agent: cross-sectional agent %s.evaluate_cross_section "
                    "raised %s: %s; emitting ERROR rows for all %d symbols.",
                    agent.__class__.__name__,
                    type(exc).__name__,
                    exc,
                    len(feature_map),
                )
                return [
                    _error_row_for(agent, sym, as_of, exc)
                    for sym in feature_map
                ]
        raise AttributeError(
            f"Cross-sectional agent {agent.__class__.__name__} must implement "
            "'evaluate_many' or 'evaluate_cross_section'."
        )

    rows: list[SignalRow] = []
    for symbol, frame in feature_map.items():
        try:
            rows.append(agent.evaluate(symbol, frame, as_of=as_of))
        except Exception as exc:  # noqa: BLE001
            _runner_logger.warning(
                "run_signal_agent: agent %s.evaluate raised %s for symbol %r: %s; "
                "emitting ERROR row.",
                agent.__class__.__name__,
                type(exc).__name__,
                symbol,
                exc,
            )
            rows.append(_error_row_for(agent, symbol, as_of, exc))
    return rows
