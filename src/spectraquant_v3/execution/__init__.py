"""Execution module package for SpectraQuant-AI-V3.

Simulates order execution with transaction costs, slippage, and spread
proxies.  Supports paper and live execution modes.

Public API::

    from spectraquant_v3.execution import ExecutionSimulator, ExecutionResult
"""

from spectraquant_v3.execution.simulator import ExecutionSimulator
from spectraquant_v3.execution.result import ExecutionResult

__all__ = ["ExecutionSimulator", "ExecutionResult"]
