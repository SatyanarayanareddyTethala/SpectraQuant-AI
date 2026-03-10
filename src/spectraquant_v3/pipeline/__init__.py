"""Pipeline orchestrators for SpectraQuant-AI-V3.

Each pipeline is a separate module.  They must never be invoked together
in the same command.  Shared scaffolding lives in ``spectraquant_v3.core``.

The :func:`run_strategy` convenience function dispatches to the correct
asset-class pipeline based on the strategy definition.
"""

from spectraquant_v3.pipeline._strategy_runner import run_strategy

__all__ = ["run_strategy"]
