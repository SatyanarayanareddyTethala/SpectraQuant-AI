"""SpectraQuant-AI-V3 Experiments package.

Exports the experiment manager, run tracker, result store, and hybrid
strategy parameterisation helpers.
"""

from spectraquant_v3.experiments.experiment_manager import ExperimentManager
from spectraquant_v3.experiments.hybrid_params import (
    EQUITY_HYBRID_ID,
    CRYPTO_HYBRID_ID,
    HYBRID_STRATEGY_IDS,
    HybridStrategyParams,
)
from spectraquant_v3.experiments.result_store import ResultStore
from spectraquant_v3.experiments.run_tracker import RunTracker

__all__ = [
    "ExperimentManager",
    "ResultStore",
    "RunTracker",
    "HybridStrategyParams",
    "EQUITY_HYBRID_ID",
    "CRYPTO_HYBRID_ID",
    "HYBRID_STRATEGY_IDS",
]
