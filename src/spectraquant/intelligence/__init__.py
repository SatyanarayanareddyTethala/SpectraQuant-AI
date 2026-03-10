"""SpectraQuant Intelligence Layer — AI trading assistant."""
from spectraquant.intelligence.config import load_config, IntelligenceConfig
from spectraquant.intelligence.failure_memory import (
    record_trade_intent,
    record_trade_outcome,
    label_failure,
    update_failure_stats,
    export_failure_report,
    TradeRecord,
    FailureRecord,
    FAILURE_TYPES,
)
from spectraquant.intelligence.analog_memory import (
    AnalogMarketMemory,
    encode_state,
    AnalogEntry,
    AnalogNeighbor,
)
from spectraquant.intelligence.regime_engine import (
    compute_regime_features,
    classify_regime,
    get_current_regime,
    REGIME_LABELS,
)
from spectraquant.intelligence.meta_learner import (
    propose_policy_update,
    validate_policy_update,
    apply_policy_update,
    rollback_policy,
)
from spectraquant.intelligence.trade_planner import generate_premarket_plan
from spectraquant.intelligence.execution_intelligence import (
    evaluate_trigger,
    monitor_plan,
    CooldownManager,
    TradeState,
    save_execution_snapshot,
)
from spectraquant.intelligence.capital_intelligence import (
    compute_exposures,
    check_trade_allowed,
    generate_risk_report,
)

__all__ = [
    # config
    "load_config",
    "IntelligenceConfig",
    # failure_memory
    "record_trade_intent",
    "record_trade_outcome",
    "label_failure",
    "update_failure_stats",
    "export_failure_report",
    "TradeRecord",
    "FailureRecord",
    "FAILURE_TYPES",
    # analog_memory
    "AnalogMarketMemory",
    "encode_state",
    "AnalogEntry",
    "AnalogNeighbor",
    # regime_engine
    "compute_regime_features",
    "classify_regime",
    "get_current_regime",
    "REGIME_LABELS",
    # meta_learner
    "propose_policy_update",
    "validate_policy_update",
    "apply_policy_update",
    "rollback_policy",
    # trade_planner
    "generate_premarket_plan",
    # execution_intelligence
    "evaluate_trigger",
    "monitor_plan",
    "CooldownManager",
    "TradeState",
    "save_execution_snapshot",
    # capital_intelligence
    "compute_exposures",
    "check_trade_allowed",
    "generate_risk_report",
]
