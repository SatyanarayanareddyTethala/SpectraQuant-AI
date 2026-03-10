"""Strategy Generator — transform hypotheses into testable strategy configs.

Every strategy produced here is *hypothesis-driven*: no random strategies.

Output schema
-------------
Each :class:`StrategyConfig` has:
  - strategy_name
  - hypothesis_id   : the hypothesis that triggered this strategy
  - features_used
  - signal_logic    : description of entry/exit logic
  - risk_rules      : position sizing and stop rules
  - expected_edge   : human-readable rationale
  - parameters      : numeric knobs for the experiment runner
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from spectraquant.intelligence.research_lab.hypothesis_engine import Hypothesis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """A testable strategy derived from a hypothesis."""

    strategy_name: str
    hypothesis_id: str
    features_used: List[str]
    signal_logic: str
    risk_rules: str
    expected_edge: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyConfig":
        return cls(
            strategy_name=d["strategy_name"],
            hypothesis_id=d["hypothesis_id"],
            features_used=d.get("features_used", []),
            signal_logic=d.get("signal_logic", ""),
            risk_rules=d.get("risk_rules", ""),
            expected_edge=d.get("expected_edge", ""),
            parameters=d.get("parameters", {}),
        )


# ---------------------------------------------------------------------------
# Template library (hypothesis-keyword → strategy template)
# ---------------------------------------------------------------------------

_TEMPLATES: List[Dict[str, Any]] = [
    {
        "keywords": ["volatility filter", "volatility-scaling"],
        "name_suffix": "vol_filtered",
        "features": ["realised_vol_20d", "atm_iv_proxy", "vix_proxy", "price_rsi"],
        "signal_logic": (
            "Enter long/short only when realised_vol < threshold; "
            "scale position by inverse-vol weighting"
        ),
        "risk_rules": "Max position = base_size / vol_20d; stop = 2×ATR",
        "expected_edge": "Avoids overtrading in high-volatility noise regimes",
        "parameters": {"vol_threshold": 0.025, "atr_stop_mult": 2.0},
    },
    {
        "keywords": ["momentum weight", "momentum signal", "momentum"],
        "name_suffix": "momentum_adjusted",
        "features": ["price_momentum_5d", "price_momentum_20d", "trend_slope", "regime_label"],
        "signal_logic": (
            "Weight momentum signal by regime confidence; "
            "disable in PANIC/CHOPPY regimes"
        ),
        "risk_rules": "Half position in uncertain regime; full position in TRENDING",
        "expected_edge": "Prevents momentum factor from firing in adverse regime",
        "parameters": {"momentum_lookback": 20, "regime_confidence_min": 0.6},
    },
    {
        "keywords": ["news sentiment", "news_shock", "sentiment"],
        "name_suffix": "sentiment_enhanced",
        "features": ["news_sentiment_score", "news_event_count", "price_rsi", "volume_ratio"],
        "signal_logic": (
            "Combine price signal with news sentiment; "
            "hold off entry 30 min before/after major announcement"
        ),
        "risk_rules": "Reduce size 50% when news_event_count > 2 in 24h; widen stop",
        "expected_edge": "Captures sentiment edge while avoiding announcement whipsaws",
        "parameters": {"sentiment_weight": 0.40, "announcement_buffer_min": 30},
    },
    {
        "keywords": ["holding horizon", "shorter holding"],
        "name_suffix": "short_horizon",
        "features": ["price_momentum_5d", "intraday_vol", "regime_label"],
        "signal_logic": "Reduce target horizon to 1d in high-uncertainty regimes",
        "risk_rules": "Force exit at end of day; max drawdown 1.5% intraday",
        "expected_edge": "Avoids overnight gap risk in uncertain regimes",
        "parameters": {"max_holding_days": 1, "intraday_stop_pct": 0.015},
    },
    {
        "keywords": ["regime gate", "disable momentum", "regime-change", "regime uncertainty"],
        "name_suffix": "regime_gated",
        "features": ["regime_label", "regime_confidence", "breadth_ratio", "vol_regime"],
        "signal_logic": (
            "Add regime gate: only trade if regime confidence > 0.65; "
            "disable conflicting signals on regime change"
        ),
        "risk_rules": "No new entries within 1 bar of regime change; reduce position 30%",
        "expected_edge": "Avoids false signals during regime transitions",
        "parameters": {"regime_confidence_min": 0.65, "regime_change_buffer_bars": 1},
    },
    {
        "keywords": ["confidence calibration", "tighten buy threshold", "overconfidence"],
        "name_suffix": "calibrated_confidence",
        "features": ["model_confidence", "calibrated_prob", "price_rsi", "vol_ratio"],
        "signal_logic": (
            "Apply Platt scaling or isotonic regression to confidence scores; "
            "raise buy threshold to calibrated_prob > 0.62"
        ),
        "risk_rules": "Max confidence-scaled position; floor at 0.5× base size",
        "expected_edge": "Eliminates overconfident trades; improves win rate",
        "parameters": {"buy_threshold": 0.62, "calibration_method": "isotonic"},
    },
    {
        "keywords": ["liquidity filter", "slippage", "low-liquidity"],
        "name_suffix": "liquidity_filtered",
        "features": ["avg_daily_volume", "bid_ask_spread_proxy", "volume_ratio", "price_impact"],
        "signal_logic": (
            "Only enter when avg_daily_volume > liquidity_floor; "
            "scale position by sqrt(volume_ratio)"
        ),
        "risk_rules": "Max position = liquidity_fraction × ADV; market-impact cap",
        "expected_edge": "Reduces slippage cost; improves net returns in small-cap universe",
        "parameters": {"liquidity_floor_shares": 100000, "adv_fraction": 0.01},
    },
    {
        "keywords": ["regularisation", "overfit", "feature count", "lookback"],
        "name_suffix": "regularised",
        "features": ["top_10_alpha_features", "regime_label", "price_momentum_20d"],
        "signal_logic": (
            "Retrain model with L2 regularisation and max 10 features; "
            "use 60-day rolling window instead of full history"
        ),
        "risk_rules": "Standard risk rules; no changes to position sizing",
        "expected_edge": "Reduces overfit; improves generalisation to unseen regimes",
        "parameters": {"max_features": 10, "lookback_days": 60, "l2_alpha": 0.1},
    },
]


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class StrategyGenerator:
    """Convert hypotheses into testable strategy configurations."""

    def generate(self, hypotheses: List[Hypothesis]) -> List[StrategyConfig]:
        """Return a list of strategy configs for the given hypotheses.

        Parameters
        ----------
        hypotheses : list[Hypothesis]
            Hypotheses produced by :class:`HypothesisEngine`.

        Returns
        -------
        list[StrategyConfig]
        """
        configs: List[StrategyConfig] = []
        for hyp in hypotheses:
            cfg = self._hypothesis_to_strategy(hyp)
            if cfg is not None:
                configs.append(cfg)
                logger.info(
                    "Strategy '%s' generated from hypothesis %s",
                    cfg.strategy_name,
                    hyp.hypothesis_id,
                )
        return configs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hypothesis_to_strategy(self, hyp: Hypothesis) -> Optional[StrategyConfig]:
        """Match hypothesis text to the best template and build config."""
        combined_text = (
            (hyp.trigger_reason + " " + hyp.suggested_feature_change).lower()
        )

        best_template: Optional[Dict[str, Any]] = None
        best_score = 0
        for tmpl in _TEMPLATES:
            score = sum(1 for kw in tmpl["keywords"] if kw.lower() in combined_text)
            if score > best_score:
                best_score = score
                best_template = tmpl

        if best_template is None or best_score == 0:
            # Fallback: generic strategy derived from hypothesis text
            return StrategyConfig(
                strategy_name=f"hyp_{hyp.hypothesis_id}_generic",
                hypothesis_id=hyp.hypothesis_id,
                features_used=["price_momentum_20d", "realised_vol_20d", "regime_label"],
                signal_logic=f"Generic strategy to address: {hyp.trigger_reason}",
                risk_rules="Standard risk rules; 1% per-trade stop",
                expected_edge=hyp.suggested_feature_change,
                parameters={"generic": True},
            )

        return StrategyConfig(
            strategy_name=f"hyp_{hyp.hypothesis_id}_{best_template['name_suffix']}",
            hypothesis_id=hyp.hypothesis_id,
            features_used=list(best_template["features"]),
            signal_logic=best_template["signal_logic"],
            risk_rules=best_template["risk_rules"],
            expected_edge=best_template["expected_edge"],
            parameters=dict(best_template["parameters"]),
        )
