"""Dynamic analysis model selector for SpectraQuant-AI.

Given a news context, regime state, and volatility conditions for a ticker
the selector deterministically chooses the most appropriate analysis model.

Available models
----------------
EVENT_DRIFT       – used when strong news exists; follows historical event drift
MOMENTUM          – used when no major news and trend persistence is detected
MEAN_REVERSION    – used after overreaction events or in choppy regimes
VOLATILITY        – used for regulatory / macro shocks with high vol
PEER_RELATIVE     – used when competitor news is the primary driver
NO_TRADE          – used when insufficient signal / risk too high

The selection is **fully deterministic** given the same inputs, which is
important for governance and reproducibility.

Usage
-----
>>> from spectraquant.intelligence.model_selector import ModelSelector, AnalysisModel
>>> selector = ModelSelector()
>>> model = selector.select(news_context, regime="TRENDING", vol_state=0.018)
>>> model.value
'EVENT_DRIFT'
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

__all__ = ["AnalysisModel", "ModelSelector"]


class AnalysisModel(str, Enum):
    """Analysis model labels used throughout the pipeline."""

    EVENT_DRIFT = "EVENT_DRIFT"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    VOLATILITY = "VOLATILITY"
    PEER_RELATIVE = "PEER_RELATIVE"
    NO_TRADE = "NO_TRADE"


# ---------------------------------------------------------------------------
# Regime → preferred technical model mapping
# ---------------------------------------------------------------------------

_REGIME_TO_TECH_MODEL: Dict[str, AnalysisModel] = {
    "TRENDING": AnalysisModel.MOMENTUM,
    "RISK_ON": AnalysisModel.MOMENTUM,
    "CHOPPY": AnalysisModel.MEAN_REVERSION,
    "RISK_OFF": AnalysisModel.MEAN_REVERSION,
    "EVENT_DRIVEN": AnalysisModel.EVENT_DRIFT,
    "PANIC": AnalysisModel.VOLATILITY,
}

_DEFAULT_TECH_MODEL = AnalysisModel.MOMENTUM

# ---------------------------------------------------------------------------
# Thresholds (tunable)
# ---------------------------------------------------------------------------

# Magnitude above which news is considered "strong"
_STRONG_NEWS_MAGNITUDE = 0.35

# Magnitude above which we classify as a volatility / shock event
_SHOCK_MAGNITUDE = 0.65

# Annualised vol (daily std * sqrt(252)) threshold for volatility model
_HIGH_VOL_THRESHOLD = 0.30

# Minimum competitor magnitude to trigger peer-relative model
_PEER_MAGNITUDE_THRESHOLD = 0.30


class ModelSelector:
    """Deterministically selects an analysis model per ticker.

    Parameters
    ----------
    strong_news_threshold : float
        Magnitude threshold above which news is authoritative.
    shock_threshold : float
        Magnitude threshold above which a volatility/shock model is used.
    high_vol_threshold : float
        Annualised volatility threshold that forces the VOLATILITY model.
    peer_threshold : float
        Competitor magnitude threshold for the PEER_RELATIVE model.
    """

    def __init__(
        self,
        strong_news_threshold: float = _STRONG_NEWS_MAGNITUDE,
        shock_threshold: float = _SHOCK_MAGNITUDE,
        high_vol_threshold: float = _HIGH_VOL_THRESHOLD,
        peer_threshold: float = _PEER_MAGNITUDE_THRESHOLD,
    ) -> None:
        self._strong_news_threshold = strong_news_threshold
        self._shock_threshold = shock_threshold
        self._high_vol_threshold = high_vol_threshold
        self._peer_threshold = peer_threshold

    def select(
        self,
        news_context: Optional[Dict[str, Any]],
        regime: str = "",
        vol_state: float = 0.0,
    ) -> AnalysisModel:
        """Select an analysis model deterministically.

        Parameters
        ----------
        news_context : dict | None
            Aggregated news context for the ticker.  Expected keys:

            * ``magnitude`` – float [0, 1] from event classifier
            * ``competitor_shock`` – float [0, 1] (peer news magnitude)
            * ``event_type`` – dominant event type string
            * ``uncertainty`` – classifier uncertainty [0, 1]

        regime : str
            Regime label from :mod:`spectraquant.intelligence.regime_engine`.
            One of ``TRENDING``, ``CHOPPY``, ``RISK_ON``, ``RISK_OFF``,
            ``EVENT_DRIVEN``, ``PANIC``.  Empty string means unknown.

        vol_state : float
            Current annualised volatility for the ticker.

        Returns
        -------
        AnalysisModel
        """
        ctx = news_context or {}
        magnitude = float(ctx.get("magnitude", 0.0))
        competitor_shock = float(ctx.get("competitor_shock", 0.0))
        uncertainty = float(ctx.get("uncertainty", 1.0))

        # Step 1: Panic / extreme vol → VOLATILITY regardless of news
        if regime == "PANIC" or vol_state >= self._high_vol_threshold:
            return AnalysisModel.VOLATILITY

        # Step 2: Shock-level news (high magnitude, low uncertainty) → EVENT_DRIFT
        if magnitude >= self._shock_threshold and uncertainty < 0.8:
            return AnalysisModel.EVENT_DRIFT

        # Step 3: Regulatory or macro shock → VOLATILITY
        event_type = str(ctx.get("event_type", "") or "")
        if event_type in ("regulatory", "macro") and magnitude >= self._strong_news_threshold:
            return AnalysisModel.VOLATILITY

        # Step 4: Strong direct news → EVENT_DRIFT
        if magnitude >= self._strong_news_threshold and uncertainty < 0.9:
            return AnalysisModel.EVENT_DRIFT

        # Step 5: Competitor shock (peer news, no direct event) → PEER_RELATIVE
        if competitor_shock >= self._peer_threshold and magnitude < self._strong_news_threshold:
            return AnalysisModel.PEER_RELATIVE

        # Step 6: Post-overreaction / EVENT_DRIVEN regime → MEAN_REVERSION
        if regime == "EVENT_DRIVEN" and magnitude < self._strong_news_threshold:
            return AnalysisModel.MEAN_REVERSION

        # Step 7: Fall back to technical regime model
        return _REGIME_TO_TECH_MODEL.get(regime, _DEFAULT_TECH_MODEL)
