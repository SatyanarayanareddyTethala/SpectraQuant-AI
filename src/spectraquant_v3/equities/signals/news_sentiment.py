"""Equity news-catalyst signal agent for SpectraQuant-AI-V3.

Ported from the V2 :class:`spectraquant.equities.signals.news_sentiment_agent.NewsSentimentAgent`.

The agent reads a ``news_sentiment_score`` column from the pre-enriched feature
DataFrame and emits a :class:`~spectraquant_v3.core.schema.SignalRow`.

Degradation contract (NON-FATAL):
- Missing column           → NO_SIGNAL / NO_NEWS_DATA
- NaN value                → NO_SIGNAL / NO_NEWS_DATA
- Score below min_confidence → NO_SIGNAL / BELOW_THRESHOLD
- Any unexpected exception → ERROR (score=0, confidence=0)

This module must never import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant_v3.core.enums import AssetClass, NoSignalReason, SignalStatus
from spectraquant_v3.core.schema import SignalRow

AGENT_ID = "equity_news_sentiment_v1"
HORIZON = "1d"  # news catalyst acts on a daily horizon


class EquityNewsSentimentAgent:
    """News-catalyst signal agent for equities.

    Reads the ``news_sentiment_score`` column from the enriched feature
    DataFrame.  The column is expected to hold a pre-normalised value in
    ``[-1.0, +1.0]`` (e.g. produced by a news-enriched feature pipeline or
    passed directly in the ``dataset`` argument to :func:`run_equity_pipeline`).

    When the column is absent or its last row is NaN the agent returns
    ``NO_SIGNAL`` with ``no_signal_reason=NO_NEWS_DATA``.  This is **non-fatal**
    and is intended to allow the strategy to degrade gracefully when news data
    is not available for a given symbol or run.

    Args:
        run_id:         Parent run identifier.
        min_confidence: Minimum absolute score to emit a non-zero signal.
                        Scores below this threshold yield NO_SIGNAL.
    """

    def __init__(
        self,
        run_id: str,
        min_confidence: float = 0.1,
    ) -> None:
        self.run_id = run_id
        self.min_confidence = float(min_confidence)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], run_id: str) -> "EquityNewsSentimentAgent":
        """Build from merged equity config."""
        signals_cfg = cfg.get("equities", {}).get("signals", {})
        return cls(
            run_id=run_id,
            min_confidence=float(signals_cfg.get("news_min_confidence", 0.1)),
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_for(
        self, df: pd.DataFrame
    ) -> tuple[float, float, str, str]:
        """Return (signal_score, confidence, rationale, no_signal_reason)."""
        df_norm = df.copy()
        df_norm.columns = [c.lower() for c in df_norm.columns]

        if "news_sentiment_score" not in df_norm.columns:
            return (
                0.0,
                0.0,
                "missing news_sentiment_score column",
                NoSignalReason.NO_NEWS_DATA.value,
            )

        val = df_norm["news_sentiment_score"].iloc[-1]
        if pd.isna(val):
            return (
                0.0,
                0.0,
                "news_sentiment_score is NaN",
                NoSignalReason.NO_NEWS_DATA.value,
            )

        score = float(np.clip(float(val), -1.0, 1.0))
        confidence = float(abs(score))

        if confidence < self.min_confidence:
            return (
                0.0,
                0.0,
                f"score={score:.3f} below min_confidence={self.min_confidence}",
                NoSignalReason.BELOW_THRESHOLD.value,
            )

        return score, confidence, f"news_sentiment_score={score:.3f}", ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        as_of: str | None = None,
    ) -> SignalRow:
        """Evaluate the news sentiment signal for a single equity symbol."""
        ts = as_of or datetime.now(timezone.utc).isoformat()

        try:
            score, conf, rationale, no_signal_reason = self._score_for(df)

            if no_signal_reason:
                return SignalRow(
                    run_id=self.run_id,
                    timestamp=ts,
                    canonical_symbol=symbol,
                    asset_class=AssetClass.EQUITY.value,
                    agent_id=AGENT_ID,
                    horizon=HORIZON,
                    signal_score=0.0,
                    confidence=0.0,
                    required_inputs=["news_sentiment_score"],
                    available_inputs=list({c.lower() for c in df.columns}),
                    rationale=rationale,
                    no_signal_reason=no_signal_reason,
                    status=SignalStatus.NO_SIGNAL.value,
                )

            return SignalRow(
                run_id=self.run_id,
                timestamp=ts,
                canonical_symbol=symbol,
                asset_class=AssetClass.EQUITY.value,
                agent_id=AGENT_ID,
                horizon=HORIZON,
                signal_score=score,
                confidence=conf,
                required_inputs=["news_sentiment_score"],
                available_inputs=list({c.lower() for c in df.columns}),
                rationale=rationale,
                status=SignalStatus.OK.value,
            )
        except Exception as exc:  # noqa: BLE001
            return SignalRow(
                run_id=self.run_id,
                timestamp=ts,
                canonical_symbol=symbol,
                asset_class=AssetClass.EQUITY.value,
                agent_id=AGENT_ID,
                horizon=HORIZON,
                status=SignalStatus.ERROR.value,
                error_reason=str(exc),
            )

    def evaluate_many(
        self,
        feature_map: dict[str, pd.DataFrame],
        as_of: str | None = None,
    ) -> list[SignalRow]:
        """Evaluate all symbols in *feature_map*."""
        ts = as_of or datetime.now(timezone.utc).isoformat()
        return [self.evaluate(sym, df, as_of=ts) for sym, df in feature_map.items()]
