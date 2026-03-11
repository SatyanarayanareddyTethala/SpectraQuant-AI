"""Hybrid equity momentum + news-catalyst + volatility-gate signal agent.

Combines three components into one score:

1. **Momentum confirmation** – N-day momentum with RSI dampening (inherited
   from :class:`~spectraquant_v3.equities.signals.momentum.EquityMomentumAgent`).
2. **News catalyst** – blends a ``news_sentiment_score`` column into the raw
   momentum score when the column is present.  Absent or NaN news falls back
   gracefully to pure momentum.
3. **Volatility gate** – if the ``vol_realised`` column shows annualised
   realised volatility above *vol_gate_threshold*, the blended score is
   dampened proportionally.  This acts as a structural risk-off gate without
   completely zeroing the signal.

Score formula
-------------
::

    blended = momentum_weight * mom_score + news_weight * news_score  (or pure mom if no news)
    gated   = blended * max(0, 1 - (vol_realised - vol_gate) / vol_gate)  (when vol > gate)

Both ``news_sentiment_score`` and ``vol_realised`` are expected to be present
as columns in the enriched feature DataFrame (e.g. produced by an augmented
feature pipeline or passed directly via ``dataset`` to
:func:`~spectraquant_v3.pipeline.equity_pipeline.run_equity_pipeline`).

This module must never import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant_v3.core.enums import AssetClass, SignalStatus
from spectraquant_v3.core.schema import SignalRow
from spectraquant_v3.equities.signals.momentum import EquityMomentumAgent

AGENT_ID = "equity_momentum_news_hybrid_v1"
HORIZON = "1m"  # equity rebalance is monthly by default


def compose_equity_hybrid_score(
    momentum_score: float,
    news_score: float | None,
    vol_realised: float | None = None,
    *,
    momentum_weight: float = 0.7,
    news_weight: float = 0.3,
    vol_gate_threshold: float = 0.25,
) -> float:
    """Compose a blended, vol-gated equity score.

    Args:
        momentum_score:     Momentum signal in ``[-1, +1]``.
        news_score:         News sentiment in ``[-1, +1]``, or ``None`` / NaN
                            to fall back to pure momentum.
        vol_realised:       Annualised realised volatility (e.g. ``0.20`` for
                            20 %).  ``None`` disables the volatility gate.
        momentum_weight:    Weight applied to *momentum_score* in the blend.
        news_weight:        Weight applied to *news_score* in the blend.
        vol_gate_threshold: Annualised vol above which dampening begins.

    Returns:
        Clipped score in ``[-1.0, +1.0]``.
    """
    base = float(momentum_score)
    news_missing = news_score is None or (
        isinstance(news_score, float) and np.isnan(news_score)
    )

    if news_missing:
        blended = base
    else:
        blended = momentum_weight * base + news_weight * float(news_score)

    # Volatility gate: linearly dampen when vol exceeds gate threshold
    if (
        vol_realised is not None
        and np.isfinite(float(vol_realised))
        and float(vol_realised) > vol_gate_threshold
    ):
        excess = float(vol_realised) - vol_gate_threshold
        dampening = max(0.0, 1.0 - excess / vol_gate_threshold)
        blended *= dampening

    return float(np.clip(blended, -1.0, 1.0))


class EquityMomentumNewsHybridAgent(EquityMomentumAgent):
    """Hybrid equity signal: momentum + news catalyst + volatility gate.

    Inherits momentum computation from
    :class:`~spectraquant_v3.equities.signals.momentum.EquityMomentumAgent`
    and augments it with optional news sentiment blending and a volatility gate.

    Args:
        run_id:             Parent run identifier.
        momentum_window:    Lookback period for the momentum feature column.
        momentum_weight:    Fraction of the final score from momentum.
        news_weight:        Fraction of the final score from news sentiment.
        vol_gate_threshold: Annualised realised vol above which signal dampening
                            activates (default ``0.25`` = 25 % annualised).
        rsi_overbought:     RSI threshold above which long signals are dampened.
        rsi_oversold:       RSI threshold below which short signals are dampened.
        min_rows:           Minimum rows needed in the DataFrame.
        min_threshold:      Minimum absolute score to emit a non-zero signal.
    """

    def __init__(
        self,
        run_id: str,
        momentum_window: int = 20,
        momentum_weight: float = 0.7,
        news_weight: float = 0.3,
        vol_gate_threshold: float = 0.25,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        min_rows: int = 20,
        min_threshold: float = 0.05,
    ) -> None:
        super().__init__(
            run_id=run_id,
            momentum_window=momentum_window,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            min_rows=min_rows,
            min_threshold=min_threshold,
        )
        self.momentum_weight = float(momentum_weight)
        self.news_weight = float(news_weight)
        self.vol_gate_threshold = float(vol_gate_threshold)

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any], run_id: str
    ) -> "EquityMomentumNewsHybridAgent":
        """Build from merged equity pipeline config.

        Strategy-specific overrides are read from
        ``cfg["strategies"]["equity_momentum_news_hybrid_v1"]``.
        """
        signals_cfg = cfg.get("equities", {}).get("signals", {})
        portfolio_cfg = cfg.get("portfolio", {})
        root = cfg.get("strategies", {}).get(AGENT_ID, {})
        blend = root.get("signal_blend", {})
        vol_cfg = root.get("vol_gate", {})

        return cls(
            run_id=run_id,
            momentum_window=int(signals_cfg.get("momentum_lookback", 20)),
            momentum_weight=float(blend.get("momentum_weight", 0.7)),
            news_weight=float(blend.get("news_weight", 0.3)),
            vol_gate_threshold=float(
                vol_cfg.get("threshold", signals_cfg.get("hybrid_vol_gate", 0.25))
            ),
            min_threshold=float(portfolio_cfg.get("min_signal_threshold", 0.05)),
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_for(self, df: pd.DataFrame) -> tuple[float, float, str]:
        """Return (signal_score, confidence, rationale) for the last row of *df*."""
        # Get momentum score from parent
        mom_score, mom_conf, mom_rationale = super()._score_for(df)

        # If momentum itself returned no-signal, propagate as-is
        if mom_score == 0.0 and mom_conf == 0.0:
            return mom_score, mom_conf, mom_rationale

        # Extract optional news and volatility columns
        cols = {c.lower(): c for c in df.columns}
        news_score: float | None = None
        vol_realised: float | None = None

        if "news_sentiment_score" in cols:
            v = df[cols["news_sentiment_score"]].iloc[-1]
            news_score = float(v) if pd.notna(v) else None

        if "vol_realised" in cols:
            v = df[cols["vol_realised"]].iloc[-1]
            vol_realised = float(v) if pd.notna(v) else None

        hybrid = compose_equity_hybrid_score(
            momentum_score=mom_score,
            news_score=news_score,
            vol_realised=vol_realised,
            momentum_weight=self.momentum_weight,
            news_weight=self.news_weight,
            vol_gate_threshold=self.vol_gate_threshold,
        )

        conf = float(abs(hybrid))
        rationale = (
            f"{mom_rationale} "
            f"news_sentiment_score={news_score} "
            f"vol_realised={vol_realised} "
            f"hybrid={hybrid:.3f}"
        )
        return hybrid, conf, rationale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        as_of: str | None = None,
    ) -> SignalRow:
        """Evaluate the hybrid signal for a single equity symbol."""
        ts = as_of or datetime.now(timezone.utc).isoformat()

        try:
            score, conf, rationale = self._score_for(df)
            status = (
                SignalStatus.OK.value if score != 0.0 else SignalStatus.NO_SIGNAL.value
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
                required_inputs=[self._momentum_col, "rsi", "news_sentiment_score"],
                available_inputs=list({c.lower() for c in df.columns}),
                rationale=rationale,
                status=status,
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
