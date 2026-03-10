"""Equity breakout signal agent for SpectraQuant-AI-V3.

The agent fires when the current close is near or above the rolling N-period
high of the *high* column, indicating a potential price breakout.

Signal logic:
- Rolling window high computed over *window* periods.
- Proximity score = (close / rolling_high) - 0.9 (positive near breakout).
- Score clipped to [-0.5, +1.0] then tanh-normalised to [-1, +1].
- Confidence proportional to |score|, capped at 1.0.

This module must never import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant_v3.core.enums import AssetClass, NoSignalReason, SignalStatus
from spectraquant_v3.core.schema import SignalRow

AGENT_ID = "equity_breakout_v1"
HORIZON = "1m"

_MIN_ROWS_DEFAULT = 20


class EquityBreakoutAgent:
    """Rolling-high breakout signal agent for equities.

    Args:
        run_id:     Parent run identifier.
        window:     Lookback period for rolling high (periods).
        min_rows:   Minimum number of rows required to produce a signal.
    """

    def __init__(
        self,
        run_id: str,
        window: int = 52,
        min_rows: int = _MIN_ROWS_DEFAULT,
    ) -> None:
        self.run_id = run_id
        self.window = window
        self.min_rows = min_rows

    @classmethod
    def from_config(cls, cfg: dict[str, Any], run_id: str) -> "EquityBreakoutAgent":
        """Build from merged equity config."""
        signals_cfg = cfg.get("equities", {}).get("signals", {})
        return cls(
            run_id=run_id,
            window=int(signals_cfg.get("breakout_window", 52)),
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_for(
        self, df: pd.DataFrame
    ) -> tuple[float, float, str, str]:
        """Return (signal_score, confidence, rationale, no_signal_reason).

        Returns (0.0, 0.0, reason, no_signal_reason) when data is insufficient.
        """
        if len(df) < self.min_rows:
            return (
                0.0,
                0.0,
                f"insufficient_rows ({len(df)} < {self.min_rows})",
                NoSignalReason.INSUFFICIENT_ROWS.value,
            )

        df_norm = df.copy()
        df_norm.columns = [c.lower() for c in df_norm.columns]

        if "high" not in df_norm.columns or "close" not in df_norm.columns:
            return (
                0.0,
                0.0,
                "missing high or close column",
                NoSignalReason.MISSING_INPUTS.value,
            )

        high = pd.to_numeric(df_norm["high"], errors="coerce").dropna()
        close = pd.to_numeric(df_norm["close"], errors="coerce")
        current = float(close.iloc[-1])

        if math.isnan(current) or current <= 0.0:
            return (
                0.0,
                0.0,
                "invalid close price",
                NoSignalReason.MISSING_INPUTS.value,
            )

        period = min(self.window, len(high))
        rolling_high = float(high.rolling(period).max().iloc[-1])

        if rolling_high <= 0.0 or math.isnan(rolling_high):
            return (
                0.0,
                0.0,
                "invalid rolling high",
                NoSignalReason.MISSING_INPUTS.value,
            )

        # proximity: +ve when price is close to / exceeding the rolling high
        proximity = (current / rolling_high) - 0.9  # 0 at 90%, +0.1 at 100%
        # Clip to meaningful range then tanh-normalise
        raw = float(np.clip(proximity * 5.0, -0.5, 1.0))
        score = float(np.tanh(raw))
        score = float(np.clip(score, -1.0, 1.0))
        confidence = float(abs(score))

        rationale = (
            f"close={current:.4f} "
            f"{period}d_high={rolling_high:.4f} "
            f"proximity={proximity:.4f} "
            f"score={score:.3f}"
        )
        return score, confidence, rationale, ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        as_of: str | None = None,
    ) -> SignalRow:
        """Evaluate the breakout signal for a single equity symbol."""
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
                    required_inputs=["high", "close"],
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
                required_inputs=["high", "close"],
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
