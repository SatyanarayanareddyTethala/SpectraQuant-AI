"""Equity volatility-regime signal agent for SpectraQuant-AI-V3.

The agent computes realised annualised volatility from daily close returns
and produces a risk-off signal when volatility exceeds a configurable
threshold.  The signal is neutral (0.0) in normal-vol regimes and
increasingly negative (risk-off) as vol rises above the threshold.

Signal logic:
- Annualised realised vol = rolling std of log returns × sqrt(252).
- Score = max(-1.0, -(ann_vol - neutral_vol) / (2 × neutral_vol)), clipped to [-1, 0].
- Score is 0.0 when ann_vol ≤ neutral_vol.
- Confidence is fixed at 0.7 (vol signals carry structural uncertainty).

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

AGENT_ID = "equity_volatility_v1"
HORIZON = "1m"

_MIN_ROWS_DEFAULT = 21  # need at least 20 returns for a 20-day window


class EquityVolatilityAgent:
    """Realised-volatility regime signal agent for equities.

    Produces a negative (risk-off) signal when annualised volatility
    exceeds the *neutral_vol* threshold, and a neutral signal otherwise.

    Args:
        run_id:       Parent run identifier.
        window:       Rolling window for realised vol (trading days).
        neutral_vol:  Annualised vol below which signal is 0.0 (default 0.15).
        min_rows:     Minimum rows required to produce a signal.
    """

    def __init__(
        self,
        run_id: str,
        window: int = 20,
        neutral_vol: float = 0.15,
        min_rows: int = _MIN_ROWS_DEFAULT,
    ) -> None:
        self.run_id = run_id
        self.window = window
        self.neutral_vol = neutral_vol
        self.min_rows = min_rows

    @classmethod
    def from_config(cls, cfg: dict[str, Any], run_id: str) -> "EquityVolatilityAgent":
        """Build from merged equity config."""
        signals_cfg = cfg.get("equities", {}).get("signals", {})
        return cls(
            run_id=run_id,
            window=int(signals_cfg.get("volatility_window", 20)),
            neutral_vol=float(signals_cfg.get("volatility_neutral_vol", 0.15)),
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_for(
        self, df: pd.DataFrame
    ) -> tuple[float, float, str, str]:
        """Return (signal_score, confidence, rationale, no_signal_reason)."""
        if len(df) < self.min_rows:
            return (
                0.0,
                0.0,
                f"insufficient_rows ({len(df)} < {self.min_rows})",
                NoSignalReason.INSUFFICIENT_ROWS.value,
            )

        df_norm = df.copy()
        df_norm.columns = [c.lower() for c in df_norm.columns]

        if "close" not in df_norm.columns:
            return (
                0.0,
                0.0,
                "missing close column",
                NoSignalReason.MISSING_INPUTS.value,
            )

        close = pd.to_numeric(df_norm["close"], errors="coerce").dropna()
        log_returns = np.log(close / close.shift(1)).dropna()

        daily_vol = float(log_returns.rolling(self.window).std().iloc[-1])

        if math.isnan(daily_vol) or daily_vol < 0.0:
            return (
                0.0,
                0.0,
                "could not compute realised vol",
                NoSignalReason.MISSING_INPUTS.value,
            )

        ann_vol = daily_vol * math.sqrt(252.0)

        if ann_vol <= self.neutral_vol:
            score = 0.0
        else:
            # Risk-off: the more vol exceeds the neutral level, the more negative
            score = float(max(-1.0, -(ann_vol - self.neutral_vol) / (2.0 * self.neutral_vol)))
            score = float(np.clip(score, -1.0, 0.0))

        confidence = 0.7  # structural uncertainty in vol signals

        rationale = (
            f"ann_vol={ann_vol:.4f} "
            f"neutral_vol={self.neutral_vol:.4f} "
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
        """Evaluate the volatility signal for a single equity symbol."""
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
                    required_inputs=["close"],
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
                required_inputs=["close"],
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
