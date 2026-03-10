"""Equity momentum signal agent for SpectraQuant-AI-V3.

The agent reads an enriched OHLCV+feature DataFrame (output of
:mod:`spectraquant_v3.equities.features.engine`) and emits a
:class:`~spectraquant_v3.core.schema.SignalRow` for every symbol.

Signal logic (identical structure to crypto agent, different defaults):
- Primary signal: N-day momentum (``ret_Nd`` column).
- Confirmation: RSI filter.
- Score normalised to [-1, +1] via tanh.
- Confidence proportional to |score|.

This module must never import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant_v3.core.enums import AssetClass, SignalStatus
from spectraquant_v3.core.schema import SignalRow

AGENT_ID = "equity_momentum_v1"
HORIZON = "1m"  # equity rebalance is monthly by default


class EquityMomentumAgent:
    """Simple cross-sectional momentum signal agent for equities.

    Args:
        run_id:           Parent run identifier.
        momentum_window:  Lookback period (must match the feature column name).
        rsi_overbought:   RSI threshold above which long signals are dampened.
        rsi_oversold:     RSI threshold below which short signals are dampened.
        min_rows:         Minimum rows needed in the DataFrame.
        min_threshold:    Minimum absolute score to emit a non-zero signal.
    """

    def __init__(
        self,
        run_id: str,
        momentum_window: int = 20,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        min_rows: int = 20,
        min_threshold: float = 0.05,
    ) -> None:
        self.run_id = run_id
        self.momentum_window = momentum_window
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.min_rows = min_rows
        self.min_threshold = min_threshold
        self._momentum_col = f"ret_{momentum_window}d"

    @classmethod
    def from_config(cls, cfg: dict[str, Any], run_id: str) -> "EquityMomentumAgent":
        """Build from merged equity config."""
        signals_cfg = cfg.get("equities", {}).get("signals", {})
        portfolio_cfg = cfg.get("portfolio", {})
        return cls(
            run_id=run_id,
            momentum_window=int(signals_cfg.get("momentum_lookback", 20)),
            min_threshold=float(portfolio_cfg.get("min_signal_threshold", 0.05)),
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_for(self, df: pd.DataFrame) -> tuple[float, float, str]:
        """Return (signal_score, confidence, rationale) for the last row of *df*."""
        if len(df) < self.min_rows:
            return 0.0, 0.0, f"insufficient_rows ({len(df)} < {self.min_rows})"

        cols_lower = {c.lower() for c in df.columns}
        if self._momentum_col not in cols_lower:
            return 0.0, 0.0, f"missing_column_{self._momentum_col}"

        df_norm = df.copy()
        df_norm.columns = [c.lower() for c in df_norm.columns]
        last = df_norm.iloc[-1]

        mom = float(last.get(self._momentum_col, np.nan))
        if np.isnan(mom):
            return 0.0, 0.0, "momentum_is_nan"

        rsi = float(last.get("rsi", 50.0))
        if np.isnan(rsi):
            rsi = 50.0

        raw_score = float(np.tanh(mom * 10.0))

        if raw_score > 0 and rsi > self.rsi_overbought:
            dampening = (100.0 - rsi) / (100.0 - self.rsi_overbought)
            raw_score *= max(0.0, dampening)
        elif raw_score < 0 and rsi < self.rsi_oversold:
            dampening = rsi / self.rsi_oversold
            raw_score *= max(0.0, dampening)

        score = float(np.clip(raw_score, -1.0, 1.0))

        # Apply minimum threshold – below this, treat as NO_SIGNAL
        if abs(score) < self.min_threshold:
            return 0.0, 0.0, f"score={score:.3f} below min_threshold={self.min_threshold}"

        confidence = float(abs(score))
        rationale = (
            f"momentum_{self.momentum_window}d={mom:.4f} "
            f"rsi={rsi:.1f} "
            f"score={score:.3f}"
        )
        return score, confidence, rationale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        as_of: str | None = None,
    ) -> SignalRow:
        """Evaluate the momentum signal for a single equity symbol."""
        ts = as_of or datetime.now(timezone.utc).isoformat()

        try:
            score, conf, rationale = self._score_for(df)
            status = SignalStatus.OK.value if score != 0.0 else SignalStatus.NO_SIGNAL.value
            return SignalRow(
                run_id=self.run_id,
                timestamp=ts,
                canonical_symbol=symbol,
                asset_class=AssetClass.EQUITY.value,
                agent_id=AGENT_ID,
                horizon=HORIZON,
                signal_score=score,
                confidence=conf,
                required_inputs=[self._momentum_col, "rsi"],
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
