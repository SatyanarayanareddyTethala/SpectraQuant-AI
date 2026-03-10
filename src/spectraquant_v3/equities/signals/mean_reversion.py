"""Equity mean-reversion signal agent for SpectraQuant-AI-V3.

The agent scores a symbol by how far its price has deviated from its
rolling mean (z-score), with the assumption that extreme deviations tend
to revert.  A large negative z-score (price well below mean) yields a
positive (buy) signal; a large positive z-score yields a negative signal.

Signal logic:
- Rolling z-score of close price over *window* periods.
- score = -z_score / (z_threshold * 2), clipped to [-1.0, +1.0].
- Confidence proportional to |z_score| / 3, capped at 1.0.

This module must never import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant_v3.core.enums import AssetClass, NoSignalReason, SignalStatus
from spectraquant_v3.core.schema import SignalRow

AGENT_ID = "equity_mean_reversion_v1"
HORIZON = "5d"

_MIN_ROWS_DEFAULT = 20


class EquityMeanReversionAgent:
    """Rolling z-score mean-reversion signal agent for equities.

    Args:
        run_id:       Parent run identifier.
        window:       Lookback window for rolling mean and std.
        z_threshold:  Z-score level beyond which the signal is considered
                      significant (default 1.0).
        min_rows:     Minimum number of rows required to produce a signal.
    """

    def __init__(
        self,
        run_id: str,
        window: int = 20,
        z_threshold: float = 1.0,
        min_rows: int = _MIN_ROWS_DEFAULT,
    ) -> None:
        self.run_id = run_id
        self.window = window
        self.z_threshold = z_threshold
        self.min_rows = min_rows

    @classmethod
    def from_config(cls, cfg: dict[str, Any], run_id: str) -> "EquityMeanReversionAgent":
        """Build from merged equity config."""
        signals_cfg = cfg.get("equities", {}).get("signals", {})
        return cls(
            run_id=run_id,
            window=int(signals_cfg.get("mean_reversion_window", 20)),
            z_threshold=float(signals_cfg.get("mean_reversion_z_threshold", 1.0)),
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

        close = pd.to_numeric(df_norm["close"], errors="coerce")
        rolling_mean = close.rolling(self.window).mean()
        rolling_std = close.rolling(self.window).std()

        z_raw = (close - rolling_mean) / rolling_std.replace(0.0, np.nan)
        z_score = float(z_raw.iloc[-1])

        if np.isnan(z_score):
            return (
                0.0,
                0.0,
                "z_score is NaN (insufficient variance)",
                NoSignalReason.MISSING_INPUTS.value,
            )

        # Negative z → price below mean → positive signal (buy)
        score = float(np.clip(
            -z_score / (self.z_threshold * 2.0),
            -1.0,
            1.0,
        ))
        confidence = float(min(1.0, abs(z_score) / 3.0 + 0.1))

        rationale = (
            f"z_score={z_score:.3f} "
            f"window={self.window} "
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
        """Evaluate the mean-reversion signal for a single equity symbol."""
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
