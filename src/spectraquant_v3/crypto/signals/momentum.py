"""Crypto momentum signal agent for SpectraQuant-AI-V3.

The agent reads an enriched OHLCV+feature DataFrame (output of
:mod:`spectraquant_v3.crypto.features.engine`) and emits a
:class:`~spectraquant_v3.core.schema.SignalRow` for every symbol.

Signal logic:
- Primary signal: N-day momentum (``ret_Nd`` column).
- Confirmation: RSI filter – avoid buying in overbought territory (>70)
  and avoid shorting in oversold territory (<30).
- Score is normalised to [-1, +1] using a tanh transformation.
- Confidence is proportional to |score|, capped at 1.0.

This module must never import from ``spectraquant_v3.equities``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant_v3.core.enums import AssetClass, SignalStatus
from spectraquant_v3.core.schema import SignalRow

AGENT_ID = "crypto_momentum_v1"
HORIZON = "1d"


class CryptoMomentumAgent:
    """Simple cross-sectional momentum signal agent for crypto.

    The agent evaluates a universe of crypto symbols on a single as-of date
    using their enriched feature DataFrames.

    Args:
        run_id:           Parent run identifier.
        momentum_window:  Lookback period (must match the feature column name).
        rsi_overbought:   RSI threshold above which we dampen long signals.
        rsi_oversold:     RSI threshold below which we dampen short signals.
        min_rows:         Minimum rows needed in the DataFrame.
    """

    def __init__(
        self,
        run_id: str,
        momentum_window: int = 20,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        min_rows: int = 20,
    ) -> None:
        self.run_id = run_id
        self.momentum_window = momentum_window
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.min_rows = min_rows
        self._momentum_col = f"ret_{momentum_window}d"

    @classmethod
    def from_config(cls, cfg: dict[str, Any], run_id: str) -> "CryptoMomentumAgent":
        """Build from merged crypto config."""
        signals_cfg = cfg.get("crypto", {}).get("signals", {})
        return cls(
            run_id=run_id,
            momentum_window=int(signals_cfg.get("momentum_lookback", 20)),
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_for(self, df: pd.DataFrame) -> tuple[float, float, str]:
        """Return (signal_score, confidence, rationale) for the last row of *df*.

        Returns (0.0, 0.0, reason) when data is insufficient.
        """
        if len(df) < self.min_rows:
            return 0.0, 0.0, f"insufficient_rows ({len(df)} < {self.min_rows})"

        cols_lower = {c.lower() for c in df.columns}
        if self._momentum_col not in cols_lower:
            return 0.0, 0.0, f"missing_column_{self._momentum_col}"

        # Align column names
        df_norm = df.copy()
        df_norm.columns = [c.lower() for c in df_norm.columns]

        last = df_norm.iloc[-1]

        mom = float(last.get(self._momentum_col, np.nan))
        if np.isnan(mom):
            return 0.0, 0.0, "momentum_is_nan"

        rsi = float(last.get("rsi", 50.0))
        if np.isnan(rsi):
            rsi = 50.0  # neutral fallback

        # Raw score: tanh-normalise the log return into [-1, +1]
        raw_score = float(np.tanh(mom * 10.0))

        # RSI dampening: reduce signal strength near extremes
        if raw_score > 0 and rsi > self.rsi_overbought:
            dampening = (100.0 - rsi) / (100.0 - self.rsi_overbought)
            raw_score *= max(0.0, dampening)
        elif raw_score < 0 and rsi < self.rsi_oversold:
            dampening = rsi / self.rsi_oversold
            raw_score *= max(0.0, dampening)

        score = float(np.clip(raw_score, -1.0, 1.0))
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
        """Evaluate the momentum signal for a single symbol.

        Args:
            symbol: Canonical crypto ticker.
            df:     Enriched OHLCV+feature DataFrame.
            as_of:  ISO-8601 timestamp; defaults to now(UTC).

        Returns:
            :class:`~spectraquant_v3.core.schema.SignalRow`
        """
        ts = as_of or datetime.now(timezone.utc).isoformat()

        try:
            score, conf, rationale = self._score_for(df)
            if score == 0.0 and conf == 0.0:
                return SignalRow(
                    run_id=self.run_id,
                    timestamp=ts,
                    canonical_symbol=symbol,
                    asset_class=AssetClass.CRYPTO.value,
                    agent_id=AGENT_ID,
                    horizon=HORIZON,
                    signal_score=0.0,
                    confidence=0.0,
                    required_inputs=[self._momentum_col, "rsi"],
                    available_inputs=list({c.lower() for c in df.columns}),
                    rationale=rationale,
                    status=SignalStatus.NO_SIGNAL.value,
                )
            return SignalRow(
                run_id=self.run_id,
                timestamp=ts,
                canonical_symbol=symbol,
                asset_class=AssetClass.CRYPTO.value,
                agent_id=AGENT_ID,
                horizon=HORIZON,
                signal_score=score,
                confidence=conf,
                required_inputs=[self._momentum_col, "rsi"],
                available_inputs=list({c.lower() for c in df.columns}),
                rationale=rationale,
                status=SignalStatus.OK.value,
            )
        except Exception as exc:  # noqa: BLE001
            return SignalRow(
                run_id=self.run_id,
                timestamp=ts,
                canonical_symbol=symbol,
                asset_class=AssetClass.CRYPTO.value,
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
        """Evaluate all symbols in *feature_map*.

        Returns one :class:`SignalRow` per symbol, including symbols that
        produce NO_SIGNAL or ERROR status.
        """
        ts = as_of or datetime.now(timezone.utc).isoformat()
        return [self.evaluate(sym, df, as_of=ts) for sym, df in feature_map.items()]
