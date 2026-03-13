"""Equity data-quality filter agent for SpectraQuant-AI-V3.

The quality agent acts as a gate: it emits a neutral ``OK`` signal (score=0.0)
when a symbol has sufficient price history and acceptable data quality, and
returns ``NO_SIGNAL`` when the data quality is too poor for other agents to
reliably use.

Quality checks:
1. Minimum row count (*min_rows* trading days of close prices).
2. Maximum fraction of zero daily returns (*max_zero_return_fraction*) — a
   proxy for illiquid or duplicated/stale prices.

This module must never import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from spectraquant_v3.core.enums import AssetClass, NoSignalReason, SignalStatus
from spectraquant_v3.core.schema import SignalRow

AGENT_ID = "equity_quality_v1"
HORIZON = "1m"


class EquityQualityAgent:
    """Price-data quality gate for equities.

    Emits ``OK`` (score=0.0) for symbols that pass all quality checks and
    ``NO_SIGNAL`` for symbols that fail.  Other signal agents should run
    only when this gate passes.

    Args:
        run_id:                   Parent run identifier.
        min_rows:                 Minimum number of close-price rows required.
        max_zero_return_fraction: Maximum tolerated fraction of zero daily
                                  returns (default 0.10 = 10 %).
    """

    def __init__(
        self,
        run_id: str,
        min_rows: int = 60,
        max_zero_return_fraction: float = 0.10,
    ) -> None:
        self.run_id = run_id
        self.min_rows = min_rows
        self.max_zero_return_fraction = max_zero_return_fraction

    @classmethod
    def from_config(cls, cfg: dict[str, Any], run_id: str) -> "EquityQualityAgent":
        """Build from merged equity config."""
        quality_cfg = cfg.get("equities", {}).get("quality_gate", {})
        signals_cfg = cfg.get("equities", {}).get("signals", {})
        return cls(
            run_id=run_id,
            min_rows=int(
                signals_cfg.get(
                    "quality_min_rows",
                    quality_cfg.get("min_history_days", 60),
                )
            ),
            max_zero_return_fraction=float(
                signals_cfg.get("quality_max_zero_fraction", 0.10)
            ),
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _check(
        self, df: pd.DataFrame
    ) -> tuple[bool, str, str]:
        """Return (passed, rationale, no_signal_reason)."""
        df_norm = df.copy()
        df_norm.columns = [c.lower() for c in df_norm.columns]

        if "close" not in df_norm.columns:
            return (
                False,
                "missing close column",
                NoSignalReason.MISSING_INPUTS.value,
            )

        close = pd.to_numeric(df_norm["close"], errors="coerce").dropna()

        if len(close) < self.min_rows:
            return (
                False,
                f"only {len(close)} rows (need {self.min_rows})",
                NoSignalReason.INSUFFICIENT_ROWS.value,
            )

        returns = close.pct_change().dropna()
        zero_frac = float((returns == 0.0).sum() / max(len(returns), 1))

        if zero_frac > self.max_zero_return_fraction:
            return (
                False,
                f"zero_return_fraction={zero_frac:.3f} exceeds threshold={self.max_zero_return_fraction:.3f}",
                NoSignalReason.MISSING_INPUTS.value,
            )

        return (
            True,
            f"quality_gate_passed rows={len(close)} zero_frac={zero_frac:.3f}",
            "",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        as_of: str | None = None,
    ) -> SignalRow:
        """Evaluate data quality for a single equity symbol."""
        ts = as_of or datetime.now(timezone.utc).isoformat()

        try:
            passed, rationale, no_signal_reason = self._check(df)

            if not passed:
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
                signal_score=0.0,   # gate: neutral score
                confidence=0.9,
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
