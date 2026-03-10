"""Hybrid crypto momentum+news signal composition.

This module combines technical momentum and 24h news sentiment into a
single score:

``score = momentum_weight * momentum_score + news_weight * news_sentiment_24h``

It also supports an optional bounded uplift when news shock exceeds a
configured z-score threshold.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant_v3.core.enums import AssetClass, SignalStatus
from spectraquant_v3.core.schema import SignalRow
from spectraquant_v3.crypto.signals.momentum import CryptoMomentumAgent

AGENT_ID = "crypto_momentum_news_hybrid_v1"
HORIZON = "1d"


def compose_hybrid_score(
    momentum_score: float,
    news_sentiment_24h: float | None,
    *,
    news_shock_zscore: float | None = None,
    momentum_weight: float = 0.7,
    news_weight: float = 0.3,
    shock_threshold: float | None = None,
    shock_max_uplift: float = 0.15,
) -> float:
    """Compose a blended score from momentum and news features.

    Missing news sentiment falls back to pure momentum score.
    """
    base = float(momentum_score)
    news_missing = news_sentiment_24h is None or np.isnan(float(news_sentiment_24h))

    if news_missing:
        score = base
    else:
        score = (momentum_weight * base) + (news_weight * float(news_sentiment_24h))

    if (
        shock_threshold is not None
        and shock_max_uplift > 0
        and news_shock_zscore is not None
        and np.isfinite(float(news_shock_zscore))
        and float(news_shock_zscore) > float(shock_threshold)
    ):
        exceed = float(news_shock_zscore) - float(shock_threshold)
        uplift = min(float(shock_max_uplift), exceed * float(shock_max_uplift))
        direction = np.sign(score if score != 0.0 else base)
        if direction == 0:
            direction = 1.0
        score += float(direction) * float(uplift)

    return float(np.clip(score, -1.0, 1.0))


class CryptoMomentumNewsHybridAgent(CryptoMomentumAgent):
    """Crypto signal agent that blends momentum and 24h news sentiment."""

    def __init__(
        self,
        run_id: str,
        momentum_window: int = 20,
        momentum_weight: float = 0.7,
        news_weight: float = 0.3,
        shock_threshold: float | None = None,
        shock_max_uplift: float = 0.15,
    ) -> None:
        super().__init__(run_id=run_id, momentum_window=momentum_window)
        self.momentum_weight = float(momentum_weight)
        self.news_weight = float(news_weight)
        self.shock_threshold = shock_threshold
        self.shock_max_uplift = float(shock_max_uplift)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], run_id: str) -> "CryptoMomentumNewsHybridAgent":
        root = cfg.get("strategies", {}).get("crypto_momentum_news_hybrid_v1", {})
        blend = root.get("signal_blend", {})
        shock = root.get("shock_uplift", {})
        signals_cfg = cfg.get("crypto", {}).get("signals", {})

        threshold_raw = shock.get("threshold")
        threshold = float(threshold_raw) if threshold_raw is not None else None

        return cls(
            run_id=run_id,
            momentum_window=int(signals_cfg.get("momentum_lookback", 20)),
            momentum_weight=float(blend.get("momentum_weight", 0.7)),
            news_weight=float(blend.get("news_weight", 0.3)),
            shock_threshold=threshold,
            shock_max_uplift=float(shock.get("max_uplift", 0.15)),
        )

    def _score_for(self, df: pd.DataFrame) -> tuple[float, float, str]:
        score, _conf, rationale = super()._score_for(df)
        if score == 0.0 and "insufficient_rows" in rationale:
            return score, 0.0, rationale

        cols = {c.lower(): c for c in df.columns}
        news_sent = None
        shock = None
        if "news_sentiment_24h" in cols:
            v = df[cols["news_sentiment_24h"]].iloc[-1]
            news_sent = float(v) if pd.notna(v) else None
        if "news_shock_zscore" in cols:
            v = df[cols["news_shock_zscore"]].iloc[-1]
            shock = float(v) if pd.notna(v) else None

        hybrid = compose_hybrid_score(
            momentum_score=score,
            news_sentiment_24h=news_sent,
            news_shock_zscore=shock,
            momentum_weight=self.momentum_weight,
            news_weight=self.news_weight,
            shock_threshold=self.shock_threshold,
            shock_max_uplift=self.shock_max_uplift,
        )
        conf = float(abs(hybrid))
        return (
            hybrid,
            conf,
            f"{rationale} news_sentiment_24h={news_sent} news_shock_zscore={shock} hybrid={hybrid:.3f}",
        )


    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        as_of: str | None = None,
    ) -> SignalRow:
        ts = as_of or datetime.now(timezone.utc).isoformat()

        try:
            score, conf, rationale = self._score_for(df)
            status = SignalStatus.OK.value
            if score == 0.0 and conf == 0.0:
                status = SignalStatus.NO_SIGNAL.value

            return SignalRow(
                run_id=self.run_id,
                timestamp=ts,
                canonical_symbol=symbol,
                asset_class=AssetClass.CRYPTO.value,
                agent_id=AGENT_ID,
                horizon=HORIZON,
                signal_score=float(score),
                confidence=float(conf),
                required_inputs=[self._momentum_col, "rsi", "news_sentiment_24h"],
                available_inputs=list({c.lower() for c in df.columns}),
                rationale=rationale,
                status=status,
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
