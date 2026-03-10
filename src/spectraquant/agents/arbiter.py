"""Arbiter – blends signals from multiple trading agents."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spectraquant.agents.regime import CryptoRegime
from spectraquant.agents.registry import AgentSignal

logger = logging.getLogger(__name__)

_DEFAULT_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    CryptoRegime.BULL.value: {
        "momentum": 0.30,
        "mean_reversion": 0.10,
        "volatility": 0.15,
        "carry_funding": 0.15,
        "news_catalyst": 0.15,
        "onchain_flow": 0.15,
    },
    CryptoRegime.BEAR.value: {
        "momentum": 0.15,
        "mean_reversion": 0.25,
        "volatility": 0.20,
        "carry_funding": 0.15,
        "news_catalyst": 0.10,
        "onchain_flow": 0.15,
    },
    CryptoRegime.RANGE.value: {
        "momentum": 0.10,
        "mean_reversion": 0.30,
        "volatility": 0.15,
        "carry_funding": 0.20,
        "news_catalyst": 0.10,
        "onchain_flow": 0.15,
    },
    CryptoRegime.HIGH_VOL.value: {
        "momentum": 0.10,
        "mean_reversion": 0.15,
        "volatility": 0.30,
        "carry_funding": 0.15,
        "news_catalyst": 0.15,
        "onchain_flow": 0.15,
    },
}


@dataclass
class ArbiterConfig:
    """Configuration knobs for the arbiter blending logic."""

    regime_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: _DEFAULT_REGIME_WEIGHTS.copy(),
    )
    perf_lookback_days: int = 30
    decay: float = 0.95
    weight_floor: float = 0.02
    weight_cap: float = 0.60


class Arbiter:
    """Blends agent signals into a single ranked score per symbol."""

    def __init__(
        self,
        config: ArbiterConfig | None = None,
        report_dir: str | Path = "reports/agents",
    ) -> None:
        self.config = config or ArbiterConfig()
        self.report_dir = Path(report_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def blend(
        self,
        signals: dict[str, list[AgentSignal]],
        regime: CryptoRegime,
        past_performance: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Blend *signals* from multiple agents into a ranked DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``symbol``, ``blended_score``, ``confidence``,
            ``contributing_agents``.
        """
        if not signals:
            return self._empty_frame()

        weights = self._compute_weights(signals, regime, past_performance)
        rows = self._aggregate(signals, weights)

        if not rows:
            return self._empty_frame()

        result = pd.DataFrame(rows)
        result = result.sort_values("blended_score", key=abs, ascending=False)
        result = result.reset_index(drop=True)

        self._persist(result, regime)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        signals: dict[str, list[AgentSignal]],
        regime: CryptoRegime,
        past_performance: dict[str, float] | None,
    ) -> dict[str, float]:
        """Derive final weight per agent from regime + performance."""
        regime_w = self.config.regime_weights.get(regime.value, {})

        # Start with regime-based static weights (default to equal)
        active_agents = [n for n, s in signals.items() if s]
        if not active_agents:
            return {}

        raw: dict[str, float] = {}
        equal = 1.0 / len(active_agents)
        for name in active_agents:
            raw[name] = regime_w.get(name, equal)

        # Adjust by rolling performance when available
        if past_performance:
            for name in active_agents:
                perf = past_performance.get(name, 0.0)
                adjustment = 1.0 + self.config.decay * perf
                raw[name] = raw[name] * max(adjustment, 0.01)

        # Clip and normalise
        for name in list(raw):
            raw[name] = max(raw[name], self.config.weight_floor)
            raw[name] = min(raw[name], self.config.weight_cap)

        total = sum(raw.values())
        if total == 0:
            for name in raw:
                raw[name] = equal
        else:
            for name in raw:
                raw[name] /= total

        return raw

    def _aggregate(
        self,
        signals: dict[str, list[AgentSignal]],
        weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Weighted aggregation of agent signals per symbol."""
        # Collect per symbol
        per_symbol: dict[str, list[tuple[str, AgentSignal, float]]] = {}
        for agent_name, sigs in signals.items():
            w = weights.get(agent_name, 0.0)
            for sig in sigs:
                per_symbol.setdefault(sig.symbol, []).append(
                    (agent_name, sig, w),
                )

        rows: list[dict[str, Any]] = []
        for symbol, entries in per_symbol.items():
            total_weight = sum(w for _, _, w in entries)
            if total_weight == 0:
                # Fallback to equal weight when all weights are zero
                total_weight = float(len(entries))
                blended = sum(s.score for _, s, _ in entries) / total_weight
                conf = sum(s.confidence for _, s, _ in entries) / total_weight
            else:
                blended = sum(s.score * w for _, s, w in entries) / total_weight
                conf = sum(s.confidence * w for _, s, w in entries) / total_weight

            blended = float(np.clip(blended, -1.0, 1.0))
            conf = float(np.clip(conf, 0.0, 1.0))

            rows.append(
                {
                    "symbol": symbol,
                    "blended_score": blended,
                    "confidence": conf,
                    "contributing_agents": sorted(
                        {name for name, _, _ in entries},
                    ),
                },
            )
        return rows

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["symbol", "blended_score", "confidence", "contributing_agents"],
        )

    def _persist(self, result: pd.DataFrame, regime: CryptoRegime) -> None:
        """Write blending decision to ``reports/agents/``."""
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = self.report_dir / f"arbiter_{ts}.json"
            payload = {
                "regime": regime.value,
                "timestamp_utc": ts,
                "decisions": result.to_dict(orient="records"),
            }
            path.write_text(json.dumps(payload, indent=2, default=str))
            logger.info("Arbiter decisions persisted to %s", path)
        except Exception:
            logger.exception("Failed to persist arbiter decisions")


__all__ = ["Arbiter", "ArbiterConfig"]
