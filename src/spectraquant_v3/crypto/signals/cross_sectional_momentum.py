"""Cross-sectional crypto momentum signal agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spectraquant_v3.core.enums import AssetClass, SignalStatus
from spectraquant_v3.core.schema import SignalRow

AGENT_ID = "crypto_cross_sectional_momentum_v1"
HORIZON = "1d"

_DEFAULT_HORIZONS = {
    "ret_5": "ret_5",
    "ret_20": "ret_20",
    "ret_60": "ret_60",
    "ret_120": "ret_120",
}

_DEFAULT_WEIGHTS = {
    "ret_20": 0.35,
    "ret_60": 0.35,
    "ret_120": 0.20,
    "ret_5": 0.10,
}


class CryptoCrossSectionalMomentumAgent:
    """Cross-sectional momentum ranker for crypto symbols."""

    execution_mode = "cross_sectional"

    def __init__(
        self,
        run_id: str,
        top_n: int = 10,
        horizons: dict[str, str] | None = None,
        weights: dict[str, float] | None = None,
        min_rows: int = 1,
    ) -> None:
        self.run_id = run_id
        self.top_n = max(int(top_n), 1)
        self.horizons = {**_DEFAULT_HORIZONS, **(horizons or {})}
        self.weights = {**_DEFAULT_WEIGHTS, **(weights or {})}
        self.min_rows = min_rows

    @classmethod
    def from_config(cls, cfg: dict[str, Any], run_id: str) -> "CryptoCrossSectionalMomentumAgent":
        strategy_id = str(cfg.get("_strategy_id", AGENT_ID))
        signals_cfg = cfg.get("crypto", {}).get("signals", {})
        strategy_cfg = cfg.get("strategies", {}).get(strategy_id, {})

        top_n = int(
            strategy_cfg.get(
                "top_n",
                signals_cfg.get("top_n", cfg.get("crypto", {}).get("universe_top_n", 10)),
            )
        )
        horizons = strategy_cfg.get("cross_sectional_horizons", signals_cfg.get("cross_sectional_horizons"))
        weights = strategy_cfg.get("cross_sectional_weights", signals_cfg.get("cross_sectional_weights"))

        return cls(run_id=run_id, top_n=top_n, horizons=horizons, weights=weights)

    def _column_aliases(self, canonical_name: str) -> list[str]:
        base = canonical_name.lower()
        return [base, base.replace("_", ""), f"{base}d", f"{base}_d"]

    def _extract_return(self, df: pd.DataFrame, canonical_name: str) -> float | None:
        if len(df) < self.min_rows:
            return None
        candidate_col = self.horizons.get(canonical_name, canonical_name)
        last = df.iloc[-1]
        lower_map = {c.lower(): c for c in df.columns}
        for c in [candidate_col] + self._column_aliases(canonical_name):
            raw_col = lower_map.get(c.lower())
            if raw_col is None:
                continue
            value = float(last.get(raw_col, np.nan))
            if not np.isnan(value):
                return value

        days = canonical_name.split("_")[-1]
        if days.isdigit() and "close" in lower_map:
            lookback = int(days)
            close_col = lower_map["close"]
            close = df[close_col].astype(float)
            if len(close) > lookback and close.iloc[-1] > 0 and close.iloc[-1 - lookback] > 0:
                return float(np.log(close.iloc[-1] / close.iloc[-1 - lookback]))
        return None

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        std = float(series.std(ddof=0))
        if std == 0.0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - float(series.mean())) / std

    def evaluate_cross_section(self, feature_map: dict[str, pd.DataFrame]) -> dict[str, dict[str, float | int]]:
        rows: list[dict[str, float | str]] = []
        horizon_keys = list(self.horizons.keys())
        for symbol, df in feature_map.items():
            values = {k: self._extract_return(df, k) for k in horizon_keys}
            if any(v is None for v in values.values()):
                continue
            vol = np.nan
            if "vol_realised" in df.columns and not df["vol_realised"].empty:
                vol = float(df["vol_realised"].iloc[-1])
            rows.append({"symbol": symbol, **{k: float(v) for k, v in values.items()}, "vol": vol})

        if not rows:
            return {}

        frame = pd.DataFrame(rows).set_index("symbol")
        z_components = {k: self._zscore(frame[k]) for k in horizon_keys}
        raw_score = sum(z_components[k] * float(self.weights.get(k, 0.0)) for k in horizon_keys)
        normalized_score = self._zscore(raw_score)

        ranked = (
            pd.DataFrame({"raw_score": raw_score, "normalized_score": normalized_score})
            .reset_index(names="symbol")
            .sort_values(["normalized_score", "symbol"], ascending=[False, True], kind="mergesort")
        )
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["confidence"] = np.tanh(ranked["normalized_score"].abs())

        out: dict[str, dict[str, float | int]] = {}
        for _, row in ranked.iterrows():
            symbol = str(row["symbol"])
            vol = float(frame.loc[symbol, "vol"])
            if not np.isfinite(vol) or vol <= 0:
                vol = 0.0
            out[symbol] = {
                "raw_score": float(row["raw_score"]),
                "normalized_score": float(row["normalized_score"]),
                "rank": int(row["rank"]),
                "confidence": float(row["confidence"]),
                "vol": vol,
            }
        return out

    def evaluate_many(self, feature_map: dict[str, pd.DataFrame], as_of: str | None = None) -> list[SignalRow]:
        ts = as_of or datetime.now(timezone.utc).isoformat()
        metrics = self.evaluate_cross_section(feature_map)

        rows: list[SignalRow] = []
        for symbol in sorted(feature_map):
            metric = metrics.get(symbol)
            if metric is None:
                rows.append(
                    SignalRow(
                        run_id=self.run_id,
                        timestamp=ts,
                        canonical_symbol=symbol,
                        asset_class=AssetClass.CRYPTO.value,
                        agent_id=AGENT_ID,
                        horizon=HORIZON,
                        status=SignalStatus.NO_SIGNAL.value,
                        rationale="insufficient_or_missing_required_returns;selected=False;no_signal_reason=missing_inputs",
                    )
                )
                continue

            rank = int(metric["rank"])
            raw_score = float(metric["raw_score"])
            norm_score = float(metric["normalized_score"])
            conf = float(metric["confidence"])
            selected = rank <= self.top_n
            no_signal_reason = "" if selected else "top_n_cutoff"

            rows.append(
                SignalRow(
                    run_id=self.run_id,
                    timestamp=ts,
                    canonical_symbol=symbol,
                    asset_class=AssetClass.CRYPTO.value,
                    agent_id=AGENT_ID,
                    horizon=HORIZON,
                    signal_score=float(np.tanh(norm_score)) if selected else 0.0,
                    confidence=conf if selected else 0.0,
                    required_inputs=list(self.horizons.keys()),
                    available_inputs=list({c.lower() for c in feature_map[symbol].columns}),
                    rationale=(
                        f"rank={rank};raw_score={raw_score:.6f};normalized_score={norm_score:.6f};"
                        f"selected={selected};no_signal_reason={no_signal_reason}"
                    ),
                    status=SignalStatus.OK.value if selected else SignalStatus.NO_SIGNAL.value,
                )
            )
        return rows
