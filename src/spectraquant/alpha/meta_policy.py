from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np


def detect_regime(index_prices: pd.DataFrame, config: dict) -> dict:
    close = pd.to_numeric(index_prices.get("close", index_prices.get("Close")), errors="coerce").dropna()
    fast = int(config.get("meta_policy", {}).get("regime", {}).get("trend_fast", 20))
    slow = int(config.get("meta_policy", {}).get("regime", {}).get("trend_slow", 50))
    hv = float(config.get("meta_policy", {}).get("regime", {}).get("high_vol_threshold", 0.25))
    trend = (close.tail(fast).mean() / close.tail(slow).mean() - 1) if len(close) >= slow else 0.0
    vol = float(close.pct_change().tail(slow).std() * np.sqrt(252)) if len(close) > 2 else 0.0
    label = "risk_on" if trend > 0 and vol < hv else "risk_off" if trend < 0 and vol >= hv else "neutral"
    return {"trend": float(trend), "vol": float(vol), "label": label}


def load_expert_performance(history_dir: str | Path, lookback_days: int, decay: float) -> dict[str, dict]:
    path = Path(history_dir)
    metrics = {}
    for file in sorted(path.glob("expert_scores_*.csv")):
        df = pd.read_csv(file)
        if "expert_name" not in df.columns or "expected_return" not in df.columns:
            continue
        for expert, sub in df.groupby("expert_name"):
            ret = pd.to_numeric(sub["expected_return"], errors="coerce").fillna(0.0)
            weights = np.power(decay, np.arange(len(ret))[::-1])
            score = float(np.average(ret, weights=weights)) if len(ret) else 0.0
            metrics[expert] = {"perf_score": score, "trades": int(len(ret.tail(lookback_days)))}
    return metrics


def compute_expert_weights(method: str, regime: dict, perf_metrics: dict[str, dict], config: dict) -> dict[str, float]:
    experts = config.get("experts", {}).get("list", list(perf_metrics.keys()))
    if method == "contextual_bandit":
        raise NotImplementedError("meta_policy.method=contextual_bandit is not implemented in v1")
    if not experts:
        return {}
    floor = float(config.get("meta_policy", {}).get("weight_floor", 0.05))
    cap = float(config.get("meta_policy", {}).get("weight_cap", 0.60))
    min_trades = int(config.get("meta_policy", {}).get("min_trades_for_trust", 20))
    if method == "rule_based":
        pref = {"risk_on": {"trend", "momentum"}, "risk_off": {"volatility", "value"}, "neutral": set(experts)}
        preferred = pref.get(regime.get("label", "neutral"), set(experts))
        raw = {e: (2.0 if e in preferred else 1.0) for e in experts}
    else:
        eta = 5.0
        raw = {}
        for e in experts:
            perf = perf_metrics.get(e, {}).get("perf_score", 0.0)
            trades = perf_metrics.get(e, {}).get("trades", 0)
            trust = 1.0 if trades >= min_trades else max(trades / max(min_trades, 1), 0.1)
            raw[e] = float(np.exp(eta * perf) * trust)
    total = sum(raw.values()) or 1.0
    weights = {e: v / total for e, v in raw.items()}
    weights = {e: min(cap, max(floor, w)) for e, w in weights.items()}
    # Re-normalize while preserving caps/floors.
    remaining = 1.0
    free = set(weights.keys())
    final = {e: 0.0 for e in weights}
    while free:
        subtotal = sum(weights[e] for e in free) or 1.0
        adjusted = {e: weights[e] / subtotal * remaining for e in free}
        moved = False
        for e, w in adjusted.items():
            if w > cap:
                final[e] = cap
                remaining -= cap
                free.remove(e)
                moved = True
                break
            if w < floor:
                final[e] = floor
                remaining -= floor
                free.remove(e)
                moved = True
                break
        if not moved:
            for e, w in adjusted.items():
                final[e] = w
            break
    s = sum(final.values()) or 1.0
    return {e: w / s for e, w in final.items()}


def blend_signals(expert_outputs: list[pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    if not expert_outputs:
        return pd.DataFrame(columns=["ticker", "date", "horizon", "score", "signal", "rank"])
    df = pd.concat(expert_outputs, ignore_index=True)
    df["weight"] = df["expert_name"].map(weights).fillna(0.0)
    df["wscore"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0) * df["weight"]
    df["wconf"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0) * df["weight"]
    out = df.groupby(["ticker", "date", "horizon"], as_index=False).agg(score=("wscore", "sum"), confidence=("wconf", "sum"))
    out["signal"] = np.where(out["score"] > 0.1, "BUY", np.where(out["score"] < -0.1, "SELL", "HOLD"))
    out["rank"] = out.groupby("horizon")["score"].rank(method="dense", ascending=False)
    return out


def persist_meta_outputs(weights: dict[str, float], regime: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    w = out / f"expert_weights_{ts}.json"
    d = out / f"arbiter_decision_{ts}.json"
    w.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    d.write_text(json.dumps({"regime": regime, "weights": weights}, indent=2), encoding="utf-8")
    return w, d
