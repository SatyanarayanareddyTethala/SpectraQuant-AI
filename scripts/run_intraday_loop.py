#!/usr/bin/env python
"""Run an intraday refresh loop with retention and lightweight pipeline."""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from spectraquant.config import get_config
from spectraquant.cli.main import _compute_rsi
from spectraquant.core.io import write_predictions, write_signals
from spectraquant.core.ranking import add_rank, normalize_scores
from spectraquant.core.schema import validate_predictions, validate_signals
from spectraquant.core.time import ensure_datetime_column
from spectraquant.data.normalize import normalize_price_columns
from scripts.fetch_intraday_1m import fetch_intraday
from spectraquant.intraday.learner import IntradayLearner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = ROOT / ".intraday_lock"
STOP_FILE = ROOT / ".stop_intraday"
LOG_FILE = ROOT / "logs" / "intraday_loop.log"
INTRADAY_DIR = ROOT / "data" / "intraday_1m"
FEATURE_DIR = ROOT / "data" / "intraday_features"
PRED_DIR = ROOT / "reports" / "predictions"
SIGNAL_DIR = ROOT / "reports" / "signals"
PORTF_DIR = ROOT / "reports" / "portfolio"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
PORTF_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            data = json.loads(LOCK_FILE.read_text())
            pid = data.get("pid")
            if pid and not Path(f"/proc/{pid}").exists():
                logger.info("Stale lock detected; recovering lock from pid %s", pid)
            else:
                logger.warning("Intraday loop already running with pid %s", pid)
                return False
        except Exception:  # noqa: BLE001
            logger.warning("Unable to parse existing lock; recovering")
    LOCK_FILE.write_text(json.dumps({"pid": os.getpid(), "timestamp": datetime.utcnow().isoformat()}))
    return True


def _release_lock() -> None:
    if LOCK_FILE.exists():
        LOCK_FILE.unlink(missing_ok=True)


def _within_market_hours(now: datetime, cfg: dict) -> bool:
    intr_cfg = cfg.get("intraday", {})
    only_market = bool(intr_cfg.get("only_run_during_market_hours", True))
    if not only_market:
        return True
    hour = now.hour
    # Simple windows: NSE 9:15-15:30 IST (~3:45-10:00 UTC), LSE 8:00-16:30 UK time (~8-16 UTC)
    return 3 <= hour <= 17


def _load_intraday_df(ticker: str) -> pd.DataFrame:
    path = INTRADAY_DIR / f"{ticker}.parquet"
    if path.exists():
        df = normalize_price_columns(pd.read_parquet(path), ticker)
        df = ensure_datetime_column(df, "date")
        return df
    return pd.DataFrame()


def compute_intraday_features(ticker: str, retention_days: int) -> pd.DataFrame:
    df = _load_intraday_df(ticker)
    if df.empty:
        return df
    df = ensure_datetime_column(df, "date")
    df = df.sort_values("date")
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    df = df[df["date"] >= pd.Timestamp(cutoff, tz="UTC")]
    if "close" not in df.columns:
        return pd.DataFrame()
    close = pd.to_numeric(df["close"], errors="coerce")
    returns_1m = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_df = pd.DataFrame({"date": df["date"], "ticker": ticker, "close": close})
    feature_df["returns_1m"] = returns_1m
    feature_df["vol_30m"] = returns_1m.rolling(30, min_periods=5).std()
    feature_df["vol_120m"] = returns_1m.rolling(120, min_periods=20).std()
    feature_df["ma_20"] = close.rolling(20, min_periods=5).mean()
    feature_df["ma_60"] = close.rolling(60, min_periods=10).mean()
    feature_df["ma_120"] = close.rolling(120, min_periods=20).mean()
    feature_df["momentum_15m"] = close.pct_change(15)
    feature_df["momentum_60m"] = close.pct_change(60)
    feature_df["rsi_14"] = _compute_rsi(close)
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        feature_df["volume_z"] = (vol - vol.rolling(60, min_periods=10).mean()) / vol.rolling(
            60, min_periods=10
        ).std()
    feature_df = feature_df.ffill().dropna(how="all")
    feature_df = ensure_datetime_column(feature_df, "date")
    out_path = FEATURE_DIR / f"{ticker}.parquet"
    tmp = out_path.with_suffix(".tmp")
    feature_df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return feature_df


def latest_feature_snapshot(tickers: List[str], retention_days: int) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        df = compute_intraday_features(ticker, retention_days)
        if df.empty:
            continue
        rows.append(df.tail(1))
    if not rows:
        return pd.DataFrame()
    snapshot = pd.concat(rows, ignore_index=True)
    snapshot = ensure_datetime_column(snapshot, "date")
    snapshot_path = FEATURE_DIR / "latest_snapshot.csv"
    snapshot.to_csv(snapshot_path, index=False)
    return snapshot


def predict_intraday(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot
    snapshot = ensure_datetime_column(snapshot, "date")
    preds = snapshot[["ticker", "date"]].copy()
    learner = IntradayLearner()
    scores = learner.predict(snapshot)
    preds["score"] = normalize_scores(scores)
    preds["probability"] = 1 / (1 + np.exp(-(preds["score"] - 50) / 12))
    preds["horizon"] = "60m"
    preds["model_version"] = learner.state.metadata.get("model_version", "intraday_v1")
    preds["factor_set_version"] = learner.state.metadata.get("factor_set_version", "intraday_default")
    preds["regime"] = "intraday"
    last_close = snapshot.get("ma_20", pd.Series(np.nan, index=preds.index))
    preds["predicted_return"] = snapshot.get("momentum_60m", pd.Series(0, index=preds.index)).fillna(0)
    preds["predicted_return_1d"] = preds["predicted_return"]
    preds["target_price"] = last_close * (1 + preds["predicted_return"])
    preds["target_price_1d"] = last_close * (1 + preds["predicted_return_1d"])
    preds["confidence"] = 1 - snapshot.get("vol_30m", pd.Series(0.0, index=preds.index)).fillna(0)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    latest_path = PRED_DIR / "intraday_predictions_latest.csv"
    rolling_path = PRED_DIR / f"intraday_predictions_{timestamp}.csv"
    preds = validate_predictions(preds)
    write_predictions(preds, latest_path)
    write_predictions(preds, rolling_path)
    return preds


def generate_intraday_signals(preds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if preds.empty:
        return preds
    preds = validate_predictions(preds)
    thresholds = cfg.get("intraday", {}).get(
        "signal_thresholds", {"buy": 0.6, "sell": 0.4}
    )
    buy = float(thresholds.get("buy", 0.6))
    sell = float(thresholds.get("sell", 0.4))
    if buy <= 1:
        buy *= 100
    if sell <= 1:
        sell *= 100
    def to_signal(score: float) -> str:
        if score >= buy:
            return "BUY"
        if score <= sell:
            return "SELL"
        return "HOLD"
    signals = preds.copy()
    if "score" not in signals:
        signals["score"] = signals.get("probability", pd.Series(0.5, index=signals.index))
    signals["score"] = normalize_scores(signals["score"])
    signals["signal"] = signals["score"].apply(to_signal)
    signals["reason_text"] = [
        f"Momentum/RSI blend; score={p:.2f}" for p in signals["score"].fillna(50.0)
    ]
    signals = add_rank(signals, "score")
    latest = SIGNAL_DIR / "intraday_signals_latest.csv"
    signals = validate_signals(signals)
    write_signals(signals, latest)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    write_signals(signals, SIGNAL_DIR / f"intraday_signals_{timestamp}.csv")
    return signals


def build_intraday_portfolio(signals: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if signals.empty:
        return signals
    signals = validate_signals(signals)
    top_n = int(cfg.get("intraday", {}).get("top_n", 5))
    buys = signals[signals["signal"] == "BUY"].copy()
    buys = buys.sort_values("score", ascending=False).head(top_n)
    if buys.empty:
        return pd.DataFrame()
    n = len(buys)
    buys["weight"] = 1 / n
    buys["expected_return"] = buys.get("predicted_return_1d", pd.Series(0, index=buys.index)).fillna(0)
    buys["risk_proxy"] = (1 - buys.get("confidence", pd.Series(0, index=buys.index)).fillna(0)).abs()
    buys = ensure_datetime_column(buys, "date")
    latest = PORTF_DIR / "intraday_portfolio_latest.csv"
    buys.to_csv(latest, index=False)
    return buys


def run_cycle(cfg: dict) -> None:
    intr_cfg = cfg.get("intraday", {})
    tickers = intr_cfg.get("tickers") or cfg.get("data", {}).get("tickers", [])
    if not tickers:
        raise ValueError("Intraday tickers must be explicitly configured")
    retention = int(intr_cfg.get("retention_days", 7))
    interval = intr_cfg.get("interval", "5m")
    rows_added = 0
    for ticker in tickers:
        before = _load_intraday_df(ticker)
        fetch_intraday(ticker, retention_days=retention, interval=interval)
        after = _load_intraday_df(ticker)
        if not after.empty:
            rows_added += max(0, len(after) - len(before))
    snapshot = latest_feature_snapshot(tickers, retention)
    learner = IntradayLearner()
    learner.update_from_features(FEATURE_DIR)
    preds = predict_intraday(snapshot)
    signals = generate_intraday_signals(preds, cfg)
    build_intraday_portfolio(signals, cfg)
    logger.info("Cycle complete: tickers=%s rows_added=%s preds=%s signals=%s", len(tickers), rows_added, len(preds), len(signals))


def main():
    cfg = get_config()
    intr_cfg = cfg.get("intraday", {})
    refresh = int(intr_cfg.get("refresh_seconds", 60))
    off_refresh = int(intr_cfg.get("off_hours_refresh_seconds", 1800))

    if not _acquire_lock():
        return
    try:
        while True:
            if STOP_FILE.exists():
                logger.info("Stop file detected; exiting loop")
                break
            start = time.time()
            now = datetime.utcnow()
            within_hours = _within_market_hours(now, cfg)
            interval = refresh if within_hours else off_refresh
            try:
                run_cycle(cfg)
            except Exception as exc:  # noqa: BLE001
                logger.error("Cycle error: %s", exc)
            elapsed = time.time() - start
            sleep_for = max(1, interval - elapsed)
            logger.info("Sleeping for %.2fs (within_hours=%s)", sleep_for, within_hours)
            time.sleep(sleep_for)
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
