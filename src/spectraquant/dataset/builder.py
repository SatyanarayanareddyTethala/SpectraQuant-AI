"""Dataset builder for multi-method ML pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from spectraquant.data.normalize import normalize_price_columns, normalize_price_frame, assert_price_frame
from spectraquant.dataset.labels import compute_forward_returns
from spectraquant.features.ohlcv_features import compute_ohlcv_features
from spectraquant.regime.simple_regime import compute_regime


DATASET_DIR = Path("reports/datasets")
RUN_REPORTS_DIR = Path("reports/run")
PRICES_DIR = Path("data/prices")
logger = logging.getLogger(__name__)


def _hash_universe(tickers: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for ticker in sorted({str(t) for t in tickers}):
        digest.update(ticker.encode("utf-8"))
    return digest.hexdigest()


def _load_price_frame(ticker: str) -> pd.DataFrame:
    candidates = [
        PRICES_DIR / f"{ticker}.parquet",
        PRICES_DIR / f"{ticker}.csv",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            df = normalize_price_columns(df, ticker)
            df = normalize_price_frame(df)
            assert_price_frame(df, context=f"dataset_builder:{ticker}")
            return df
    raise FileNotFoundError(f"No price history found for {ticker} in {PRICES_DIR}")


def _filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None
    if start_ts is not None:
        df = df.loc[df.index >= start_ts]
    if end_ts is not None:
        df = df.loc[df.index <= end_ts]
    return df


def _write_manifest(run_id: str, payload: dict) -> Path:
    run_dir = RUN_REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing.update(payload)
        payload = existing
    else:
        payload.setdefault("run_id", run_id)
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(payload, indent=2, default=str))
    return manifest_path


def build_dataset(
    tickers: list[str],
    start: str,
    end: str,
    horizons: list[int] | None = None,
) -> Path:
    """Build dataset from cached OHLCV prices and save as parquet."""

    if horizons is None:
        horizons = [5, 20]

    if not tickers:
        raise ValueError("Tickers list is empty; cannot build dataset")

    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        df = _load_price_frame(ticker)
        df = _filter_window(df, start, end)
        if df.empty:
            continue

        features = compute_ohlcv_features(df)
        if features.empty:
            continue

        regime = compute_regime(
            pd.DataFrame(
                {
                    "close": df.loc[features.index, "close"],
                    "vol_20": features["vol_20"],
                    "sma_50": features["sma_50"],
                },
                index=features.index,
            )
        )

        labels = {}
        for horizon in horizons:
            fwd = compute_forward_returns(df.loc[features.index, "close"], horizon)
            labels[f"fwd_ret_{horizon}d"] = fwd
            labels[f"up_{horizon}d"] = fwd > 0

        combined = features.copy()
        for name, series in labels.items():
            combined[name] = series
        combined["regime"] = regime
        combined["ticker"] = ticker
        combined = combined.dropna()
        if combined.empty:
            continue
        combined = combined.reset_index().rename(columns={"date": "date"})
        rows.append(combined)

    if not rows:
        raise ValueError("No data available to build dataset")

    dataset = pd.concat(rows, ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"], utc=True, errors="coerce")
    dataset = dataset.dropna(subset=["date"])
    dataset = dataset.sort_values(["date", "ticker"])
    dataset = dataset.set_index(["date", "ticker"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = DATASET_DIR / f"dataset_{run_id}.parquet"
    dataset_format = "parquet"
    try:
        dataset.to_parquet(dataset_path)
    except ImportError as exc:
        logger.warning("Parquet engine unavailable (%s); falling back to CSV.", exc)
        dataset_path = DATASET_DIR / f"dataset_{run_id}.csv"
        dataset.to_csv(dataset_path)
        dataset_format = "csv"

    features_list = [col for col in dataset.columns if col not in {"regime"} and not col.startswith("fwd_") and not col.startswith("up_")]
    metadata = {
        "dataset": {
            "path": str(dataset_path),
            "features": features_list,
            "horizons": horizons,
            "format": dataset_format,
            "date_range": {
                "start": dataset.index.get_level_values(0).min().isoformat(),
                "end": dataset.index.get_level_values(0).max().isoformat(),
            },
            "universe_hash": _hash_universe(tickers),
        }
    }
    _write_manifest(run_id, metadata)
    return dataset_path


__all__ = ["build_dataset"]
