"""yfinance provider implementation."""
from __future__ import annotations

import json
import logging
import random
import time

import pandas as pd

logger = logging.getLogger(__name__)

_YFINANCE_IMPORT_ERROR: Exception | None = None
try:
    import yfinance as yf
except Exception as exc:  # noqa: BLE001
    _YFINANCE_IMPORT_ERROR = exc

    class _MissingYFinance:
        def download(self, *args, **kwargs):
            raise ImportError(
                "yfinance is unavailable; install a compatible version for this Python runtime."
            ) from _YFINANCE_IMPORT_ERROR

    yf = _MissingYFinance()
from spectraquant.core.providers.base import DataProvider
from spectraquant.data.normalize import assert_price_frame, normalize_price_columns, normalize_price_frame
from spectraquant.data.retention import STATE_PATH

COOLDOWN_SECONDS = 900
MAX_RETRIES = 3
BACKOFF_BASE = 1.0
STATE_DOWNLOAD_KEY = "downloads"
STATE_PROVIDER_KEY = "yfinance"

_HEALTH = {"calls": 0, "success": 0, "failures": 0, "retries": 0, "cached": 0}


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(STATE_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2))
    tmp_path.replace(STATE_PATH)


def _cache_key(ticker: str, period: str, interval: str) -> str:
    return f"{ticker}:{period}:{interval}"


def _ensure_download_state(state: dict) -> dict:
    state.setdefault(STATE_DOWNLOAD_KEY, {})
    state[STATE_DOWNLOAD_KEY].setdefault(STATE_PROVIDER_KEY, {})
    return state[STATE_DOWNLOAD_KEY][STATE_PROVIDER_KEY]


def _lookup_last_fetch(state: dict, ticker: str, period: str, interval: str) -> float | None:
    provider_state = state.get(STATE_DOWNLOAD_KEY, {}).get(STATE_PROVIDER_KEY, {})
    if isinstance(provider_state, dict):
        tickers_state = provider_state.get("tickers", {})
        if isinstance(tickers_state, dict):
            record = tickers_state.get(ticker)
            if isinstance(record, dict):
                intervals = record.get("intervals", {})
                if isinstance(intervals, dict):
                    ts = intervals.get(interval)
                    if ts is not None:
                        return float(ts)
                ts = record.get("last_fetch")
                if ts is not None:
                    return float(ts)
        legacy_key = _cache_key(ticker, period, interval)
        legacy_ts = provider_state.get(legacy_key)
        if legacy_ts is not None:
            return float(legacy_ts)
    return None


def _record_last_fetch(state: dict, ticker: str, interval: str, timestamp: float) -> None:
    provider_state = _ensure_download_state(state)
    tickers_state = provider_state.setdefault("tickers", {})
    ticker_state = tickers_state.setdefault(ticker, {})
    ticker_state["last_fetch"] = timestamp
    intervals = ticker_state.setdefault("intervals", {})
    if isinstance(intervals, dict):
        intervals[interval] = timestamp


def _should_skip_download(
    ticker: str,
    period: str,
    interval: str,
    cooldown_seconds: int,
    force_download: bool,
) -> bool:
    if force_download or cooldown_seconds <= 0:
        return False
    state = _load_state()
    ts = _lookup_last_fetch(state, ticker, period, interval)
    if ts is None:
        return False
    if time.time() - float(ts) < cooldown_seconds:
        _HEALTH["cached"] += 1
        return True
    return False


def _record_download(ticker: str, period: str, interval: str) -> None:
    state = _load_state()
    _record_last_fetch(state, ticker, interval, time.time())
    _write_state(state)


def provider_health_summary() -> dict:
    return dict(_HEALTH)


def _compute_backoff(attempt: int, base: float = BACKOFF_BASE, cap: float = 10.0) -> float:
    jitter = random.uniform(0, 0.5)
    return min(cap, base * (2 ** (attempt - 1)) + jitter)


class YfinanceProvider(DataProvider):
    def __init__(self, config: dict | None = None) -> None:
        data_cfg = (config or {}).get("data", {}) if isinstance(config, dict) else {}
        self._cooldown_seconds = int(data_cfg.get("cooldown_seconds", COOLDOWN_SECONDS) or 0)
        self._force_download = bool(data_cfg.get("force_download", False))
        self._max_retries = int(data_cfg.get("max_retries", MAX_RETRIES) or MAX_RETRIES)

    def fetch_daily(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        _HEALTH["calls"] += 1
        if _should_skip_download(
            ticker,
            period,
            interval,
            cooldown_seconds=self._cooldown_seconds,
            force_download=self._force_download,
        ):
            logger.info("Skipping yfinance download for %s due to cooldown window.", ticker)
            return pd.DataFrame()

        df = pd.DataFrame()
        for attempt in range(1, self._max_retries + 1):
            try:
                df = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if df is not None and not df.empty:
                    _record_download(ticker, period, interval)
                    _HEALTH["success"] += 1
                    df = df.reset_index()
                    df = normalize_price_columns(df, ticker)
                    df = normalize_price_frame(df)
                    assert_price_frame(df, context=f"yfinance daily {ticker}")
                    return df
                break
            except Exception as exc:  # noqa: BLE001
                _HEALTH["retries"] += 1
                logger.warning(
                    "yfinance fetch failed for %s (attempt %s/%s): %s",
                    ticker,
                    attempt,
                    self._max_retries,
                    exc,
                )
                time.sleep(_compute_backoff(attempt))
        _HEALTH["failures"] += 1
        return pd.DataFrame()

    def fetch_intraday(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        _HEALTH["calls"] += 1
        if _should_skip_download(
            ticker,
            period,
            interval,
            cooldown_seconds=self._cooldown_seconds,
            force_download=self._force_download,
        ):
            logger.info("Skipping yfinance download for %s due to cooldown window.", ticker)
            return pd.DataFrame()

        df = pd.DataFrame()
        for attempt in range(1, self._max_retries + 1):
            try:
                df = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    prepost=False,
                    threads=False,
                )
                if df is not None and not df.empty:
                    _record_download(ticker, period, interval)
                    _HEALTH["success"] += 1
                    df = df.reset_index()
                    df = normalize_price_columns(df, ticker)
                    df = normalize_price_frame(df)
                    assert_price_frame(df, context=f"yfinance intraday {ticker}")
                    return df
                break
            except Exception as exc:  # noqa: BLE001
                _HEALTH["retries"] += 1
                logger.warning(
                    "yfinance intraday fetch failed for %s (attempt %s/%s): %s",
                    ticker,
                    attempt,
                    self._max_retries,
                    exc,
                )
                time.sleep(_compute_backoff(attempt))
        _HEALTH["failures"] += 1
        return pd.DataFrame()
