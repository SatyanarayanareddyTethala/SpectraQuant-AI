"""Crypto feature engine for SpectraQuant-AI-V3.

Computes technical, news, and market-context features from OHLCV DataFrames.
All features are computed in-memory and returned as new columns appended to the
input DataFrame. The original OHLCV columns are preserved.

This module must never import from ``spectraquant_v3.equities``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spectraquant_v3.core.errors import DataSchemaError, EmptyPriceDataError

FEATURE_VERSION = "2.1.0"

# Minimum rows needed to compute meaningful features
_MIN_ROWS = 14


@dataclass(frozen=True)
class FeatureManifestMetadata:
    """Metadata describing the generated crypto feature set."""

    feature_version: str
    nan_policy: str
    feature_set: str


def _validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> None:
    label = f" for '{symbol}'" if symbol else ""
    if df.empty:
        raise EmptyPriceDataError(
            f"Feature engine{label}: DataFrame is empty."
        )
    required = {"open", "high", "low", "close", "volume"}
    cols = {c.lower() for c in df.columns}
    missing = required - cols
    if missing:
        raise DataSchemaError(
            f"Feature engine{label}: missing columns {sorted(missing)}."
        )


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothed moving average."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _ensure_datetime_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    out = out.sort_index()
    if out.index.has_duplicates:
        raise DataSchemaError(f"{label}: index contains duplicate timestamps.")
    return out


def _merge_news_features(
    out: pd.DataFrame,
    news_df: pd.DataFrame | None,
    symbol: str,
    merge_tolerance: str,
) -> pd.DataFrame:
    """Merge rolling news sentiment/count features with no lookahead leakage."""
    cols = [
        "news_sentiment_1h",
        "news_sentiment_24h",
        "news_sentiment_3d",
        "news_volume_24h",
        "news_shock_zscore",
    ]
    default = pd.DataFrame(index=out.index, data={c: np.nan for c in cols})
    if news_df is None or news_df.empty:
        return default

    ndf = _ensure_datetime_index(news_df, "news_df")
    ndf.columns = [c.lower() for c in ndf.columns]

    if "symbol" in ndf.columns and symbol:
        ndf = ndf[ndf["symbol"].astype(str).str.upper() == symbol.upper()]
        if ndf.empty:
            return default

    if "sentiment" not in ndf.columns:
        raise DataSchemaError("news_df: missing required 'sentiment' column.")

    sentiment = ndf["sentiment"].astype(float).sort_index()
    counts = pd.Series(1.0, index=sentiment.index)

    cumulative = pd.DataFrame(index=sentiment.index)
    cumulative["sent_cum"] = sentiment.cumsum()
    cumulative["cnt_cum"] = counts.cumsum()

    out_idx = out.sort_index().index

    def _asof_frame(ts: pd.DatetimeIndex) -> pd.DataFrame:
        probe = pd.DataFrame(index=ts, data={"_ts": ts})
        merged = pd.merge_asof(
            probe,
            cumulative,
            left_on="_ts",
            right_index=True,
            direction="backward",
            tolerance=pd.Timedelta(merge_tolerance),
            allow_exact_matches=True,
        )
        return merged[["sent_cum", "cnt_cum"]]

    current = _asof_frame(out_idx)

    def _window_mean(window: str) -> pd.Series:
        start = _asof_frame(out_idx - pd.Timedelta(window))
        start.index = out_idx
        sent_sum = current["sent_cum"] - start["sent_cum"].fillna(0.0)
        cnt_sum = current["cnt_cum"] - start["cnt_cum"].fillna(0.0)
        return sent_sum / cnt_sum.replace(0.0, np.nan)

    def _window_count(window: str) -> pd.Series:
        start = _asof_frame(out_idx - pd.Timedelta(window))
        start.index = out_idx
        return current["cnt_cum"] - start["cnt_cum"].fillna(0.0)

    aligned = pd.DataFrame(index=out_idx)
    aligned["news_sentiment_1h"] = _window_mean("1h")
    aligned["news_sentiment_24h"] = _window_mean("24h")
    aligned["news_sentiment_3d"] = _window_mean("3d")
    aligned["news_volume_24h"] = _window_count("24h")

    vol_mu = aligned["news_volume_24h"].rolling(24, min_periods=6).mean()
    vol_sigma = aligned["news_volume_24h"].rolling(24, min_periods=6).std(ddof=0).replace(0, np.nan)
    aligned["news_shock_zscore"] = (aligned["news_volume_24h"] - vol_mu) / vol_sigma

    return aligned[cols]


def _compute_context_features(
    out: pd.DataFrame,
    context_df: pd.DataFrame | None,
    benchmark_symbol: str,
) -> pd.DataFrame:
    """Compute per-symbol and cross-market context factors."""
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    context = pd.DataFrame(index=out.index)
    ma_50 = close.rolling(window=50, min_periods=10).mean()
    ma_200 = close.rolling(window=200, min_periods=20).mean()
    context["price_vs_ma_50"] = close / ma_50.replace(0, np.nan) - 1.0
    context["price_vs_ma_200"] = close / ma_200.replace(0, np.nan) - 1.0

    log_ret = np.log(close / close.shift(1))
    context["rolling_volatility"] = log_ret.rolling(20, min_periods=2).std() * (252 ** 0.5)

    vol_mu = volume.rolling(30, min_periods=5).mean()
    vol_sigma = volume.rolling(30, min_periods=5).std(ddof=0).replace(0, np.nan)
    context["volume_zscore"] = (volume - vol_mu) / vol_sigma

    context["btc_regime"] = np.nan
    context["market_breadth"] = np.nan
    context["alt_btc_relative_strength"] = np.nan

    if context_df is None or context_df.empty:
        return context

    cdf = _ensure_datetime_index(context_df, "context_df")
    cdf.columns = [c.lower() for c in cdf.columns]

    if "btc_close" in cdf.columns:
        btc_ret = np.log(cdf["btc_close"].astype(float) / cdf["btc_close"].astype(float).shift(1))
        regime = np.sign(btc_ret.rolling(20, min_periods=5).mean()).replace(0, 0.0)
        context["btc_regime"] = regime.reindex(context.index, method="ffill")

    if "market_breadth" in cdf.columns:
        context["market_breadth"] = cdf["market_breadth"].astype(float).reindex(context.index, method="ffill")
    elif "advancing_ratio" in cdf.columns:
        context["market_breadth"] = cdf["advancing_ratio"].astype(float).reindex(context.index, method="ffill")

    if "symbol" in cdf.columns and "close" in cdf.columns:
        benchmark = cdf[cdf["symbol"].str.upper() == benchmark_symbol.upper()]["close"].astype(float)
        if not benchmark.empty:
            bench_close = benchmark.groupby(level=0).last()
            sym_rel = close / close.shift(20)
            btc_rel = bench_close.reindex(context.index, method="ffill") / bench_close.reindex(context.index, method="ffill").shift(20)
            context["alt_btc_relative_strength"] = sym_rel - btc_rel
    elif "btc_close" in cdf.columns:
        btc_close = cdf["btc_close"].astype(float).reindex(context.index, method="ffill")
        sym_rel = close / close.shift(20)
        btc_rel = btc_close / btc_close.shift(20)
        context["alt_btc_relative_strength"] = sym_rel - btc_rel

    return context


def _apply_nan_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Apply deterministic NaN handling."""
    if policy == "keep":
        return df
    if policy == "ffill_bfill":
        filled = df.copy()
        filled = filled.ffill().bfill()
        return filled
    if policy == "zero":
        return df.fillna(0.0)
    raise ValueError(f"Unsupported nan_policy: {policy}")


def compute_features(
    df: pd.DataFrame,
    symbol: str = "",
    momentum_window: int = 20,
    rsi_period: int = 14,
    vol_window: int = 20,
    volume_ma_window: int = 20,
    atr_window: int = 14,
    news_df: pd.DataFrame | None = None,
    context_df: pd.DataFrame | None = None,
    news_merge_tolerance: str = "7d",
    nan_policy: str = "keep",
    benchmark_symbol: str = "BTC",
) -> pd.DataFrame:
    """Compute crypto features and return an enriched DataFrame."""
    _validate_ohlcv(df, symbol)

    # Normalise column names to lower-case for uniform access
    out = _ensure_datetime_index(df, "ohlcv")
    out.columns = [c.lower() for c in out.columns]

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    # 1-day log return
    log_ret = np.log(close / close.shift(1))
    out["ret_1d"] = log_ret

    # N-day momentum (cumulative log return)
    out[f"ret_{momentum_window}d"] = np.log(
        close / close.shift(momentum_window)
    )

    out["rsi"] = _rsi(close, period=rsi_period)

    vol_ma = volume.rolling(window=volume_ma_window, min_periods=1).mean()
    out["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_window, min_periods=1).mean()
    out["atr_norm"] = atr / close.replace(0, np.nan)

    out["vol_realised"] = log_ret.rolling(window=vol_window, min_periods=2).std() * (
        252 ** 0.5
    )

    news_features = _merge_news_features(
        out=out,
        news_df=news_df,
        symbol=symbol,
        merge_tolerance=news_merge_tolerance,
    )
    for col in news_features.columns:
        out[col] = news_features[col]

    context_features = _compute_context_features(
        out=out,
        context_df=context_df,
        benchmark_symbol=benchmark_symbol,
    )
    for col in context_features.columns:
        out[col] = context_features[col]

    return _apply_nan_policy(out, nan_policy)


class CryptoFeatureEngine:
    """Stateless feature engine for the crypto pipeline."""

    def __init__(
        self,
        momentum_window: int = 20,
        rsi_period: int = 14,
        vol_window: int = 20,
        volume_ma_window: int = 20,
        atr_window: int = 14,
        news_merge_tolerance: str = "7d",
        nan_policy: str = "keep",
        feature_version: str = FEATURE_VERSION,
    ) -> None:
        self.momentum_window = momentum_window
        self.rsi_period = rsi_period
        self.vol_window = vol_window
        self.volume_ma_window = volume_ma_window
        self.atr_window = atr_window
        self.news_merge_tolerance = news_merge_tolerance
        self.nan_policy = nan_policy
        self.feature_version = feature_version

    @classmethod
    def from_config(cls, cfg: dict) -> "CryptoFeatureEngine":
        """Build from merged crypto config."""
        signals_cfg = cfg.get("crypto", {}).get("signals", {})
        feature_cfg = cfg.get("crypto", {}).get("features", {})
        return cls(
            momentum_window=int(signals_cfg.get("momentum_lookback", 20)),
            rsi_period=int(signals_cfg.get("rsi_period", 14)),
            news_merge_tolerance=str(feature_cfg.get("news_merge_tolerance", "7d")),
            nan_policy=str(feature_cfg.get("nan_policy", "keep")),
            feature_version=str(feature_cfg.get("feature_version", FEATURE_VERSION)),
        )

    def manifest_metadata(self) -> FeatureManifestMetadata:
        """Return metadata to attach to dataset manifests."""
        return FeatureManifestMetadata(
            feature_version=self.feature_version,
            nan_policy=self.nan_policy,
            feature_set="crypto_ohlcv_news_context",
        )

    def transform(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        news_df: pd.DataFrame | None = None,
        context_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute and return features for *df*."""
        return compute_features(
            df,
            symbol=symbol,
            momentum_window=self.momentum_window,
            rsi_period=self.rsi_period,
            vol_window=self.vol_window,
            volume_ma_window=self.volume_ma_window,
            atr_window=self.atr_window,
            news_df=news_df,
            context_df=context_df,
            news_merge_tolerance=self.news_merge_tolerance,
            nan_policy=self.nan_policy,
        )

    def transform_many(
        self,
        price_map: dict[str, pd.DataFrame],
        news_map: dict[str, pd.DataFrame] | None = None,
        context_map: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Compute features for multiple symbols."""
        result: dict[str, pd.DataFrame] = {}
        for sym, df in price_map.items():
            try:
                result[sym] = self.transform(
                    df,
                    symbol=sym,
                    news_df=(news_map or {}).get(sym),
                    context_df=(context_map or {}).get(sym),
                )
            except (DataSchemaError, EmptyPriceDataError):
                continue
        return result
