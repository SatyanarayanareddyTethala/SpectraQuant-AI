"""Configuration utilities for SpectraQuant."""
from __future__ import annotations

import logging
import os
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

import yaml

from spectraquant.universe import parse_universe_override, resolve_universe

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.getenv("SPECTRAQUANT_CONFIG", "config.yaml"))

DEFAULT_TICKERS: list[str] = []


DEFAULT_CONFIG: Dict[str, Any] = {
    "alpha": {
        "enabled": True,
        "weights": {
            "momentum": 0.35,
            "trend": 0.25,
            "volatility": 0.20,
            "value": 0.20,
        },
    },
    "portfolio": {
        "rebalance": "monthly",
        "weighting": "equal",
        "alpha_threshold": 0.0,
        "volatility_target": None,
        "max_asset_weight": None,
        "sector_limits": {},
        "sector_map": {},
        "top_k": 20,
        "min_weight_threshold": 0.0,
        "liquidity_min_volume": None,
        "max_positions": 20,
        "max_weight": None,
        "max_turnover": None,
        "policies": {},
        "policy": {"auto_repair": False},
        "horizon": "1d",
    },
    "mlops": {
        "auto_retrain": True,
        "retrain_interval_days": 7,
        "min_improvement": 0.01,
        "seed": 42,
        "walk_forward_splits": 3,
        "drift_threshold": 0.6,
        "drift_recent_fraction": 0.1,
    },
    "qa": {
        "min_price_rows": 30,
        "min_eligible_tickers": 10,
        "stale_tolerance_minutes": 30,
        "flatline_window": 5,
        "max_abs_daily_return": 0.8,
        "split_like_enabled": True,
        "split_like_factors": [2, 3, 4, 5, 10, 20],
        "split_like_tolerance": 0.03,
        "min_expected_return_std_daily": 1e-6,
        "min_expected_return_std_intraday": 1e-9,
        "min_volume": 0,
        "max_missing_pct": 0.2,
        "force_pass_tests": False,
    },
    "research_mode": False,
    "data": {
        "tickers": [],
        "synthetic": False,
        "source": "yfinance",
        "provider": "yfinance",
        "prices_dir": "data/prices",
        "batch_size": 50,
        "batch_sleep_seconds": 3,
        "max_retries": 5,
        "cooldown_seconds": 0,
        "force_download": False,
        "daily_retention_years": 5,
        "max_tickers_per_run": 2000,
    },
    "data_retention": {
        "initial_years_for_training": 10,
        "post_training_years_to_keep": 5,
        "enforce_on_download": True,
        "enforce_on_train_complete": True,
    },
    "intraday": {
        "enabled": True,
        "interval": "5m",
        "lookback_days": 30,
        "fallback_intervals": ["15m", "30m", "60m"],
        "refresh_seconds": 300,
        "retention_days": 7,
        "tickers": [],
        "only_run_during_market_hours": True,
        "off_hours_refresh_seconds": 1800,
        "signal_thresholds": {"buy": 0.60, "sell": 0.40},
        "top_n": 5,
    },
    "execution": {
        "mode": "eod",
        "slippage_bps": 5,
        "transaction_cost_bps": 1,
        "volume_participation": 0.1,
    },
    "predictions": {
        "daily_horizons": ["1d", "5d", "20d"],
        "intraday_horizons": ["5m", "30m", "60m"],
    },
    "explain": {"enabled": False},
    "perf": {"max_seconds": 60, "max_mb": 512, "stages": {"download": {"max_seconds": 1800}}},
    "test_mode": {"enabled": False, "limit_tickers": 2},
    "sentiment": {
        "enabled": False,
        "provider": "newsapi",
        "lookback_days": 30,
        "max_articles_per_ticker": 50,
        "newsapi_max_lookback_days": 30,
        "use_news": True,
        "use_social": False,
        "refresh_cache": True,
        "social": {"provider": "stocktwits"},
    },
    "news_universe": {
        "enabled": False,
        "lookback_hours": 12,
        "max_candidates": 50,
        "min_liquidity_avg_volume": 200000,
        "min_source_rank": 0.0,
        "sentiment_model": "finbert",
        "require_price_confirmation": True,
        "confirmation": {
            "method": "gap_or_volume",
            "gap_abs_return_threshold": 0.015,
            "volume_z_threshold": 1.5,
            "lookback_days": 20,
        },
        "recency_decay_half_life_hours": 6,
        "source_weights_path": "data/news/source_weights.csv",
        "aliases_path": "data/universe/company_aliases.csv",
        "cache_dir": "data/news_cache",
        "persist_articles_json": True,
    },
    "experts": {
        "enabled": False,
        "list": ["trend", "momentum", "mean_reversion", "volatility", "value", "news_catalyst"],
        "min_coverage": 5,
        "output_dir": "reports/experts",
    },
    "meta_policy": {
        "enabled": False,
        "method": "perf_weighted",
        "lookback_days": 90,
        "decay": 0.97,
        "weight_floor": 0.05,
        "weight_cap": 0.60,
        "min_trades_for_trust": 20,
        "regime": {
            "index_ticker": "^NSEI",
            "vol_lookback": 20,
            "trend_fast": 20,
            "trend_slow": 50,
            "high_vol_threshold": 0.25,
        },
        "risk_guardrails": {
            "disable_on_drawdown": 0.15,
            "min_calibration": 0.55,
            "max_turnover": None,
        },
    },
    "filesystem": {"ignore_synthetic_folders": True},
    "universe": {
        "path": "data/universe/universe_nse.csv",
        "india": {
            "source": "csv",
            "tickers_file": "data/universe/universe_nse.csv",
            "symbol_column": "symbol",
            "suffix": ".NS",
            "filter_series_eq": False,
        },
        "uk": {"source": "lse", "tickers_file": "data/universe/lse_all.csv"},
        "selected_sets": ["india"],
        "dry_run": False,
    },
    "news_universe": {
        "enabled": False,
        "lookback_hours": 12,
        "max_candidates": 50,
        "min_liquidity_avg_volume": 200000,
        "min_source_rank": 0.0,
        "sentiment_model": "finbert",
        "require_price_confirmation": True,
        "confirmation": {
            "method": "gap_or_volume",
            "gap_abs_return_threshold": 0.015,
            "volume_z_threshold": 1.5,
            "lookback_days": 20,
        },
        "recency_decay_half_life_hours": 6,
        "source_weights_path": "data/news/source_weights.csv",
        "aliases_path": "data/universe/company_aliases.csv",
        "cache_dir": "data/news_cache",
        "persist_articles_json": True,
    },
    "experts": {
        "enabled": False,
        "list": ["trend", "momentum", "mean_reversion", "volatility", "value", "news_catalyst"],
        "min_coverage": 5,
        "output_dir": "reports/experts",
    },
    "meta_policy": {
        "enabled": False,
        "method": "perf_weighted",
        "lookback_days": 90,
        "decay": 0.97,
        "weight_floor": 0.05,
        "weight_cap": 0.60,
        "min_trades_for_trust": 20,
        "regime": {
            "index_ticker": "^NSEI",
            "vol_lookback": 20,
            "trend_fast": 20,
            "trend_slow": 50,
            "high_vol_threshold": 0.25,
        },
        "risk_guardrails": {
            "disable_on_drawdown": 0.15,
            "min_calibration": 0.55,
            "max_turnover": None,
        },
    },
    "filesystem": {
        "ignore_synthetic_folders": True,
    },
    "crypto": {
        "enabled": False,
        "environment": "production",
        "endpoint": "ws-feed",
        "symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "timeframe": "1m",
        "candle_intervals": ["1m", "5m"],
        "prices_dir": "data/prices/crypto",
        "universe_csv": "src/spectraquant/crypto/universe/crypto_universe.csv",
    },
    "news_ai": {
        "enabled": False,
        "lookback_hours": 24,
        "recency_half_life_hours": 6,
        "source_rank_min": 0.3,
        "output_dir": "reports/news",
    },
    "onchain_ai": {
        "enabled": False,
        "refresh_minutes": 15,
        "anomaly_threshold": 2.5,
        "output_dir": "reports/onchain",
    },
    "agents": {
        "enabled": False,
        "list": [
            "momentum",
            "mean_reversion",
            "volatility",
            "carry_funding",
            "news_catalyst",
            "onchain_flow",
        ],
        "output_dir": "reports/agents",
    },
    "crypto_meta_policy": {
        "enabled": False,
        "weighting": "regime_perf",
        "perf_lookback_days": 30,
        "decay": 0.95,
        "weight_floor": 0.05,
        "weight_cap": 0.50,
    },
    "crypto_portfolio": {
        "allocator": "vol_target",
        "target_vol": 0.15,
        "max_weight": 0.25,
        "max_turnover": 0.30,
        "stop_trading_drawdown": 0.10,
        "rebalance_frequency": "daily",
    },
}


def _suppress_ssl_warnings() -> None:
    try:
        from urllib3.exceptions import NotOpenSSLWarning
    except Exception:
        return
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def _warn_streamlit_pandas_compat() -> None:
    try:
        pandas_version = metadata.version("pandas")
        streamlit_version = metadata.version("streamlit")
    except metadata.PackageNotFoundError:
        return
    if pandas_version != "3.0.0":
        return
    try:
        streamlit_major = int(streamlit_version.split(".", 1)[0])
    except ValueError:
        return
    if streamlit_major < 2:
        logger.warning(
            "Detected pandas==%s with Streamlit==%s. Streamlit <2.0 may be incompatible with pandas 3.0.0.",
            pandas_version,
            streamlit_version,
        )


def validate_runtime_defaults(cfg: dict) -> None:
    """Validate default runtime expectations for markets and safety."""

    _warn_streamlit_pandas_compat()

    data_cfg = cfg.get("data") or {}
    tickers, meta = resolve_universe(cfg)
    if not tickers:
        raise ValueError(
            "Universe resolved from %s but became empty after cleaning. "
            "raw=%s cleaned=%s dropped_empty=%s dropped_placeholders=%s. "
            "Set universe.tickers or data.tickers or create data/universe/*.csv."
            % (
                meta.get("source"),
                meta.get("raw_count"),
                meta.get("cleaned_count"),
                meta.get("dropped_empty"),
                meta.get("dropped_placeholders"),
            )
        )

    if meta.get("invalid_suffix_count"):
        invalid = ", ".join(meta.get("dropped_invalid_suffix") or [])
        raise ValueError(
            "Invalid ticker suffixes detected from %s: %s (total=%s). "
            "Allowed suffixes: %s."
            % (
                meta.get("source"),
                invalid or "unknown",
                meta.get("invalid_suffix_count"),
                ", ".join(meta.get("allowed_suffixes") or []),
            )
        )

    cfg.setdefault("data", {})
    cfg["data"]["tickers"] = tickers
    universe_cfg = cfg.get("universe")
    if isinstance(universe_cfg, dict):
        universe_cfg["tickers"] = tickers
        cfg["universe"] = universe_cfg
    tickers_upper = [str(t).upper() for t in tickers]

    synthetic_flag = bool(data_cfg.get("synthetic", False))
    mlops_cfg = cfg.get("mlops") or {}
    if "auto_retrain" not in mlops_cfg or not isinstance(mlops_cfg.get("auto_retrain"), bool):
        raise ValueError("mlops.auto_retrain must be explicitly set to True or False")

    if "portfolio" not in cfg or "alpha" not in cfg:
        raise ValueError("Configuration must include portfolio and alpha sections")

    markets = sorted({"UK" if t.endswith(".L") else "India" for t in tickers_upper})
    logger.info("Active universe: %s", ", ".join(tickers_upper))
    logger.info("Synthetic mode enabled: %s", synthetic_flag)
    logger.info("Markets detected: %s", "/".join(markets))


def get_config() -> Dict[str, Any]:
    """Load configuration with defaults merged."""

    _suppress_ssl_warnings()

    cfg: Dict[str, Any] = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            cfg.update(loaded)
        logger.info("Loaded configuration from %s", CONFIG_PATH)
    else:
        logger.info("Config file %s not found; using defaults.", CONFIG_PATH)

    # Merge defaults where keys are missing
    for key, value in DEFAULT_CONFIG.items():
        if key not in cfg:
            cfg[key] = value
        elif isinstance(cfg[key], dict) and isinstance(value, dict):
            merged = value.copy()
            merged.update(cfg[key])
            cfg[key] = merged

    env_research = os.getenv("SPECTRAQUANT_RESEARCH_MODE")
    if env_research is not None:
        cfg["research_mode"] = env_research.strip().lower() in {"1", "true", "yes", "on"}

    env_sentiment = os.getenv("SPECTRAQUANT_USE_SENTIMENT")
    if env_sentiment is not None:
        cfg.setdefault("sentiment", {})
        cfg["sentiment"]["enabled"] = env_sentiment.strip().lower() in {"1", "true", "yes", "on"}

    env_universe = os.getenv("SPECTRAQUANT_UNIVERSE")
    if env_universe:
        mode, selections = parse_universe_override(env_universe, cfg)
        cfg.setdefault("universe", {})
        if mode == "tickers":
            cfg["universe"]["tickers"] = selections
            cfg["universe"]["selected_sets"] = []
        else:
            cfg["universe"]["selected_sets"] = selections

    env_dry_run = os.getenv("SPECTRAQUANT_DRY_RUN")
    if env_dry_run is not None:
        cfg.setdefault("universe", {})
        cfg["universe"]["dry_run"] = env_dry_run.strip().lower() in {"1", "true", "yes", "on"}

    env_test_mode = os.getenv("SPECTRAQUANT_TEST_MODE")
    if env_test_mode is not None:
        enabled = env_test_mode.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(cfg.get("test_mode"), dict):
            cfg["test_mode"]["enabled"] = enabled
        else:
            cfg["test_mode"] = enabled

    env_force_pass = os.getenv("SPECTRAQUANT_FORCE_PASS_TESTS")
    if env_force_pass is not None:
        cfg.setdefault("qa", {})
        cfg["qa"]["force_pass_tests"] = env_force_pass.strip().lower() in {"1", "true", "yes", "on"}

    # Backfill universe alias for callers expecting cfg["universe"]["tickers"].
    if "universe" not in cfg or not cfg.get("universe"):
        data_tickers = cfg.get("data", {}).get("tickers", [])
        cfg["universe"] = {"tickers": data_tickers}
    elif "tickers" not in cfg["universe"]:
        cfg["universe"]["tickers"] = cfg.get("data", {}).get("tickers", [])

    if "data" in cfg:
        if "tickers" not in cfg["data"]:
            cfg["data"]["tickers"] = cfg.get("universe", {}).get("tickers", [])

    validate_runtime_defaults(cfg)

    return cfg
