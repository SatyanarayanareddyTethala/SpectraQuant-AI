"""End-to-end crypto pipeline orchestration.

Ingests market data, builds features, runs agents, blends signals,
allocates portfolio, and writes reports.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from spectraquant.core.enums import AssetClass
from spectraquant.core.errors import EmptyOHLCVError

logger = logging.getLogger(__name__)


def _download_prices_yfinance(
    symbols: list[str],
    prices_dir: Path,
    period: str = "5d",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download OHLCV prices for *symbols* via yfinance and cache to parquet.

    Parameters
    ----------
    symbols:
        Canonical or yfinance symbols (e.g. ``["BTC-USD", "ETH-USD"]``).
    prices_dir:
        Directory where parquet files are written.
    period:
        yfinance period string (``"5d"``, ``"1mo"``, etc.)
    interval:
        yfinance interval string (``"1d"``, ``"1h"``, etc.)

    Returns
    -------
    dict
        ``{symbol: DataFrame}`` for successfully downloaded symbols.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available; skipping price download")
        return {}

    prices_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        # Ensure symbol has -USD suffix for yfinance
        yf_sym = sym if "-" in sym else f"{sym}-USD"
        try:
            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period=period, interval=interval, auto_adjust=True)
            if hist.empty:
                logger.warning("yfinance returned empty data for %s", yf_sym)
                continue
            hist.columns = [str(c).lower() for c in hist.columns]
            # Cache to parquet
            safe_name = yf_sym.replace("/", "_")
            parquet_path = prices_dir / f"{safe_name}.parquet"
            hist.to_parquet(parquet_path, engine="pyarrow")
            downloaded[sym] = hist
            logger.info("Downloaded and cached prices for %s → %s", yf_sym, parquet_path)
        except Exception:
            logger.warning("Failed to download prices for %s", yf_sym, exc_info=True)

    return downloaded


def _build_inputs_matrix(
    symbols: list[str],
    price_data: dict[str, pd.DataFrame],
    news_features: pd.DataFrame,
    onchain_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-asset inputs availability matrix.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, has_ohlcv, has_volume, has_onchain, has_funding,
        has_oi.
    """
    rows = []
    for sym in symbols:
        df = price_data.get(sym)
        cols = set(df.columns) if df is not None else set()
        ohlcv_cols = {"open", "high", "low", "close"} | {"Open", "High", "Low", "Close"}
        vol_cols = {"volume", "Volume"}

        has_ohlcv = bool(df is not None and not df.empty and ohlcv_cols & cols)
        has_volume = bool(df is not None and not df.empty and vol_cols & cols)

        sym_oc = (
            onchain_features[onchain_features["symbol"] == sym]
            if (not onchain_features.empty and "symbol" in onchain_features.columns)
            else pd.DataFrame()
        )
        has_onchain = not sym_oc.empty

        has_funding = bool(df is not None and "funding_rate" in cols)
        has_oi = bool(df is not None and "open_interest" in cols)

        rows.append({
            "symbol": sym,
            "has_ohlcv": has_ohlcv,
            "has_volume": has_volume,
            "has_onchain": has_onchain,
            "has_funding": has_funding,
            "has_oi": has_oi,
        })

    return pd.DataFrame(rows)


def run_crypto_pipeline(
    cfg: dict[str, Any] | None = None,
    dry_run: bool = False,
    refresh_prices: bool = False,
) -> dict[str, Any]:
    """Execute the full crypto pipeline.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dict.  Loaded from ``config.yaml`` when *None*.
    dry_run : bool
        When *True*, skip execution / order placement.
    refresh_prices : bool
        When *True*, force-download prices even if cache exists.

    Returns
    -------
    dict
        Summary with keys: weights, signals, news_features, onchain_features.

    Raises
    ------
    EmptyOHLCVError
        If zero symbols have OHLCV data after the inputs availability matrix is
        built (non-test-mode only).  Test mode is exempt so that CI runs with an
        empty price cache do not abort unexpectedly.
    """
    if cfg is None:
        from spectraquant.config import get_config
        cfg = get_config()

    crypto_cfg = cfg.get("crypto", {})
    dataset_cfg = cfg.get("crypto_dataset", {})
    now_utc = datetime.now(timezone.utc)
    ts_label = now_utc.strftime("%Y%m%d_%H%M%S")
    result: dict[str, Any] = {"timestamp": now_utc}

    test_mode_val = cfg.get("test_mode", {})
    if isinstance(test_mode_val, dict):
        test_mode = test_mode_val.get("enabled", False)
    else:
        test_mode = bool(test_mode_val)

    # --- 1. Universe -------------------------------------------------------
    logger.info("Step 1: Building crypto universe")

    universe_mode = crypto_cfg.get("universe_mode", "news_first" if crypto_cfg.get("news_first", False) else "static")
    data_dir = dataset_cfg.get("data_dir", "data/crypto")

    symbols: list[str] = []

    if universe_mode == "dataset_topN":
        from spectraquant.crypto.dataset.ingest import load_market_snapshot
        from spectraquant.crypto.universe.quality_gate import (
            build_dataset_topN_universe,
            apply_quality_gate,
            write_universe_report,
        )
        snapshot = load_market_snapshot(data_dir)
        gate_cfg = dataset_cfg.get("quality_gate", {})
        gate_kwargs = {k: v for k, v in gate_cfg.items() if v is not None}
        if not snapshot.empty:
            top_n = crypto_cfg.get("universe_top_n", 20)
            symbols = build_dataset_topN_universe(snapshot, top_n=top_n, gate_kwargs=gate_kwargs)
            # Write full universe report
            full_gated = apply_quality_gate(snapshot, **gate_kwargs)
            write_universe_report(full_gated)
        else:
            logger.warning("dataset_topN mode: no snapshot found; falling back to config symbols")
            symbols = [s.replace("-USD", "") for s in crypto_cfg.get("symbols", ["BTC-USD", "ETH-USD", "SOL-USD"])]

    elif universe_mode == "hybrid_news_dataset":
        from spectraquant.crypto.universe import build_news_crypto_universe
        from spectraquant.crypto.dataset.ingest import load_market_snapshot
        from spectraquant.crypto.universe.quality_gate import build_hybrid_universe, write_universe_report, apply_quality_gate

        news_symbols = build_news_crypto_universe(cfg)
        snapshot = load_market_snapshot(data_dir)
        gate_cfg = dataset_cfg.get("quality_gate", {})
        gate_kwargs = {k: v for k, v in gate_cfg.items() if v is not None}
        top_n = crypto_cfg.get("universe_top_n", 20)
        symbols = build_hybrid_universe(news_symbols, snapshot, top_n=top_n, gate_kwargs=gate_kwargs)
        if not snapshot.empty:
            full_gated = apply_quality_gate(snapshot, **gate_kwargs)
            write_universe_report(full_gated)

    elif universe_mode == "news_first" or crypto_cfg.get("news_first", False):
        from spectraquant.crypto.universe import build_news_crypto_universe

        news_symbols = build_news_crypto_universe(cfg)
        logger.info("News-first crypto universe selected: %s", news_symbols)
        # Strip -USD suffix for canonical form
        symbols = [s.replace("-USD", "") for s in news_symbols]

    else:
        from spectraquant.crypto.universe import build_crypto_universe

        csv_path = crypto_cfg.get(
            "universe_csv",
            "src/spectraquant/crypto/universe/crypto_universe.csv",
        )
        assets = build_crypto_universe(csv_path=csv_path)
        symbols = [a.symbol for a in assets]

    # Resolve yfinance symbols
    try:
        from spectraquant.crypto.dataset.ingest import get_yfinance_symbol
        yf_symbols = [get_yfinance_symbol(s, data_dir) for s in symbols]
    except Exception:
        yf_symbols = [f"{s}-USD" for s in symbols]

    logger.info("Universe: %d assets → %s", len(symbols), symbols)

    # --- 2. Load / download prices -----------------------------------------
    prices_dir = Path(crypto_cfg.get("prices_dir", "data/prices/crypto"))

    test_mode_env = os.getenv("SPECTRAQUANT_TEST_MODE", "").strip().lower()
    _is_test_mode = test_mode or test_mode_env in {"1", "true", "yes", "on"}

    returns_dict: dict[str, pd.Series] = {}
    price_data: dict[str, pd.DataFrame] = {}

    def _try_load_from_cache(sym: str, yf_sym: str) -> pd.DataFrame | None:
        """Try to load price data from cache files."""
        candidates = [
            prices_dir / f"{yf_sym}.parquet",
            prices_dir / f"{yf_sym.replace('/', '_')}.parquet",
            prices_dir / f"{sym}.parquet",
            prices_dir / f"{yf_sym}_1m.parquet",
            prices_dir / f"{yf_sym}_5m.parquet",
        ]
        for parquet in candidates:
            if parquet.exists():
                df = pd.read_parquet(parquet)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                df.columns = [str(c).lower() for c in df.columns]
                return df
        return None

    logger.info("Step 2: Loading price data from cache (prices_dir=%s)", prices_dir)
    for sym, yf_sym in zip(symbols, yf_symbols):
        df = _try_load_from_cache(sym, yf_sym)
        if df is not None:
            price_data[sym] = df
            if "close" in df.columns and len(df) > 1:
                returns_dict[sym] = df["close"].pct_change().dropna()

    missing_syms = [s for s in symbols if s not in price_data]
    logger.info(
        "Loaded prices for %d/%d symbols (%d missing)",
        len(price_data),
        len(symbols),
        len(missing_syms),
    )

    # Step 2a: Download missing prices (unless test-mode or dry-run)
    if missing_syms:
        if _is_test_mode:
            missing_paths = [
                str(prices_dir / f"{yf_symbols[symbols.index(s)]}.parquet")
                for s in missing_syms
            ]
            logger.error(
                "test-mode: prices missing for %s. "
                "Expected cache paths: %s. "
                "Run without --test-mode to download, or pre-populate the cache.",
                missing_syms,
                missing_paths,
            )
        elif dry_run:
            logger.info(
                "dry-run: skipping price download for missing symbols: %s",
                missing_syms,
            )
        elif refresh_prices or not price_data:
            logger.info("Step 2a: Downloading prices for missing symbols: %s", missing_syms)
            missing_yf = [yf_symbols[symbols.index(s)] for s in missing_syms]
            downloaded = _download_prices_yfinance(missing_yf, prices_dir)
            for sym, df in downloaded.items():
                # sym here is the yf_sym; map back to canonical
                canonical = sym.replace("-USD", "")
                price_data[canonical] = df
                if "close" in df.columns and len(df) > 1:
                    returns_dict[canonical] = df["close"].pct_change().dropna()
            still_missing = [s for s in missing_syms if s not in price_data]
            if still_missing:
                logger.warning("Prices still missing after download: %s", still_missing)
            else:
                logger.info("Successfully downloaded prices for all missing symbols")

    returns_df = pd.DataFrame(returns_dict)
    logger.info("Final price data: %d symbols loaded", len(price_data))

    # --- 3. Features -------------------------------------------------------
    logger.info("Step 3: Computing features")
    from spectraquant.crypto.features import (
        compute_microstructure_features,
        compute_derivatives_features,
    )

    features_frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = price_data.get(sym)
        if df is None:
            continue
        feats = compute_microstructure_features(df)
        deriv = compute_derivatives_features(df)
        combined = pd.concat([feats, deriv], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined["symbol"] = sym
        features_frames.append(combined)

    if features_frames:
        all_features = pd.concat(features_frames, ignore_index=False)
    else:
        all_features = pd.DataFrame()

    # --- 4. News features --------------------------------------------------
    news_cfg = cfg.get("news_ai", {})
    news_features = pd.DataFrame()

    if news_cfg.get("enabled", False):
        logger.info("Step 4: Collecting news")
        try:
            from spectraquant.news.collector import collect_rss
            from spectraquant.news.dedupe import dedupe_articles
            from spectraquant.news.entity_map import map_articles_to_symbols
            from spectraquant.news.impact_scoring import build_news_features

            articles = collect_rss()
            articles = dedupe_articles(articles)
            articles = map_articles_to_symbols(
                articles,
                known_symbols=set(symbols),
            )
            news_features = build_news_features(
                articles,
                now_utc=now_utc,
                half_life_hours=news_cfg.get("recency_half_life_hours", 6.0),
            )

            out_dir = Path(news_cfg.get("output_dir", "reports/news"))
            out_dir.mkdir(parents=True, exist_ok=True)
            news_path = out_dir / f"news_features_{ts_label}.parquet"
            if not news_features.empty:
                news_features.to_parquet(news_path, engine="pyarrow")
                logger.info("Wrote %s", news_path)
        except Exception:
            logger.exception("News feature collection failed")
    else:
        logger.info("Step 4: News AI disabled — skipping")

    result["news_features"] = news_features

    # --- 5. On-chain features ----------------------------------------------
    onchain_cfg = cfg.get("onchain_ai", {})
    onchain_features = pd.DataFrame()

    if onchain_cfg.get("enabled", False):
        logger.info("Step 5: Collecting on-chain data")
        try:
            from spectraquant.onchain.collectors.free_sources import collect_all
            from spectraquant.onchain.features import compute_onchain_features
            from spectraquant.onchain.anomaly import compute_anomaly_scores

            raw = collect_all(symbols)
            if not raw.empty:
                onchain_features = compute_onchain_features(raw)
                onchain_features = compute_anomaly_scores(
                    onchain_features,
                    threshold=onchain_cfg.get("anomaly_threshold", 2.5),
                )

            out_dir = Path(onchain_cfg.get("output_dir", "reports/onchain"))
            out_dir.mkdir(parents=True, exist_ok=True)
            oc_path = out_dir / f"onchain_features_{ts_label}.parquet"
            if not onchain_features.empty:
                onchain_features.to_parquet(oc_path, engine="pyarrow")
                logger.info("Wrote %s", oc_path)
        except Exception:
            logger.exception("On-chain feature collection failed")
    else:
        logger.info("Step 5: On-chain AI disabled — skipping")

    result["onchain_features"] = onchain_features

    # --- 5b. Inputs availability matrix -----------------------------------
    logger.info("Step 5b: Building inputs availability matrix")
    inputs_matrix = _build_inputs_matrix(symbols, price_data, news_features, onchain_features)
    qa_dir = Path("reports/qa")
    qa_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = qa_dir / f"crypto_inputs_{ts_label}.csv"
    inputs_matrix.to_csv(inputs_path, index=False)

    # Log concise summary
    n_ohlcv = int(inputs_matrix["has_ohlcv"].sum()) if not inputs_matrix.empty else 0
    n_onchain = int(inputs_matrix["has_onchain"].sum()) if not inputs_matrix.empty else 0
    logger.info(
        "Inputs matrix: %d symbols, %d have OHLCV, %d have on-chain data. Report: %s",
        len(inputs_matrix),
        n_ohlcv,
        n_onchain,
        inputs_path,
    )

    # Hard guard: abort loudly when zero symbols have OHLCV data.
    # Test mode is exempt because CI caches may be empty by design.
    if n_ohlcv == 0 and symbols and not _is_test_mode:
        raise EmptyOHLCVError(
            asset_class=AssetClass.CRYPTO,
            symbols=symbols,
            run_mode="normal",
        )

    result["inputs_matrix"] = inputs_matrix

    # --- 6. Agents ---------------------------------------------------------
    agents_cfg = cfg.get("agents", {})
    agent_signals: dict[str, list] = {}

    if agents_cfg.get("enabled", False):
        logger.info("Step 6: Running agents")
        try:
            from spectraquant.agents.registry import AgentRegistry

            registry = AgentRegistry()
            for sym in symbols:
                df = price_data.get(sym)
                if df is None:
                    continue
                market = df.copy()
                # Join news features
                if not news_features.empty and "symbol" in news_features.columns:
                    nf = news_features[news_features["symbol"] == sym]
                    if not nf.empty:
                        for col in ["news_impact_mean", "news_sentiment_mean", "news_article_count"]:
                            if col in nf.columns:
                                market[col] = nf[col].values[0] if len(nf) > 0 else 0.0
                # Join onchain features
                if not onchain_features.empty and "symbol" in onchain_features.columns:
                    oc = onchain_features[onchain_features["symbol"] == sym]
                    if not oc.empty:
                        for col in ["anomaly_score"]:
                            if col in oc.columns:
                                market[col] = oc[col].values[0] if len(oc) > 0 else 0.0

                sigs = registry.run_all(market, symbol_override=sym)
                for agent_name, sig_list in sigs.items():
                    agent_signals.setdefault(agent_name, []).extend(sig_list)

            out_dir = Path(agents_cfg.get("output_dir", "reports/agents"))
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_agent_csv(agent_signals, out_dir / f"agent_decisions_{ts_label}.csv")
        except Exception:
            logger.exception("Agent execution failed")
    else:
        logger.info("Step 6: Agents disabled — skipping")

    result["agent_signals"] = agent_signals

    # --- 7. Arbiter --------------------------------------------------------
    blended_scores = pd.Series(dtype=float)

    meta_cfg = cfg.get("crypto_meta_policy", {})
    if meta_cfg.get("enabled", False) and agent_signals:
        logger.info("Step 7: Running arbiter")
        try:
            from spectraquant.agents.arbiter import Arbiter
            from spectraquant.agents.regime import detect_regime, CryptoRegime

            first_sym = symbols[0] if symbols else None
            regime = CryptoRegime.RANGE
            if first_sym and first_sym in price_data:
                regime = detect_regime(price_data[first_sym])

            arbiter = Arbiter()
            blended_df = arbiter.blend(agent_signals, regime)
            if not blended_df.empty:
                blended_scores = blended_df.set_index("symbol")["blended_score"]

            logger.info("Regime=%s, blended %d symbols", regime.value, len(blended_scores))
        except Exception:
            logger.exception("Arbiter failed")
    else:
        logger.info("Step 7: Meta-policy disabled — skipping")

    # --- 8. Portfolio allocation -------------------------------------------
    port_cfg = cfg.get("crypto_portfolio", {})
    weights = pd.Series(dtype=float)

    if blended_scores.abs().sum() > 1e-12 and not returns_df.empty:
        logger.info("Step 8: Allocating portfolio")
        try:
            from spectraquant.portfolio.allocator import allocate
            from spectraquant.portfolio.constraints import (
                PortfolioConstraints,
                apply_constraints,
            )

            weights = allocate(
                scores=blended_scores,
                returns=returns_df,
                method=port_cfg.get("allocator", "vol_target"),
                target_vol=port_cfg.get("target_vol", 0.15),
            )

            constraints = PortfolioConstraints(
                max_weight=port_cfg.get("max_weight", 0.25),
                max_turnover=port_cfg.get("max_turnover"),
            )
            weights = apply_constraints(weights, constraints)

            out_dir = Path("reports/portfolio")
            out_dir.mkdir(parents=True, exist_ok=True)
            w_path = out_dir / f"weights_{ts_label}.csv"
            weights.to_csv(w_path, header=True)
            logger.info("Wrote %s", w_path)
        except Exception:
            logger.exception("Portfolio allocation failed")
    else:
        logger.info("Step 8: No valid signals — skipping allocation")

    result["weights"] = weights

    # --- 9. Run artifact ------------------------------------------------------
    logger.info("Step 9: Writing run artifact")

    run_dir = Path("reports/run")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_artifact = {
        "timestamp": now_utc.isoformat(),
        "dry_run": dry_run,
        "universe_mode": universe_mode,
        "universe_size": len(symbols),
        "symbols": symbols,
        "prices_loaded": len(price_data),
        "features_rows": len(all_features),
        "news_features_rows": len(news_features),
        "onchain_features_rows": len(onchain_features),
        "agent_signal_groups": len(agent_signals),
        "total_agent_signals": sum(len(v) for v in agent_signals.values()),
        "blended_scores_count": len(blended_scores),
        "weights_count": len(weights),
    }
    artifact_path = run_dir / f"crypto_run_{ts_label}.json"
    artifact_path.write_text(json.dumps(run_artifact, indent=2, default=str))
    logger.info("Wrote run artifact to %s", artifact_path)
    result["artifact_path"] = str(artifact_path)

    if dry_run:
        logger.info("Dry run — skipping execution")
    else:
        logger.info("Pipeline complete (execution not yet wired)")

    return result


def _write_agent_csv(
    signals: dict[str, list],
    path: Path,
) -> None:
    """Flatten agent signals into a CSV."""
    rows: list[dict[str, Any]] = []
    for agent_name, sig_list in signals.items():
        for sig in sig_list:
            rows.append({
                "agent": agent_name,
                "symbol": sig.symbol,
                "score": sig.score,
                "confidence": sig.confidence,
                "horizon": sig.horizon,
                "rationale_tags": ",".join(sig.rationale_tags),
                "asof_utc": sig.asof_utc.isoformat(),
            })
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("Wrote %d agent decisions to %s", len(rows), path)
