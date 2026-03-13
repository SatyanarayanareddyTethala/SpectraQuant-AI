"""Research CLI commands for SpectraQuant-AI-V3.

Available commands (registered as sub-commands of the ``research`` group):
  dataset   Build a labeled research dataset from cached OHLCV + feature data.
"""

from __future__ import annotations

import typer

research_app = typer.Typer(
    name="research",
    help="Research dataset building commands. Note: 'research run' is not implemented; use 'research dataset'.",
    no_args_is_help=True,
)


@research_app.command("dataset")
def research_dataset(
    asset_class: str = typer.Option(
        "crypto",
        "--asset-class",
        "-a",
        help="Asset class: crypto | equity",
    ),
    output_dir: str = typer.Option(
        "data/research",
        "--output-dir",
        "-o",
        help="Directory to write parquet + manifest files",
    ),
    forward_windows: str = typer.Option(
        "1,5",
        "--forward-windows",
        help="Comma-separated forward-return horizons in days, e.g. '1,5,20'",
    ),
    train_frac: float = typer.Option(
        0.70, "--train-frac", help="Fraction of dates for training partition"
    ),
    val_frac: float = typer.Option(
        0.15, "--val-frac", help="Fraction of dates for validation partition"
    ),
    symbols: str = typer.Option(
        "", "--symbols", help="Comma-separated symbols (default: from config)"
    ),
    run_id: str = typer.Option("research", "--run-id", help="Run identifier"),
    config_dir: str = typer.Option(
        "", "--config-dir", help="Override config/v3/ directory"
    ),
    include_news: bool = typer.Option(False, "--include-news", help="Include crypto news features when available"),
    include_context: bool = typer.Option(False, "--include-context", help="Include cross-sectional market-context features when available"),
    news_dir: str = typer.Option("", "--news-dir", help="Directory containing per-symbol news parquet files"),
    context_dir: str = typer.Option("", "--context-dir", help="Directory containing per-symbol context parquet files"),
) -> None:
    """Build a labeled research dataset from cached OHLCV + feature data.

    Reads OHLCV data from the local cache, computes features, adds
    forward-return labels, splits by date, and writes train/val/test
    parquet files to ``--output-dir``.
    """
    from pathlib import Path

    from spectraquant_v3.core.cache import CacheManager
    from spectraquant_v3.core.config import get_crypto_config, get_equity_config
    from spectraquant_v3.core.errors import EmptyPriceDataError
    from spectraquant_v3.research.dataset_builder import ResearchDatasetBuilder

    # Validate asset class
    if asset_class not in ("crypto", "equity"):
        typer.echo(
            f"[research dataset] ERROR: --asset-class must be 'crypto' or 'equity', "
            f"got {asset_class!r}.",
            err=True,
        )
        raise typer.Exit(1)

    # Parse forward windows
    try:
        fwd_windows = [int(w.strip()) for w in forward_windows.split(",") if w.strip()]
    except ValueError:
        typer.echo(
            f"[research dataset] ERROR: invalid --forward-windows={forward_windows!r}. "
            "Use comma-separated integers, e.g. '1,5,20'.",
            err=True,
        )
        raise typer.Exit(1)

    # Load config
    try:
        if asset_class == "crypto":
            cfg = get_crypto_config(config_dir or None)
        else:
            cfg = get_equity_config(config_dir or None)
    except FileNotFoundError as exc:
        typer.echo(f"[research dataset] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    # Load feature data from cache
    if symbols:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif asset_class == "crypto":
        sym_list = [s.upper() for s in cfg.get("crypto", {}).get("symbols", [])]
    else:
        sym_list = cfg.get("equities", {}).get("universe", {}).get("tickers", [])

    if not sym_list:
        typer.echo(
            "[research dataset] No symbols found. Pass --symbols or configure in YAML.",
            err=True,
        )
        raise typer.Exit(1)

    # Read cached OHLCV
    if asset_class == "crypto":
        cache_dir = cfg.get("crypto", {}).get("prices_dir", "data/cache/crypto/prices")
    else:
        cache_dir = cfg.get("equities", {}).get("prices_dir", "data/cache/equities/prices")

    from spectraquant_v3.core.enums import RunMode  # noqa: PLC0415

    cache = CacheManager(cache_dir, run_mode=RunMode.TEST)
    price_data: dict = {}
    for sym in sym_list:
        try:
            df = cache.read_parquet(sym)
            if df is not None and not df.empty:
                price_data[sym] = df
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[research dataset] WARNING: could not load cache for '{sym}': {exc}")

    if not price_data:
        typer.echo(
            f"[research dataset] No cached OHLCV data found in '{cache_dir}'. "
            "Run the download command first.",
            err=True,
        )
        raise typer.Exit(1)

    # Compute features
    typer.echo(
        f"[research dataset] Computing features for {len(price_data)} symbol(s) …"
    )
    if asset_class == "crypto":
        from spectraquant_v3.crypto.features.engine import CryptoFeatureEngine  # noqa: PLC0415

        engine = CryptoFeatureEngine.from_config(cfg)
    else:
        from spectraquant_v3.equities.features.engine import EquityFeatureEngine  # noqa: PLC0415

        engine = EquityFeatureEngine.from_config(cfg)

    news_map: dict = {}
    context_map: dict = {}
    if asset_class == "crypto" and include_news:
        resolved_news_dir = Path(
            news_dir
            or cfg.get("crypto", {}).get("news", {}).get("store_dir", "data/cache/crypto/news_store")
        )
        for sym in price_data:
            path = resolved_news_dir / f"{sym}.parquet"
            if path.exists():
                try:
                    import pandas as pd  # noqa: PLC0415

                    news_map[sym] = pd.read_parquet(path)
                except Exception as exc:  # noqa: BLE001
                    typer.echo(f"[research dataset] WARNING: could not load news for '{sym}': {exc}")

    if asset_class == "crypto" and include_context:
        resolved_context_dir = Path(
            context_dir
            or cfg.get("crypto", {}).get("context", {}).get("store_dir", "data/cache/crypto/context")
        )
        for sym in price_data:
            path = resolved_context_dir / f"{sym}.parquet"
            if path.exists():
                try:
                    import pandas as pd  # noqa: PLC0415

                    context_map[sym] = pd.read_parquet(path)
                except Exception as exc:  # noqa: BLE001
                    typer.echo(f"[research dataset] WARNING: could not load context for '{sym}': {exc}")

    feature_map = engine.transform_many(
        price_data,
        news_map=(news_map if asset_class == "crypto" else None),
        context_map=(context_map if asset_class == "crypto" else None),
    )

    if not feature_map:
        typer.echo(
            "[research dataset] Feature computation produced no output. "
            "Check that cached data meets the minimum row requirements.",
            err=True,
        )
        raise typer.Exit(1)

    # Build dataset
    typer.echo(
        f"[research dataset] Building dataset "
        f"(forward_windows={fwd_windows} train={train_frac:.0%} val={val_frac:.0%}) …"
    )
    try:
        builder = ResearchDatasetBuilder(output_dir=output_dir, run_id=run_id)
        result = builder.build(
            feature_map=feature_map,
            forward_windows=fwd_windows,
            train_frac=train_frac,
            val_frac=val_frac,
        )
    except (EmptyPriceDataError, ValueError) as exc:
        typer.echo(f"[research dataset] ERROR: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(result.summary())
    typer.echo(f"[research dataset] Manifest: {result.manifest_path}")
    raise typer.Exit(0)
