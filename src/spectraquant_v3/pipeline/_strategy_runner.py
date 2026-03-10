"""Strategy-aware pipeline runner for SpectraQuant-AI-V3.

Provides :func:`run_strategy`, a convenience function that:

1. Loads and validates the strategy from the registry.
2. Merges the strategy's risk config into the pipeline config.
3. Dispatches to the correct asset-class pipeline
   (:func:`~spectraquant_v3.pipeline.crypto_pipeline.run_crypto_pipeline`
   or :func:`~spectraquant_v3.pipeline.equity_pipeline.run_equity_pipeline`).
4. Enforces asset-class segregation: running a crypto strategy on the
   equity pipeline (or vice versa) raises :class:`~spectraquant_v3.core.errors.MixedAssetClassRunError`.

Usage::

    from spectraquant_v3.pipeline import run_strategy
    from spectraquant_v3.core.config import get_crypto_config

    cfg = get_crypto_config()
    result = run_strategy("crypto_momentum_v1", cfg=cfg, dry_run=True)
    print(result["status"])  # "success"
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from spectraquant_v3.core.enums import RunMode


def run_strategy(
    strategy_id: str,
    cfg: dict[str, Any],
    run_mode: RunMode = RunMode.NORMAL,
    dry_run: bool = False,
    price_data: dict[str, pd.DataFrame] | None = None,
    market_data: dict[str, dict[str, Any]] | None = None,
    dataset: dict[str, pd.DataFrame] | None = None,
    run_id: str | None = None,
    project_root: str | None = None,
) -> dict[str, Any]:
    """Execute a registered strategy through the appropriate pipeline.

    Validates that the strategy's ``asset_class`` matches the pipeline being
    invoked.  If a strategy is registered as ``"crypto"`` but the supplied
    *cfg* belongs to an equity run (and vice versa), a
    :class:`~spectraquant_v3.core.errors.MixedAssetClassRunError` is raised.

    The strategy's :class:`~spectraquant_v3.strategies.strategy_definition.RiskConfig`
    is merged into *cfg* via
    :meth:`~spectraquant_v3.strategies.loader.StrategyLoader.build_pipeline_config`
    so that the existing pipeline code picks it up transparently.

    Args:
        strategy_id:  Registered strategy identifier.
        cfg:          Merged pipeline config dict.  Should be the output of
                      :func:`~spectraquant_v3.core.config.get_crypto_config` or
                      :func:`~spectraquant_v3.core.config.get_equity_config`.
        run_mode:     Cache behaviour (NORMAL / TEST / REFRESH).
        dry_run:      When True, skip all writes and network calls.
        price_data:   Pre-loaded OHLCV DataFrames keyed by canonical symbol.
        market_data:  Optional market metrics per symbol for universe gating.
        dataset:      Optional pre-computed per-symbol dataset containing
                      technical and/or news features for the selected strategy.
        run_id:       Override the auto-generated run ID.
        project_root: Override repo-root discovery (useful in tests).

    Returns:
        Pipeline result dict with keys: ``run_id``, ``status``, ``universe``,
        ``signals``, ``decisions``, ``allocations``, ``artefact_paths``,
        ``strategy_id``.

    Raises:
        KeyError:               If *strategy_id* is not registered.
        ValueError:             If the strategy is disabled or misconfigured.
        MixedAssetClassRunError: If the strategy's asset class does not match
                                 the config's asset class.
    """
    # Lazy imports to avoid circular dependency:
    # strategies.loader → strategies.allocators.registry → pipeline.allocator
    #                                                     → pipeline.__init__
    #                                                     → _strategy_runner (this module)
    from spectraquant_v3.core.errors import MixedAssetClassRunError  # noqa: PLC0415
    from spectraquant_v3.strategies.loader import StrategyLoader  # noqa: PLC0415

    # Load and validate the strategy
    defn = StrategyLoader.load(strategy_id)

    # Detect config asset class from the presence of the "crypto" / "equities" key
    if "crypto" in cfg and "equities" not in cfg:
        cfg_asset_class = "crypto"
    elif "equities" in cfg and "crypto" not in cfg:
        cfg_asset_class = "equity"
    else:
        # Ambiguous – fall back to the strategy definition
        cfg_asset_class = defn.asset_class

    if defn.asset_class != cfg_asset_class:
        raise MixedAssetClassRunError(
            f"Strategy '{strategy_id}' is a {defn.asset_class!r} strategy but "
            f"the supplied config belongs to a {cfg_asset_class!r} pipeline. "
            "Do not mix asset classes in a single invocation."
        )

    # Merge strategy risk config into the pipeline config
    merged_cfg = StrategyLoader.build_pipeline_config(strategy_id, cfg)

    # Dispatch to the correct pipeline
    if defn.asset_class == "crypto":
        from spectraquant_v3.pipeline.crypto_pipeline import run_crypto_pipeline

        result = run_crypto_pipeline(
            cfg=merged_cfg,
            run_mode=run_mode,
            dry_run=dry_run,
            price_data=price_data,
            market_data=market_data,
            dataset=dataset,
            run_id=run_id,
            project_root=project_root,
        )
    else:
        from spectraquant_v3.pipeline.equity_pipeline import run_equity_pipeline

        result = run_equity_pipeline(
            cfg=merged_cfg,
            run_mode=run_mode,
            dry_run=dry_run,
            price_data=price_data,
            market_data=market_data,
            dataset=dataset,
            run_id=run_id,
            project_root=project_root,
        )

    result["strategy_id"] = strategy_id
    return result
