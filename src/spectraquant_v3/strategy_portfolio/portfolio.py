"""Strategy portfolio layer for SpectraQuant-AI-V3.

Combines multiple strategies into a single portfolio with configurable
weighting schemes and risk budgets.

Supported weighting schemes
---------------------------
``equal``
    Each strategy receives an equal weight of 1/N.

``risk_budget``
    Weights are proportional to the supplied *risk_budget* dict.
    Missing strategies default to equal share of remaining budget.

``custom``
    Weights come directly from the *weights* parameter.  They are
    normalised to sum to 1.0.

Usage::

    from spectraquant_v3.strategy_portfolio import StrategyPortfolio

    portfolio = StrategyPortfolio(
        portfolio_id="multi_alpha_v1",
        strategy_ids=["crypto_momentum_v1", "equity_momentum_v1"],
        weighting_scheme="equal",
        rebalance_frequency="W",
    )
    result = portfolio.run(cfg_by_strategy={
        "crypto_momentum_v1": crypto_cfg,
        "equity_momentum_v1": equity_cfg,
    })
    print(result.metrics)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from spectraquant_v3.pipeline._strategy_runner import run_strategy
from spectraquant_v3.strategy_portfolio.result import PortfolioResult

logger = logging.getLogger(__name__)

_SUPPORTED_SCHEMES = {"equal", "risk_budget", "custom"}


class StrategyPortfolio:
    """Orchestrate a multi-strategy research portfolio.

    Args:
        portfolio_id:       Unique identifier for this portfolio.
        strategy_ids:       Ordered list of strategy IDs to combine.
        weighting_scheme:   How to assign weights to strategies.
                            One of ``"equal"``, ``"risk_budget"``, or
                            ``"custom"``.
        risk_budget:        Mapping of strategy_id → risk budget fraction
                            (used when *weighting_scheme* is ``"risk_budget"``).
        custom_weights:     Mapping of strategy_id → weight
                            (used when *weighting_scheme* is ``"custom"``).
        rebalance_frequency: Pandas offset alias (e.g. ``"W"``, ``"M"``).
        max_strategy_weight: Maximum weight for any single strategy (0–1].
        output_dir:          Directory for portfolio result artefacts.
    """

    def __init__(
        self,
        portfolio_id: str,
        strategy_ids: list[str],
        weighting_scheme: str = "equal",
        risk_budget: dict[str, float] | None = None,
        custom_weights: dict[str, float] | None = None,
        rebalance_frequency: str = "M",
        max_strategy_weight: float = 1.0,
        output_dir: str | Path = "reports/strategy_portfolio",
    ) -> None:
        if weighting_scheme not in _SUPPORTED_SCHEMES:
            raise ValueError(
                f"weighting_scheme must be one of {_SUPPORTED_SCHEMES}, "
                f"got {weighting_scheme!r}"
            )
        if not strategy_ids:
            raise ValueError("strategy_ids must not be empty")

        self.portfolio_id = portfolio_id
        self.strategy_ids = list(strategy_ids)
        self.weighting_scheme = weighting_scheme
        self.risk_budget = risk_budget or {}
        self.custom_weights = custom_weights or {}
        self.rebalance_frequency = rebalance_frequency
        self.max_strategy_weight = max_strategy_weight
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def compute_weights(self) -> dict[str, float]:
        """Compute normalised portfolio weights for each strategy.

        Each raw weight is first computed by the selected scheme, then
        capped at ``max_strategy_weight``.  The capped weights are always
        renormalised so they sum to 1.0, redistributing the excess from
        capped strategies proportionally to the others.

        Returns:
            Dict mapping strategy_id → weight (values sum to 1.0).
        """
        n = len(self.strategy_ids)

        if self.weighting_scheme == "equal":
            raw = {sid: 1.0 / n for sid in self.strategy_ids}

        elif self.weighting_scheme == "risk_budget":
            total = sum(self.risk_budget.get(sid, 1.0) for sid in self.strategy_ids)
            raw = {
                sid: self.risk_budget.get(sid, 1.0) / total
                for sid in self.strategy_ids
            }

        else:  # custom
            total = sum(self.custom_weights.get(sid, 0.0) for sid in self.strategy_ids)
            if total <= 0:
                raw = {sid: 1.0 / n for sid in self.strategy_ids}
            else:
                raw = {
                    sid: self.custom_weights.get(sid, 0.0) / total
                    for sid in self.strategy_ids
                }

        # Enforce per-strategy cap then renormalise unconditionally.
        # This redistributes the "excess" from capped strategies proportionally.
        capped = {sid: min(w, self.max_strategy_weight) for sid, w in raw.items()}
        total_capped = sum(capped.values())
        if total_capped <= 0:
            return {sid: 1.0 / n for sid in self.strategy_ids}
        return {sid: w / total_capped for sid, w in capped.items()}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        cfg_by_strategy: dict[str, dict[str, Any]],
        price_data_by_strategy: dict[str, dict] | None = None,
        market_data_by_strategy: dict[str, dict] | None = None,
        dry_run: bool = False,
    ) -> PortfolioResult:
        """Run each strategy and aggregate portfolio-level metrics.

        Args:
            cfg_by_strategy:          Dict of strategy_id → pipeline config.
            price_data_by_strategy:   Optional dict of strategy_id → price_data
                                      (passed to the strategy runner when provided).
            market_data_by_strategy:  Optional dict of strategy_id → market_data.
            dry_run:                  When True, skip all writes and network calls.

        Returns:
            :class:`PortfolioResult` with per-strategy and aggregate metrics.
        """


        weights = self.compute_weights()
        strategy_results: dict[str, Any] = {}
        all_returns: list[float] = []
        all_sharpes: list[float] = []

        for sid in self.strategy_ids:
            cfg = cfg_by_strategy.get(sid, {})
            price_data = (price_data_by_strategy or {}).get(sid)
            market_data = (market_data_by_strategy or {}).get(sid)

            logger.info("StrategyPortfolio: running strategy %s", sid)
            try:
                result = run_strategy(
                    strategy_id=sid,
                    cfg=cfg,
                    dry_run=dry_run,
                    price_data=price_data,
                    market_data=market_data,
                )
                strategy_results[sid] = result
                metrics = result.get("metrics", {})
                w = weights.get(sid, 0.0)
                cagr = metrics.get("cagr", 0.0) or 0.0
                sharpe = metrics.get("sharpe", 0.0) or 0.0
                all_returns.append(cagr * w)
                all_sharpes.append(sharpe * w)
            except Exception as exc:
                logger.warning("StrategyPortfolio: strategy %s failed: %s", sid, exc)
                strategy_results[sid] = {"error": str(exc)}

        # Portfolio-level aggregate metrics
        portfolio_metrics: dict[str, Any] = {
            "weighted_cagr": sum(all_returns),
            "weighted_sharpe": sum(all_sharpes),
            "strategy_count": len(self.strategy_ids),
            "weighting_scheme": self.weighting_scheme,
            "rebalance_frequency": self.rebalance_frequency,
        }

        result_obj = PortfolioResult(
            portfolio_id=self.portfolio_id,
            strategy_ids=self.strategy_ids,
            weights=weights,
            metrics=portfolio_metrics,
            strategy_results=strategy_results,
        )

        if not dry_run:
            out_path = result_obj.write(self.output_dir)
            result_obj.artifact_paths.append(str(out_path))
            logger.info(
                "StrategyPortfolio: result written to %s", out_path
            )

        return result_obj
