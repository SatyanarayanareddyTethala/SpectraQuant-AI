"""Hybrid strategy parameterisation for SpectraQuant-AI-V3.

:class:`HybridStrategyParams` captures all tunable blend parameters for an
equity (or crypto) hybrid strategy variant.  It is the single source of truth
for an experiment's variant identity:

- It serialises cleanly to/from a plain dict so it can be stored alongside
  the experiment record and compared across runs.
- :meth:`inject_into_cfg` returns a *new* config dict with the parameters
  applied, leaving the original unchanged.  This is safe to pass directly
  to :class:`~spectraquant_v3.backtest.engine.BacktestEngine`.
- :meth:`run_id` derives a short deterministic identifier from the parameter
  values so that identical parameter sets always get the same run ID.

Supported hybrid strategy IDs
------------------------------
- ``equity_momentum_news_hybrid_v1``  (equity; uses all four params)
- ``crypto_momentum_news_hybrid_v1``  (crypto;  uses momentum_weight,
  news_weight, and min_confidence; vol_gate_threshold is ignored)

Usage::

    from spectraquant_v3.experiments.hybrid_params import HybridStrategyParams

    params = HybridStrategyParams(
        strategy_id="equity_momentum_news_hybrid_v1",
        momentum_weight=0.6,
        news_weight=0.4,
        vol_gate_threshold=0.30,
        min_confidence=0.10,
        min_signal_threshold=0.05,
    )

    cfg = params.inject_into_cfg(base_cfg)
    # cfg["strategies"]["equity_momentum_news_hybrid_v1"]["signal_blend"]
    #   → {"momentum_weight": 0.6, "news_weight": 0.4}
    # cfg["strategies"]["equity_momentum_news_hybrid_v1"]["vol_gate"]
    #   → {"threshold": 0.30}

    run_id = params.run_id()   # e.g. "hybrid_emnh_mw0.60_nw0.40_vg0.30_mc0.10"
    d      = params.to_dict()  # round-trip via HybridStrategyParams.from_dict(d)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Equity hybrid strategy ID (used to key the ``strategies`` config block)
EQUITY_HYBRID_ID = "equity_momentum_news_hybrid_v1"

#: Crypto hybrid strategy ID
CRYPTO_HYBRID_ID = "crypto_momentum_news_hybrid_v1"

#: Recognised hybrid strategy IDs
HYBRID_STRATEGY_IDS: frozenset[str] = frozenset({EQUITY_HYBRID_ID, CRYPTO_HYBRID_ID})

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class HybridStrategyParams:
    """Parameterisation for a hybrid momentum + news strategy variant.

    These parameters uniquely identify one configuration of a hybrid strategy.
    They are intended to be varied across experiment runs for systematic
    comparison.

    Args:
        strategy_id:          Registered hybrid strategy to parameterise.
                              Must be one of :data:`HYBRID_STRATEGY_IDS`.
        momentum_weight:      Fraction of the blended score contributed by
                              the momentum signal (``0 < momentum_weight <= 1``).
        news_weight:          Fraction of the blended score contributed by
                              the news-sentiment signal.  Together with
                              ``momentum_weight`` these need not sum to 1;
                              the formula is additive, not a normalised mix.
        vol_gate_threshold:   Annualised realised volatility above which signal
                              dampening activates.  Only used by the equity
                              hybrid; ignored for crypto.
        min_confidence:       Minimum signal confidence required to pass the
                              meta-policy filter.
        min_signal_threshold: Minimum absolute score to treat as actionable.
    """

    strategy_id: str = EQUITY_HYBRID_ID
    momentum_weight: float = 0.7
    news_weight: float = 0.3
    vol_gate_threshold: float = 0.25
    min_confidence: float = 0.10
    min_signal_threshold: float = 0.05
    # arbitrary extra metadata (e.g. description, tags) stored for reference only
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.strategy_id not in HYBRID_STRATEGY_IDS:
            raise ValueError(
                f"HybridStrategyParams: strategy_id must be one of "
                f"{sorted(HYBRID_STRATEGY_IDS)}, got {self.strategy_id!r}."
            )
        if not (0.0 < self.momentum_weight <= 1.0):
            raise ValueError(
                f"momentum_weight must be in (0, 1], got {self.momentum_weight}."
            )
        if not (0.0 <= self.news_weight <= 1.0):
            raise ValueError(
                f"news_weight must be in [0, 1], got {self.news_weight}."
            )
        if self.vol_gate_threshold < 0.0:
            raise ValueError(
                f"vol_gate_threshold must be >= 0, got {self.vol_gate_threshold}."
            )
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0, 1], got {self.min_confidence}."
            )
        if not (0.0 <= self.min_signal_threshold <= 1.0):
            raise ValueError(
                f"min_signal_threshold must be in [0, 1], "
                f"got {self.min_signal_threshold}."
            )

    # ------------------------------------------------------------------
    # Config injection
    # ------------------------------------------------------------------

    def inject_into_cfg(self, base_cfg: dict[str, Any]) -> dict[str, Any]:
        """Return a *copy* of *base_cfg* with these parameters applied.

        The following keys are set:

        ``cfg["strategies"][strategy_id]["signal_blend"]``
            ``{"momentum_weight": <float>, "news_weight": <float>}``

        ``cfg["strategies"][strategy_id]["vol_gate"]``  *(equity only)*
            ``{"threshold": <float>}``

        ``cfg["portfolio"]["min_confidence"]``
            Set to :attr:`min_confidence`.

        ``cfg["portfolio"]["min_signal_threshold"]``
            Set to :attr:`min_signal_threshold`.

        Args:
            base_cfg: Existing merged pipeline config (not mutated).

        Returns:
            A shallow copy of *base_cfg* with these parameters applied.
        """
        cfg: dict[str, Any] = dict(base_cfg)

        # Deep-copy only the sub-dicts we're going to modify
        cfg["strategies"] = dict(cfg.get("strategies", {}))
        cfg["strategies"][self.strategy_id] = dict(
            cfg["strategies"].get(self.strategy_id, {})
        )

        blend: dict[str, float] = {
            "momentum_weight": self.momentum_weight,
            "news_weight": self.news_weight,
        }
        cfg["strategies"][self.strategy_id]["signal_blend"] = blend

        # Vol gate is equity-specific but harmlessly stored for crypto too
        cfg["strategies"][self.strategy_id]["vol_gate"] = {
            "threshold": self.vol_gate_threshold,
        }

        cfg["portfolio"] = dict(cfg.get("portfolio", {}))
        cfg["portfolio"]["min_confidence"] = self.min_confidence
        cfg["portfolio"]["min_signal_threshold"] = self.min_signal_threshold

        return cfg

    # ------------------------------------------------------------------
    # Identity / determinism
    # ------------------------------------------------------------------

    def run_id(self, prefix: str = "hybrid") -> str:
        """Return a short, human-readable, deterministic run identifier.

        The ID encodes the strategy abbreviation and the key parameter values
        so that experiment records can be identified by run ID alone.

        Args:
            prefix: Optional prefix prepended to the ID.

        Returns:
            A string like
            ``"hybrid_emnh_mw0.70_nw0.30_vg0.25_mc0.10"``
            for the equity hybrid strategy.
        """
        abbrev = {
            EQUITY_HYBRID_ID: "emnh",
            CRYPTO_HYBRID_ID: "cmnh",
        }.get(self.strategy_id, self.strategy_id[:4])

        return (
            f"{prefix}_{abbrev}"
            f"_mw{self.momentum_weight:.2f}"
            f"_nw{self.news_weight:.2f}"
            f"_vg{self.vol_gate_threshold:.2f}"
            f"_mc{self.min_confidence:.2f}"
        )

    def config_hash(self) -> str:
        """Return a short SHA-256 hex digest of the parameter values."""
        serialised = json.dumps(self._blend_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialised.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this parameter set."""
        return {
            "strategy_id": self.strategy_id,
            "momentum_weight": self.momentum_weight,
            "news_weight": self.news_weight,
            "vol_gate_threshold": self.vol_gate_threshold,
            "min_confidence": self.min_confidence,
            "min_signal_threshold": self.min_signal_threshold,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HybridStrategyParams":
        """Deserialise from a plain dict (e.g. loaded from JSON)."""
        extra = d.get("extra")
        return cls(
            strategy_id=d.get("strategy_id", EQUITY_HYBRID_ID),
            momentum_weight=float(d.get("momentum_weight", 0.7)),
            news_weight=float(d.get("news_weight", 0.3)),
            vol_gate_threshold=float(d.get("vol_gate_threshold", 0.25)),
            min_confidence=float(d.get("min_confidence", 0.10)),
            min_signal_threshold=float(d.get("min_signal_threshold", 0.05)),
            extra=dict(extra) if isinstance(extra, dict) else {},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blend_dict(self) -> dict[str, Any]:
        """Return the parameter values as a stable dict (excludes ``extra``)."""
        return {
            "strategy_id": self.strategy_id,
            "momentum_weight": self.momentum_weight,
            "news_weight": self.news_weight,
            "vol_gate_threshold": self.vol_gate_threshold,
            "min_confidence": self.min_confidence,
            "min_signal_threshold": self.min_signal_threshold,
        }
