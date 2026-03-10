"""Portfolio policy enforcement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class PolicyViolation(Exception):
    constraint: str
    details: str
    proposed_action: str

    def __str__(self) -> str:
        return f"Policy violation ({self.constraint}): {self.details}. Proposed repair: {self.proposed_action}"


@dataclass(frozen=True)
class PolicyLimits:
    max_positions: int | None = None
    max_weight: float | None = None
    max_turnover: float | None = None
    auto_repair: bool = False


def _policy_limits(config: Dict) -> PolicyLimits:
    policy_cfg = config.get("portfolio", {}) if config else {}
    defaults = {
        "max_positions": policy_cfg.get("max_positions", policy_cfg.get("top_k")),
        "max_weight": policy_cfg.get("max_weight"),
        "max_turnover": policy_cfg.get("max_turnover"),
    }

    policies = policy_cfg.get("policies")
    if isinstance(policies, dict):
        defaults.update(policies)
    elif isinstance(policies, list):
        for item in policies:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("type")
            if name in defaults:
                defaults[name] = item.get("value", defaults[name])

    policy_settings = policy_cfg.get("policy", {}) if isinstance(policy_cfg.get("policy"), dict) else {}
    auto_repair = bool(policy_settings.get("auto_repair", False))

    return PolicyLimits(
        max_positions=int(defaults["max_positions"]) if defaults.get("max_positions") is not None else None,
        max_weight=float(defaults["max_weight"]) if defaults.get("max_weight") is not None else None,
        max_turnover=float(defaults["max_turnover"]) if defaults.get("max_turnover") is not None else None,
        auto_repair=auto_repair,
    )


def enforce_policy(tickers: List[str], config: dict) -> Tuple[List[str], List[Dict]]:
    limits = _policy_limits(config)
    repairs: List[Dict] = []
    if limits.max_positions is not None and len(tickers) > limits.max_positions:
        if limits.auto_repair:
            kept = sorted(tickers)[: limits.max_positions]
            dropped = sorted(set(tickers) - set(kept))
            repairs.append(
                {
                    "constraint": "max_positions",
                    "action": "drop_excess",
                    "dropped_tickers": ",".join(dropped),
                }
            )
            return kept, repairs
        raise PolicyViolation(
            "max_positions",
            f"observed={len(tickers)} limit={limits.max_positions}",
            "Drop lowest-ranked tickers or reduce universe size.",
        )
    return tickers, repairs


def enforce_weight_policy(weights: pd.Series, config: dict) -> Tuple[pd.Series, List[Dict]]:
    limits = _policy_limits(config)
    repairs: List[Dict] = []
    if limits.max_weight is not None and (weights > limits.max_weight).any():
        if limits.auto_repair:
            clipped = weights.clip(upper=limits.max_weight)
            if clipped.sum() > 0:
                clipped = clipped / clipped.sum()
            repairs.append(
                {
                    "constraint": "max_weight",
                    "action": "clip_and_renormalize",
                    "max_weight": limits.max_weight,
                }
            )
            return clipped, repairs
        raise PolicyViolation(
            "max_weight",
            f"limit={limits.max_weight} weights={weights[weights > limits.max_weight].to_dict()}",
            "Clip weights to max_weight and renormalize.",
        )
    return weights, repairs


def enforce_turnover_policy(
    weights: pd.Series, prior_weights: pd.Series, config: dict
) -> Tuple[pd.Series, List[Dict]]:
    limits = _policy_limits(config)
    repairs: List[Dict] = []
    if limits.max_turnover is None:
        return weights, repairs
    aligned_prior = prior_weights.reindex(weights.index).fillna(0.0)
    turnover = float((weights - aligned_prior).abs().sum())
    if turnover > limits.max_turnover:
        if limits.auto_repair:
            scale = limits.max_turnover / turnover if turnover > 0 else 0
            repaired = aligned_prior + (weights - aligned_prior) * scale
            if repaired.sum() > 0:
                repaired = repaired / repaired.sum()
            repairs.append(
                {
                    "constraint": "max_turnover",
                    "action": "scale_turnover",
                    "max_turnover": limits.max_turnover,
                    "observed": turnover,
                }
            )
            return repaired, repairs
        raise PolicyViolation(
            "max_turnover",
            f"observed={turnover:.4f} limit={limits.max_turnover}",
            "Scale turnover by reducing weight changes.",
        )
    return weights, repairs
