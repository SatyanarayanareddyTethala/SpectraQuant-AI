"""Meta-Learner Controller — auto-tune signal thresholds and expert weights.

Uses Failure Memory statistics and recent trade outcomes to propose bounded
policy updates that are persisted in ``data/state/intelligence_policy.json``.

Safety constraints
------------------
- All changes are bounded (never exceed configured limits).
- Updates are versioned; any version can be rolled back.
- Policy is validated before being applied.
- Never modifies the on-disk ``config.yaml`` directly.
"""
from __future__ import annotations

import copy
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default policy structure
# ---------------------------------------------------------------------------

_DEFAULT_POLICY: Dict[str, Any] = {
    "version": "0",
    "created_at": "",
    "thresholds": {
        "buy": 0.55,
        "sell": 0.45,
        "alpha": 0.0,
    },
    "expert_weights": {},       # ticker or expert-id → weight multiplier
    "max_candidates": 50,       # news_universe max candidates
    "cooldown_minutes": 60,
}

# Bounds for safety
_POLICY_BOUNDS: Dict[str, Tuple[float, float]] = {
    "thresholds.buy": (0.40, 0.80),
    "thresholds.sell": (0.20, 0.60),
    "thresholds.alpha": (-1.0, 1.0),
    "max_candidates": (5.0, 200.0),
    "cooldown_minutes": (5.0, 1440.0),
}

# How aggressively to update thresholds per meta-update cycle
_STEP_SIZE = 0.02
_EXPERT_STEP = 0.05
_MAX_EXPERT_WEIGHT = 3.0
_MIN_EXPERT_WEIGHT = 0.1

# Failure-rate thresholds that drive policy adjustments
_HIGH_FAILURE_RATE = 0.50      # failure rate above this → tighten buy threshold
_LOW_FAILURE_RATE = 0.20       # failure rate below this (+ good win rate) → relax
_HIGH_WIN_RATE = 0.60          # win rate above this considered "good"
_POOR_EXPERT_ACCURACY = 0.40   # accuracy below this → reduce weight
_GOOD_EXPERT_ACCURACY = 0.65   # accuracy above this → increase weight


# ---------------------------------------------------------------------------
# Policy persistence helpers
# ---------------------------------------------------------------------------

def _policy_path(state_dir: str) -> Path:
    return Path(state_dir) / "intelligence_policy.json"


def _load_policy(state_dir: str) -> Dict[str, Any]:
    """Load current policy from disk; fall back to defaults if not found."""
    path = _policy_path(state_dir)
    if path.exists():
        try:
            with open(path) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load policy (%s); using defaults", exc)
    return copy.deepcopy(_DEFAULT_POLICY)


def _save_policy(policy: Dict[str, Any], state_dir: str) -> None:
    """Persist policy JSON and keep the last N versions."""
    path = _policy_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Archive previous version
    if path.exists():
        archive_dir = path.parent / "policy_archive"
        archive_dir.mkdir(exist_ok=True)
        old_policy = json.loads(path.read_text())
        old_ver = old_policy.get("version", "0")
        archive_path = archive_dir / f"intelligence_policy_v{old_ver}.json"
        archive_path.write_text(json.dumps(old_policy, indent=2))
        # Prune archives: keep last 20
        archives = sorted(archive_dir.glob("*.json"))
        for excess in archives[:-20]:
            excess.unlink()

    with open(path, "w") as fh:
        json.dump(policy, fh, indent=2)
    logger.debug("Policy saved (version=%s)", policy.get("version"))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def propose_policy_update(
    metrics: Dict[str, Any],
    current_policy: Optional[Dict[str, Any]] = None,
    state_dir: str = "data/state",
) -> Tuple[Dict[str, Any], List[str]]:
    """Suggest a new policy based on failure/performance metrics.

    Parameters
    ----------
    metrics : dict
        Failure statistics (output of ``update_failure_stats``) plus optional
        keys: ``recent_win_rate``, ``recent_return``, ``expert_accuracy``.
    current_policy : dict, optional
        Current policy; loaded from disk if *None*.
    state_dir : str
        Directory for policy JSON.

    Returns
    -------
    tuple[dict, list[str]]
        ``(new_policy, reasons)`` where *reasons* explains each change.
    """
    if current_policy is None:
        current_policy = _load_policy(state_dir)

    new_policy = copy.deepcopy(current_policy)
    reasons: List[str] = []

    total_trades = metrics.get("total_trades", 0)
    total_failures = metrics.get("total_failures", 0)
    win_rate = metrics.get("recent_win_rate", None)
    expert_accuracy: Dict[str, float] = metrics.get("expert_accuracy", {})

    if total_trades < 5:
        return new_policy, ["Insufficient trades for policy update"]

    failure_rate = total_failures / max(total_trades, 1)

    # ---- Adjust buy/sell thresholds based on failure rate ---------------
    buy_t = new_policy["thresholds"]["buy"]
    sell_t = new_policy["thresholds"]["sell"]

    if failure_rate > _HIGH_FAILURE_RATE:
        # Too many failures → tighten buy threshold (be more selective)
        new_buy = _clamp(buy_t + _STEP_SIZE, *_POLICY_BOUNDS["thresholds.buy"])
        if abs(new_buy - buy_t) > 1e-6:
            new_policy["thresholds"]["buy"] = round(new_buy, 4)
            reasons.append(
                f"Failure rate {failure_rate:.1%} > {_HIGH_FAILURE_RATE:.0%}: raised buy threshold "
                f"{buy_t:.3f} → {new_buy:.3f}"
            )
    elif failure_rate < _LOW_FAILURE_RATE and win_rate is not None and win_rate > _HIGH_WIN_RATE:
        # Low failures + high win rate → can relax threshold slightly
        new_buy = _clamp(buy_t - _STEP_SIZE / 2, *_POLICY_BOUNDS["thresholds.buy"])
        if abs(new_buy - buy_t) > 1e-6:
            new_policy["thresholds"]["buy"] = round(new_buy, 4)
            reasons.append(
                f"Low failure rate {failure_rate:.1%} + win_rate {win_rate:.1%}: "
                f"lowered buy threshold {buy_t:.3f} → {new_buy:.3f}"
            )

    # ---- Adjust expert weights based on accuracy ------------------------
    for expert_id, accuracy in expert_accuracy.items():
        current_weight = new_policy["expert_weights"].get(expert_id, 1.0)
        if accuracy < _POOR_EXPERT_ACCURACY:
            # Poor expert → reduce weight
            new_weight = _clamp(
                current_weight - _EXPERT_STEP,
                _MIN_EXPERT_WEIGHT,
                _MAX_EXPERT_WEIGHT,
            )
        elif accuracy > _GOOD_EXPERT_ACCURACY:
            # Good expert → increase weight
            new_weight = _clamp(
                current_weight + _EXPERT_STEP,
                _MIN_EXPERT_WEIGHT,
                _MAX_EXPERT_WEIGHT,
            )
        else:
            continue

        new_weight = round(new_weight, 4)
        if abs(new_weight - current_weight) > 1e-6:
            new_policy["expert_weights"][expert_id] = new_weight
            reasons.append(
                f"Expert '{expert_id}' accuracy={accuracy:.1%}: "
                f"weight {current_weight:.3f} → {new_weight:.3f}"
            )

    # ---- Check failure by type for specialized adjustments --------------
    by_regime = metrics.get("by_regime", {})
    overconfidence_total = sum(
        counts.get("OVERCONFIDENCE", 0) for counts in by_regime.values()
    )
    if overconfidence_total > 5:
        new_buy = _clamp(
            new_policy["thresholds"]["buy"] + _STEP_SIZE * 0.5,
            *_POLICY_BOUNDS["thresholds.buy"],
        )
        new_policy["thresholds"]["buy"] = round(new_buy, 4)
        reasons.append(
            f"Overconfidence failures={overconfidence_total}: "
            f"further tightened buy threshold to {new_buy:.3f}"
        )

    # Stamp new version
    new_policy["version"] = str(uuid.uuid4())[:8]
    new_policy["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

    if not reasons:
        reasons.append("No adjustment needed; policy unchanged")

    return new_policy, reasons


def validate_policy_update(
    new_policy: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Check that a proposed policy is within bounds.

    Parameters
    ----------
    new_policy : dict
        Proposed policy dictionary.

    Returns
    -------
    tuple[bool, list[str]]
        ``(ok, errors)``
    """
    errors: List[str] = []

    def _check(path: str, val: Any) -> None:
        if path in _POLICY_BOUNDS:
            lo, hi = _POLICY_BOUNDS[path]
            if not (lo <= float(val) <= hi):
                errors.append(f"{path}={val} out of bounds [{lo}, {hi}]")

    thresholds = new_policy.get("thresholds", {})
    _check("thresholds.buy", thresholds.get("buy", 0.55))
    _check("thresholds.sell", thresholds.get("sell", 0.45))
    _check("thresholds.alpha", thresholds.get("alpha", 0.0))
    _check("max_candidates", new_policy.get("max_candidates", 50))
    _check("cooldown_minutes", new_policy.get("cooldown_minutes", 60))

    # Expert weights
    for eid, w in new_policy.get("expert_weights", {}).items():
        lo, hi = _MIN_EXPERT_WEIGHT, _MAX_EXPERT_WEIGHT
        if not (lo <= float(w) <= hi):
            errors.append(f"expert_weight[{eid}]={w} out of bounds [{lo}, {hi}]")

    # buy > sell
    buy = float(thresholds.get("buy", 0.55))
    sell = float(thresholds.get("sell", 0.45))
    if buy <= sell:
        errors.append(f"thresholds.buy ({buy}) must be > thresholds.sell ({sell})")

    return len(errors) == 0, errors


def apply_policy_update(
    config: Dict[str, Any],
    new_policy: Dict[str, Any],
    state_dir: str = "data/state",
) -> Dict[str, Any]:
    """Apply a validated policy to an in-memory config dict and persist policy.

    Parameters
    ----------
    config : dict
        The current loaded config (mutated in-place).
    new_policy : dict
        Validated proposed policy.
    state_dir : str
        Directory for policy JSON.

    Returns
    -------
    dict
        Updated config dict.
    """
    ok, errors = validate_policy_update(new_policy)
    if not ok:
        raise ValueError(f"Policy validation failed: {errors}")

    # Apply thresholds to intraday signal thresholds section
    thresholds = new_policy.get("thresholds", {})
    if "intraday" in config:
        config["intraday"].setdefault("signal_thresholds", {})
        if "buy" in thresholds:
            config["intraday"]["signal_thresholds"]["buy"] = thresholds["buy"]
        if "sell" in thresholds:
            config["intraday"]["signal_thresholds"]["sell"] = thresholds["sell"]

    # Apply alpha threshold to portfolio section
    if "portfolio" in config and "alpha" in thresholds:
        config["portfolio"]["alpha_threshold"] = thresholds["alpha"]

    # Apply max_candidates to news_universe
    if "news_universe" in config and "max_candidates" in new_policy:
        config["news_universe"]["max_candidates"] = int(new_policy["max_candidates"])

    # Persist policy
    _save_policy(new_policy, state_dir)
    logger.info("Policy applied (version=%s)", new_policy.get("version"))
    return config


def rollback_policy(
    version_id: str,
    state_dir: str = "data/state",
) -> Optional[Dict[str, Any]]:
    """Restore a previous policy version.

    Parameters
    ----------
    version_id : str
        Version string (partial match against archive filenames).
    state_dir : str
        Directory containing policy JSON and archive.

    Returns
    -------
    dict or None
        Restored policy, or *None* if not found.
    """
    archive_dir = Path(state_dir) / "policy_archive"
    if not archive_dir.exists():
        logger.warning("No policy archive found at %s", archive_dir)
        return None

    candidates = list(archive_dir.glob(f"*{version_id}*.json"))
    if not candidates:
        logger.warning("No archive match for version_id=%s", version_id)
        return None

    # Use the most recently modified matching file
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    target = candidates[0]
    try:
        policy = json.loads(target.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load archived policy %s: %s", target, exc)
        return None

    _save_policy(policy, state_dir)
    logger.info("Rolled back to policy version from %s", target.name)
    return policy
