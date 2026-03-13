"""Quality-gated universe selection for crypto assets.

Provides three universe modes:
  - ``news_first``         – existing news-driven selection
  - ``dataset_topN``       – rank by liquidity score from dataset
  - ``hybrid_news_dataset`` – union of news picks + top-liquid, then gate

Scoring
-------
  liquidity_score  = log10(market_cap_usd + 1) * 0.6
                   + log10(volume_24h_usd  + 1) * 0.4
  momentum_hint    = clip(change_24h_pct, -20, 20)   # for ranking only

Quality gate defaults
---------------------
  min_market_cap_usd    : 50_000_000  (50 M USD)
  min_24h_volume_usd    : 1_000_000   (1 M USD)
  min_age_days          : 180         (6 months)
  min_community_rank    : None        (disabled)
"""
from __future__ import annotations

import csv
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default gate thresholds (may be overridden via config)
# ---------------------------------------------------------------------------
_DEFAULT_MIN_MCAP = 50_000_000.0
_DEFAULT_MIN_VOLUME = 1_000_000.0
_DEFAULT_MIN_AGE_DAYS = 180
_DEFAULT_MAX_N = 20


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _liquidity_score(mcap: float, vol: float) -> float:
    return math.log10(max(mcap, 0) + 1) * 0.6 + math.log10(max(vol, 0) + 1) * 0.4


def _momentum_hint(change_pct: float) -> float:
    """Clip 24 h change to [-20, 20] – used only for ranking, not signals."""
    return max(-20.0, min(20.0, float(change_pct or 0.0)))


# ---------------------------------------------------------------------------
# Age computation
# ---------------------------------------------------------------------------

def _age_days(launch_date: Any) -> float | None:
    """Return age in days from launch_date to UTC now, or None if unparseable."""
    if pd.isna(launch_date):
        return None
    now = datetime.now(timezone.utc)
    try:
        if isinstance(launch_date, (pd.Timestamp, datetime)):
            ld = pd.Timestamp(launch_date)
            if ld.tzinfo is None:
                ld = ld.tz_localize("UTC")
            return (now - ld.to_pydatetime()).days
        parsed = pd.to_datetime(str(launch_date), utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return (now - parsed.to_pydatetime()).days
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------

def apply_quality_gate(
    df: pd.DataFrame,
    *,
    min_market_cap_usd: float = _DEFAULT_MIN_MCAP,
    min_24h_volume_usd: float = _DEFAULT_MIN_VOLUME,
    min_age_days: int = _DEFAULT_MIN_AGE_DAYS,
    min_community_rank: float | None = None,
) -> pd.DataFrame:
    """Filter *df* (asset_master + market_snapshot merged view) by quality criteria.

    Parameters
    ----------
    df:
        DataFrame with at least ``canonical_symbol``.  Optional columns:
        ``market_cap_usd``, ``volume_24h_usd``, ``launch_date``,
        ``community_rank``.
    min_market_cap_usd, min_24h_volume_usd, min_age_days, min_community_rank:
        Filtering thresholds.

    Returns
    -------
    pd.DataFrame
        Filtered + annotated DataFrame with added columns:
        ``liquidity_score``, ``momentum_hint``, ``age_days``,
        ``universe_reason`` (comma-separated reasons for inclusion/exclusion).
    """
    out = df.copy()

    # ---- computed fields --------------------------------------------------
    mcap = pd.to_numeric(out.get("market_cap_usd", 0), errors="coerce").fillna(0.0)
    vol = pd.to_numeric(out.get("volume_24h_usd", 0), errors="coerce").fillna(0.0)
    chg = pd.to_numeric(out.get("change_24h_pct", 0), errors="coerce").fillna(0.0)

    out["liquidity_score"] = [_liquidity_score(m, v) for m, v in zip(mcap, vol)]
    out["momentum_hint"] = chg.apply(_momentum_hint)

    if "launch_date" in out.columns:
        out["age_days"] = out["launch_date"].apply(_age_days)
    else:
        out["age_days"] = None

    # ---- gate flags -------------------------------------------------------
    reasons: list[list[str]] = [[] for _ in range(len(out))]

    if min_market_cap_usd > 0:
        fails = mcap < min_market_cap_usd
        for i in fails[fails].index:
            reasons[out.index.get_loc(i)].append(
                f"mcap_too_low({mcap[i]:.0f}<{min_market_cap_usd:.0f})"
            )

    if min_24h_volume_usd > 0:
        fails = vol < min_24h_volume_usd
        for i in fails[fails].index:
            reasons[out.index.get_loc(i)].append(
                f"vol_too_low({vol[i]:.0f}<{min_24h_volume_usd:.0f})"
            )

    if min_age_days > 0 and "age_days" in out.columns:
        for idx, row in out.iterrows():
            age = row.get("age_days")
            if age is None or age < min_age_days:
                reasons[out.index.get_loc(idx)].append(
                    f"too_young({age!r}<{min_age_days})"
                )

    if min_community_rank is not None and "community_rank" in out.columns:
        cr = pd.to_numeric(out["community_rank"], errors="coerce").fillna(0.0)
        fails = cr < min_community_rank
        for i in fails[fails].index:
            reasons[out.index.get_loc(i)].append(
                f"community_rank_low({cr[i]:.1f}<{min_community_rank:.1f})"
            )

    out["_exclusion_reasons"] = [",".join(r) for r in reasons]
    out["universe_included"] = out["_exclusion_reasons"] == ""
    out["universe_reason"] = out.apply(
        lambda r: "included" if r["universe_included"] else r["_exclusion_reasons"],
        axis=1,
    )
    out = out.drop(columns=["_exclusion_reasons"])

    included = out[out["universe_included"]].copy()
    logger.info(
        "Quality gate: %d/%d assets passed (mcap>=%s, vol>=%s, age>=%s days)",
        len(included),
        len(out),
        min_market_cap_usd,
        min_24h_volume_usd,
        min_age_days,
    )
    return included


# ---------------------------------------------------------------------------
# Universe builders
# ---------------------------------------------------------------------------

def build_dataset_topN_universe(
    snapshot: pd.DataFrame,
    top_n: int = _DEFAULT_MAX_N,
    gate_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Rank by liquidity score and return top-N symbols passing the gate.

    Parameters
    ----------
    snapshot:
        Latest market snapshot with ``canonical_symbol``, ``market_cap_usd``,
        ``volume_24h_usd``, optional ``launch_date``.
    top_n:
        Maximum assets to return.
    gate_kwargs:
        Extra kwargs forwarded to :func:`apply_quality_gate`.

    Returns
    -------
    list[str]
        Canonical symbols (e.g. ``["BTC", "ETH"]``).
    """
    if snapshot.empty:
        logger.warning("dataset_topN: snapshot is empty; returning []")
        return []

    gated = apply_quality_gate(snapshot, **(gate_kwargs or {}))
    if gated.empty:
        return []

    ranked = gated.sort_values("liquidity_score", ascending=False).head(top_n)
    return list(ranked["canonical_symbol"].values)


def build_hybrid_universe(
    news_symbols: list[str],
    snapshot: pd.DataFrame,
    top_n: int = _DEFAULT_MAX_N,
    gate_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Union of news picks + top-liquid from dataset, then quality gate.

    Parameters
    ----------
    news_symbols:
        Symbols from news pipeline (may include ``-USD`` suffix).
    snapshot:
        Latest market snapshot.
    top_n, gate_kwargs:
        Forwarded to :func:`build_dataset_topN_universe`.
    """
    # Normalise news symbols to canonical form
    news_canonical = [s.replace("-USD", "").upper() for s in news_symbols]

    dataset_top = build_dataset_topN_universe(snapshot, top_n=top_n, gate_kwargs=gate_kwargs)

    union: list[str] = list(dict.fromkeys(news_canonical + dataset_top))  # preserve order
    logger.info(
        "Hybrid universe: %d news + %d dataset_top = %d unique",
        len(news_canonical),
        len(dataset_top),
        len(union),
    )

    # Apply gate on the union (only symbols that appear in snapshot)
    if not snapshot.empty and "canonical_symbol" in snapshot.columns:
        snap_filtered = snapshot[snapshot["canonical_symbol"].isin(union)]
        gated = apply_quality_gate(snap_filtered, **(gate_kwargs or {}))
        passed = set(gated["canonical_symbol"].values)
        # News symbols that don't appear in dataset still pass through (no data to gate them)
        no_data_syms = [s for s in news_canonical if s not in snapshot["canonical_symbol"].values]
        result = [s for s in union if s in passed or s in no_data_syms]
    else:
        result = union

    return result[:top_n]


# ---------------------------------------------------------------------------
# Universe report
# ---------------------------------------------------------------------------

def write_universe_report(
    full_df: pd.DataFrame,
    out_dir: str | Path = "reports/universe",
) -> Path:
    """Write ``crypto_universe_<ts>.csv`` with inclusion reasons.

    Parameters
    ----------
    full_df:
        DataFrame returned by ``apply_quality_gate`` on the *full* dataset
        (including excluded assets).  Must have ``universe_reason`` column.
    out_dir:
        Directory to write the report.

    Returns
    -------
    Path
        Path to the written report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"crypto_universe_{ts}.csv"

    cols_to_write = [
        c for c in [
            "canonical_symbol", "name", "price_usd", "market_cap_usd",
            "volume_24h_usd", "change_24h_pct", "age_days",
            "liquidity_score", "momentum_hint", "community_rank",
            "category", "network_type", "universe_included", "universe_reason",
        ]
        if c in full_df.columns
    ]
    full_df[cols_to_write].to_csv(path, index=False)
    logger.info("Wrote universe report → %s", path)
    return path
