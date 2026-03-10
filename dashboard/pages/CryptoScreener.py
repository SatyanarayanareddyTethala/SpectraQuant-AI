"""Crypto Screener dashboard page.

Displays a filterable, sortable table of crypto assets derived from
``data/crypto/asset_master.parquet`` and ``data/crypto/market_snapshot.parquet``.

Features
--------
- Columns: name, symbol, price_usd, market_cap_usd, volume_24h, change_24h_pct,
  all_time_high_usd, all_time_low_usd, category, network_type, community_rank, age_days
- Filters: category, network_type, min market cap, min volume, age range, change range
- Sorting: click any column header
- Universe explanation: shows "Why in universe?" from latest universe report
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ImportError:
    raise SystemExit("streamlit is required for the dashboard")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

DATA_CRYPTO_DIR = BASE_DIR / "data" / "crypto"
REPORTS_UNIVERSE_DIR = BASE_DIR / "reports" / "universe"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_screener_data() -> pd.DataFrame:
    """Load and merge asset_master + latest market_snapshot."""
    am_path = DATA_CRYPTO_DIR / "asset_master.parquet"
    snap_path = DATA_CRYPTO_DIR / "market_snapshot.parquet"

    if not am_path.exists() and not snap_path.exists():
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []

    if snap_path.exists():
        snap = pd.read_parquet(snap_path, engine="pyarrow")
        # Keep latest snapshot per symbol
        if "as_of" in snap.columns:
            snap = snap.sort_values("as_of", ascending=False).drop_duplicates(
                subset=["canonical_symbol"], keep="first"
            )
        dfs.append(snap)

    if am_path.exists():
        master = pd.read_parquet(am_path, engine="pyarrow")
        if dfs:
            merged = dfs[0].merge(
                master,
                on="canonical_symbol",
                how="left",
                suffixes=("", "_master"),
            )
            # Prefer snapshot values for any duplicated columns
            dup_cols = [c for c in merged.columns if c.endswith("_master")]
            for col in dup_cols:
                base = col[: -len("_master")]
                if base not in merged.columns:
                    merged[base] = merged[col]
                merged = merged.drop(columns=[col])
            return merged
        else:
            return master

    return dfs[0] if dfs else pd.DataFrame()


def _load_universe_reasons() -> dict[str, str]:
    """Load the latest universe report and return {symbol: reason}."""
    if not REPORTS_UNIVERSE_DIR.exists():
        return {}
    reports = sorted(REPORTS_UNIVERSE_DIR.glob("crypto_universe_*.csv"), reverse=True)
    if not reports:
        return {}
    try:
        df = pd.read_csv(reports[0])
        if "canonical_symbol" in df.columns and "universe_reason" in df.columns:
            return dict(zip(df["canonical_symbol"], df["universe_reason"]))
    except Exception:
        pass
    return {}


def _age_days(launch_date: object) -> float | None:
    if pd.isna(launch_date):
        return None
    now = datetime.now(timezone.utc)
    try:
        ts = pd.Timestamp(launch_date)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return (now - ts.to_pydatetime()).days
    except Exception:
        return None


def _fmt_large(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "-"
    if val >= 1e9:
        return f"${val / 1e9:.2f}B"
    if val >= 1e6:
        return f"${val / 1e6:.1f}M"
    if val >= 1e3:
        return f"${val / 1e3:.1f}K"
    return f"${val:.2f}"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render() -> None:
    st.set_page_config(page_title="Crypto Screener", layout="wide")
    st.title("📊 Crypto Screener")
    st.caption(
        "Asset data from `data/crypto/asset_master.parquet` + "
        "`data/crypto/market_snapshot.parquet`. "
        "Ingest fresh data with: `spectraquant crypto-ingest-dataset --path <csv>`"
    )

    df = _load_screener_data()

    if df.empty:
        st.warning(
            "No crypto dataset found. Run: "
            "`spectraquant crypto-ingest-dataset --path data/crypto/cryptocurrency_dataset.csv`"
        )
        return

    # --- Compute derived columns ------------------------------------------
    if "launch_date" in df.columns:
        df["age_days"] = df["launch_date"].apply(_age_days).astype("float64")
    else:
        df["age_days"] = None

    reasons = _load_universe_reasons()
    if reasons:
        df["universe_reason"] = df["canonical_symbol"].map(reasons).fillna("not evaluated")

    # --- Sidebar filters --------------------------------------------------
    st.sidebar.header("Filters")

    # Category filter
    if "category" in df.columns:
        cats = sorted(df["category"].dropna().unique().tolist())
        selected_cats = st.sidebar.multiselect("Category", options=cats, default=[])
        if selected_cats:
            df = df[df["category"].isin(selected_cats)]

    # Network type filter
    if "network_type" in df.columns:
        nets = sorted(df["network_type"].dropna().unique().tolist())
        selected_nets = st.sidebar.multiselect("Network Type", options=nets, default=[])
        if selected_nets:
            df = df[df["network_type"].isin(selected_nets)]

    # Min market cap
    if "market_cap_usd" in df.columns:
        mcap_vals = df["market_cap_usd"].dropna()
        if not mcap_vals.empty:
            min_mcap = st.sidebar.number_input(
                "Min Market Cap (USD)",
                min_value=0,
                max_value=int(mcap_vals.max()),
                value=0,
                step=1_000_000,
            )
            df = df[df["market_cap_usd"].fillna(0) >= min_mcap]

    # Min volume
    if "volume_24h_usd" in df.columns:
        vol_vals = df["volume_24h_usd"].dropna()
        if not vol_vals.empty:
            min_vol = st.sidebar.number_input(
                "Min 24h Volume (USD)",
                min_value=0,
                max_value=int(vol_vals.max()),
                value=0,
                step=100_000,
            )
            df = df[df["volume_24h_usd"].fillna(0) >= min_vol]

    # Age range
    if "age_days" in df.columns:
        age_vals = df["age_days"].dropna()
        if not age_vals.empty:
            min_age, max_age = st.sidebar.slider(
                "Age (days)",
                min_value=0,
                max_value=int(age_vals.max()),
                value=(0, int(age_vals.max())),
            )
            df = df[
                (df["age_days"].fillna(0) >= min_age)
                & (df["age_days"].fillna(0) <= max_age)
            ]

    # Change range
    if "change_24h_pct" in df.columns:
        chg_vals = df["change_24h_pct"].dropna()
        if not chg_vals.empty:
            chg_min = float(chg_vals.min())
            chg_max = float(chg_vals.max())
            chg_lo, chg_hi = st.sidebar.slider(
                "24h Change (%)",
                min_value=round(chg_min, 1),
                max_value=round(chg_max, 1),
                value=(round(chg_min, 1), round(chg_max, 1)),
                step=0.1,
            )
            df = df[
                (df["change_24h_pct"].fillna(0) >= chg_lo)
                & (df["change_24h_pct"].fillna(0) <= chg_hi)
            ]

    # --- Sort control -----------------------------------------------------
    sort_options = [
        c for c in ["market_cap_usd", "volume_24h_usd", "change_24h_pct", "age_days", "community_rank"]
        if c in df.columns
    ]
    if sort_options:
        sort_col = st.selectbox("Sort by", options=sort_options, index=0)
        sort_asc = st.checkbox("Ascending", value=False)
        df = df.sort_values(sort_col, ascending=sort_asc, na_position="last")

    # --- Display columns --------------------------------------------------
    display_cols = [
        c for c in [
            "canonical_symbol", "name", "price_usd", "market_cap_usd",
            "volume_24h_usd", "change_24h_pct",
            "all_time_high_usd", "all_time_low_usd",
            "category", "network_type", "community_rank", "age_days",
        ]
        if c in df.columns
    ]
    if "universe_reason" in df.columns:
        display_cols.append("universe_reason")

    st.metric("Assets shown", len(df))
    st.dataframe(df[display_cols].reset_index(drop=True), use_container_width=True)

    # --- Watchlist --------------------------------------------------------
    st.subheader("Watchlist")
    if "canonical_symbol" in df.columns:
        watchlist = st.multiselect(
            "Add to watchlist",
            options=list(df["canonical_symbol"].values),
            default=[],
        )
        if watchlist:
            st.dataframe(
                df[df["canonical_symbol"].isin(watchlist)][display_cols].reset_index(drop=True),
                use_container_width=True,
            )


if __name__ == "__main__":
    render()
else:
    render()
