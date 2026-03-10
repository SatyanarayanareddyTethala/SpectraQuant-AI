from __future__ import annotations
import io, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from typing import Optional
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SpectraQuant Dashboard", layout="wide", page_icon="📊")

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from spectraquant.config import get_config
from dashboard.utils.artifacts import load_latest_prices
from dashboard.utils.diagnostics import (
    EMPTY_AFTER_DROPNA,
    MISSING_REQUIRED_ARTIFACT,
    Diagnostic,
    make_diagnostic,
    render_diagnostics,
)
from dashboard.utils.simulator import (
    simulate_portfolio_from_signals,
)
from dashboard.utils.feature_importance import parse_feature_importance
from dashboard.utils.logging import configure_logger, get_recent_logs
from dashboard.utils.manifest import discover_runs, load_run_manifest, resolve_manifest_paths
from dashboard.utils.streamlit_compat import rerun


def parse_horizon_days(horizon: str, trading_minutes_per_day: int = 390) -> float:
    horizon = horizon.strip().lower()
    if horizon.endswith("d"):
        return float(horizon[:-1] or 1)
    if horizon.endswith("m"):
        return float(horizon[:-1] or 0) / trading_minutes_per_day
    if horizon.endswith("h"):
        return float(horizon[:-1] or 0) * 60 / trading_minutes_per_day
    return 1.0


def infer_market(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker.endswith(".NS"):
        return "India"
    if ticker.endswith(".L"):
        return "UK"
    return "Global"


def infer_currency(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker.endswith(".NS"):
        return "INR"
    if ticker.endswith(".L"):
        return "GBP"
    return "Local"

ROOT_DIR = Path(__file__).resolve().parent.parent
PIPELINE_STEPS = ["build-dataset", "train", "predict", "signals", "portfolio"]
PREDICTIONS_GLOB = "reports/predictions/research/predictions_*.csv"
SIGNALS_GLOB = "reports/signals/research/top_signals_*.csv"
PORTFOLIO_DIR = ROOT_DIR / "reports" / "portfolio" / "research"

CSS = """<style>:root{--bg:#eef2f7;--card:#fff;--text:#1f2a44;--muted:#6b7280;--shadow:0 8px 20px rgba(30,41,59,.08);--border:#e5e7eb;--green:#22c55e;--red:#ef4444}.stApp{background:var(--bg)}.block-container{padding-top:1.2rem}.header-title{text-align:center;font-size:1.8rem;font-weight:700;color:var(--text)}.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px;box-shadow:var(--shadow)}.badge{padding:6px 14px;border-radius:999px;color:#fff;font-weight:700;font-size:.85rem}</style>"""
st.markdown(CSS, unsafe_allow_html=True)

logger = configure_logger()

if "run_id" not in st.session_state:
    st.session_state.run_id = "latest"


def _cache_key(prefix: str, run_id: str, params: dict | None = None) -> str:
    if not params:
        return f"{prefix}::{run_id}"
    payload = json.dumps(params, sort_keys=True, default=str)
    return f"{prefix}::{run_id}::{hash(payload)}"


def _run_option_label(run_id: str, has_manifest: bool) -> str:
    marker = "✅" if has_manifest else "—"
    return f"{run_id} {marker}"


def render_run_selector() -> dict[str, Optional[Path]]:
    runs = discover_runs(ROOT_DIR)
    run_lookup = {run.run_id: run for run in runs}
    run_ids = [run.run_id for run in runs]
    current_run = st.session_state.run_id if st.session_state.run_id in run_lookup else run_ids[-1]
    selected = st.sidebar.selectbox(
        "Run Selector",
        options=run_ids,
        index=run_ids.index(current_run) if current_run in run_ids else 0,
        format_func=lambda run_id: _run_option_label(run_id, run_lookup[run_id].has_manifest),
    )
    if selected != st.session_state.run_id:
        previous = st.session_state.run_id
        for prefix in ("artifacts", "sim", "explain", "metrics"):
            st.session_state.pop(_cache_key(prefix, previous), None)
        st.session_state.run_id = selected
        rerun()
    run_info = run_lookup[selected]
    return {
        "run_id": run_info.run_id,
        "run_dir": run_info.run_dir,
        "manifest_path": run_info.manifest_path,
    }


@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, mtime: float | None) -> Optional[pd.DataFrame]:
    if not path_str:
        return None
    try:
        return pd.read_csv(path_str)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_json_cached(path_str: str, mtime: float | None) -> Optional[dict]:
    if not path_str:
        return None
    try:
        return json.loads(Path(path_str).read_text())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_feature_importance_cached(run_id: str, model_path: str, mtime: float | None) -> pd.DataFrame:
    _ = run_id
    return parse_feature_importance(model_path)


def _artifact_mtime(path: Optional[Path]) -> float | None:
    if not path or not path.exists():
        return None
    return path.stat().st_mtime


def resolve_artifact_paths(run_dir: Optional[Path], manifest_path: Optional[Path]) -> dict[str, Path]:
    manifest = load_run_manifest(run_dir) if run_dir else None
    if manifest_path and manifest is None:
        manifest = load_json_cached(str(manifest_path), _artifact_mtime(manifest_path))
    if manifest:
        resolved = resolve_manifest_paths(manifest, run_dir)
        if resolved:
            return resolved
    return {
        "predictions": latest_file(PREDICTIONS_GLOB),
        "signals": latest_file(SIGNALS_GLOB),
        "weights": PORTFOLIO_DIR / "portfolio_weights.csv",
        "returns": PORTFOLIO_DIR / "portfolio_returns.csv",
        "metrics": PORTFOLIO_DIR / "portfolio_metrics.json",
        "config": ROOT_DIR / "config.yaml",
    }

def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def latest_file(pattern: str) -> Optional[Path]:
    matches = [p for p in ROOT_DIR.glob(pattern) if p.is_file()]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None

def load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    return load_csv_cached(str(path), _artifact_mtime(path))


def load_json(path: Path) -> Optional[dict]:
    return load_json_cached(str(path), _artifact_mtime(path))

@st.cache_data(show_spinner=False)
def load_artifacts_cached(
    run_id: str,
    paths: dict[str, str],
    mtimes: dict[str, float | None],
) -> dict[str, Optional[object]]:
    _ = run_id, mtimes
    artifacts: dict[str, Optional[object]] = {}
    for key, path_str in paths.items():
        if not path_str:
            artifacts[key] = None
            continue
        path = Path(path_str)
        if not path.exists():
            artifacts[key] = None
            continue
        if path.suffix == ".json":
            artifacts[key] = load_json_cached(path_str, mtimes.get(key))
        elif path.suffix in {".csv", ".parquet"}:
            if path.suffix == ".parquet":
                try:
                    artifacts[key] = pd.read_parquet(path)
                except Exception:
                    artifacts[key] = None
            else:
                artifacts[key] = load_csv_cached(path_str, mtimes.get(key))
        else:
            artifacts[key] = None
    return artifacts


@st.cache_data(show_spinner=False)
def compute_performance_cached(run_id: str, returns_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    _ = run_id
    return performance_data(returns_df)


@st.cache_data(show_spinner=False)
def simulate_portfolio_cached(
    run_id: str,
    signals_df: Optional[pd.DataFrame],
    returns_df: Optional[pd.DataFrame],
    prices_df: Optional[pd.DataFrame],
    horizon: str,
    alpha_threshold: float,
) -> tuple[pd.Series, list[Diagnostic], Optional[pd.DataFrame]]:
    _ = run_id
    outcome = simulate_portfolio_from_signals(
        signals_df,
        returns_df,
        prices_df,
        horizon=horizon,
        alpha_threshold=alpha_threshold,
    )
    return outcome.weights, outcome.diagnostics, outcome.aligned_returns

def expected_return_col(frame: pd.DataFrame) -> Optional[str]:
    return next((c for c in ("expected_return_horizon", "predicted_return", "expected_return", "predicted_return_1d") if c in frame.columns), None)

def build_top10_table(
    predictions_df: Optional[pd.DataFrame],
    signals_df: Optional[pd.DataFrame],
) -> tuple[Optional[pd.DataFrame], Optional[str], bool, list[Diagnostic]]:
    diagnostics: list[Diagnostic] = []
    source = "predictions" if predictions_df is not None else "signals"
    frame = predictions_df if predictions_df is not None else signals_df
    if frame is None:
        diagnostics.append(
            make_diagnostic(
                MISSING_REQUIRED_ARTIFACT,
                detected={"artifact": source},
                suggestion="Load predictions or signals artifacts before ranking.",
                message="Ranking input missing.",
            )
        )
        return None, source, False, diagnostics
    if frame.empty or "ticker" not in frame.columns:
        diagnostics.append(
            make_diagnostic(
                EMPTY_AFTER_DROPNA,
                detected={"artifact": source, "rows": 0, "columns": list(frame.columns)},
                suggestion="Verify the prediction/signal files contain ticker rows.",
                message="Ranking input empty.",
            )
        )
        return None, source, False, diagnostics

    return_col = expected_return_col(frame)
    score_col = "score" if "score" in frame.columns else None
    prob_col = "probability" if "probability" in frame.columns else None
    for col in (return_col, score_col, prob_col):
        if col:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["_rank"] = frame[return_col] if return_col else frame.get(score_col, 0)
    best = frame.sort_values("_rank", ascending=False).groupby("ticker", as_index=False).head(1)
    sort_cols = [c for c in (return_col, prob_col, score_col) if c]
    if sort_cols:
        best = best.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    top = best.head(10).reset_index(drop=True)

    if "signal" in top.columns:
        signals = top["signal"].astype(str).str.upper()
        top["action"] = signals.where(signals == "BUY", "HOLD")
        if not (signals == "BUY").any():
            top["action"] = "HOLD (No BUYs)"
    else:
        top["action"] = "HOLD"

    horizon_map = {"1d": "Hold 1 day", "5d": "Hold ~1 week", "20d": "Hold ~1 month"}
    top["hold_time"] = top.get("horizon", "").astype(str).map(horizon_map).fillna("Hold period unknown")
    confidence = top[prob_col].apply(lambda v: f"{float(v) * 100:.1f}%" if pd.notna(v) else "N/A") if prob_col else "N/A"
    expected = top[return_col].apply(lambda v: f"{v * 100:+.2f}%" if pd.notna(v) else "N/A") if return_col else "N/A"
    as_of = top["date"].astype(str) if "date" in top.columns else "N/A"

    table = pd.DataFrame({
        "Rank": range(1, len(top) + 1),
        "Ticker": top["ticker"].astype(str),
        "Action": top["action"].astype(str),
        "Confidence": confidence,
        "Expected Return": expected,
        "Hold Time": top["hold_time"].astype(str),
        "As-of Date": as_of
    })
    return table, source, (top["action"] == "BUY").any(), diagnostics

def performance_data(returns_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if returns_df is None or returns_df.empty or "return" not in returns_df.columns:
        return None
    frame = returns_df.copy(); frame["return"] = pd.to_numeric(frame["return"], errors="coerce").fillna(0)
    portfolio = (1 + frame["return"]).cumprod() - 1
    benchmark_col = next((c for c in ("benchmark_return", "benchmark") if c in frame.columns), None)
    benchmark = (1 + pd.to_numeric(frame[benchmark_col], errors="coerce").fillna(0)).cumprod() - 1 if benchmark_col else pd.Series(0, index=portfolio.index)
    return pd.DataFrame({"Portfolio": portfolio, "Benchmark": benchmark})

def weights_series(weights_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if weights_df is None or weights_df.empty:
        return None
    frame = weights_df.tail(1).drop(columns=[c for c in ("date", "schema_version") if c in weights_df.columns])
    if frame.empty:
        return None
    series = frame.iloc[0].apply(pd.to_numeric, errors="coerce").dropna()
    return series[series != 0].sort_values(ascending=False)

def artifact_diagnostics(artifact_paths: dict[str, Path], artifacts: dict[str, object] | None) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if artifacts is None:
        diagnostics.append(
            make_diagnostic(
                MISSING_REQUIRED_ARTIFACT,
                detected={"artifact": "all"},
                suggestion="Use 'Load artifacts' to pull the latest run outputs.",
                message="Artifacts have not been loaded for this run.",
            )
        )
        return diagnostics
    for key, path in artifact_paths.items():
        if not path or not path.exists():
            diagnostics.append(
                make_diagnostic(
                    MISSING_REQUIRED_ARTIFACT,
                    detected={"artifact": key, "path": str(path) if path else "missing"},
                    suggestion="Re-run the pipeline or select a run with this artifact.",
                    message=f"Missing artifact: {key}.",
                )
            )
            continue
        value = artifacts.get(key)
        if isinstance(value, pd.DataFrame) and value.empty:
            diagnostics.append(
                make_diagnostic(
                    EMPTY_AFTER_DROPNA,
                    detected={"artifact": key, "rows": 0, "columns": list(value.columns)},
                    suggestion="Verify the artifact contains rows after preprocessing.",
                    message=f"{key} loaded but empty.",
                )
            )
    return diagnostics


run_context = render_run_selector()
run_id = run_context["run_id"]
artifact_paths = resolve_artifact_paths(run_context["run_dir"], run_context["manifest_path"])
artifact_paths_str = {key: str(path) if path else "" for key, path in artifact_paths.items()}
artifact_mtimes = {key: _artifact_mtime(path) for key, path in artifact_paths.items()}

with st.sidebar:
    st.markdown("### Compute")
    load_artifacts_clicked = st.button("Load artifacts", use_container_width=True)
    compute_metrics_clicked = st.button("Compute metrics", use_container_width=True)
    simulate_clicked = st.button("Simulate portfolio", use_container_width=True)
    explain_clicked = st.button("Compute explainability", use_container_width=True)

if load_artifacts_clicked:
    st.session_state[_cache_key("artifacts", run_id)] = load_artifacts_cached(
        run_id, artifact_paths_str, artifact_mtimes
    )
    logger.info("Loaded artifacts for run %s", run_id)

if compute_metrics_clicked:
    artifacts = st.session_state.get(_cache_key("artifacts", run_id))
    returns_df = artifacts.get("returns") if artifacts else None
    st.session_state[_cache_key("metrics", run_id)] = compute_performance_cached(run_id, returns_df)
    logger.info("Computed metrics for run %s", run_id)

if explain_clicked:
    model_path = ROOT_DIR / "models" / "promoted" / "model.txt"
    st.session_state[_cache_key("explain", run_id)] = load_feature_importance_cached(
        run_id, str(model_path), _artifact_mtime(model_path)
    )
    logger.info("Computed explainability for run %s", run_id)

artifacts = st.session_state.get(_cache_key("artifacts", run_id))
metrics_data = st.session_state.get(_cache_key("metrics", run_id))
explain_df = st.session_state.get(_cache_key("explain", run_id))
metrics_json_path = artifact_paths.get("metrics")
metrics = artifacts.get("metrics") if artifacts else None
regime = (metrics or {}).get("regime")
predictions_df = artifacts.get("predictions") if artifacts else None
signals_df = artifacts.get("signals") if artifacts else None

return_dispersion = None
probability_dispersion = None
if predictions_df is not None and not predictions_df.empty:
    return_col = expected_return_col(predictions_df)
    if return_col:
        return_dispersion = pd.to_numeric(predictions_df[return_col], errors="coerce").std()
    if "probability" in predictions_df.columns:
        probability_dispersion = pd.to_numeric(predictions_df["probability"], errors="coerce").std()
header = st.columns([4, 1])
with header[0]:
    st.markdown("<div class='header-title'>SpectraQuant-AI Dashboard</div>", unsafe_allow_html=True)
with header[1]:
    badge_color = "var(--green)" if regime == "RISK_ON" else "var(--red)" if regime == "RISK_OFF" else "#64748b"
    st.markdown(f"<div style='text-align:right'>Current Regime:&nbsp;<span class='badge' style='background:{badge_color}'>{regime or 'UNKNOWN'}</span><span class='toggle-pill'></span></div>", unsafe_allow_html=True)
render_diagnostics(artifact_diagnostics(artifact_paths, artifacts))
if "running" not in st.session_state:
    st.session_state.running = False
center = st.columns([1, 2, 1])
with center[1]:
    st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
    refresh_clicked = st.button("🔄 Refresh – Run Full Pipeline", type="primary", use_container_width=True, disabled=st.session_state.running)
    st.markdown("</div>", unsafe_allow_html=True)
status_expander = st.expander("Pipeline Status", expanded=False)
output_expander = st.expander("Pipeline Logs", expanded=False)
if refresh_clicked and not st.session_state.running:
    st.session_state.running = True
    env = {**os.environ, "PYTHONPATH": str(ROOT_DIR / "src")}
    for step in PIPELINE_STEPS:
        with status_expander:
            st.write(f"Running: {step}")
        result = subprocess.run([sys.executable, "-m", "spectraquant.cli.main", step], env=env, capture_output=True, text=True)
        with output_expander:
            st.code(result.stdout or "", language="text"); st.code(result.stderr or "", language="text")
        if result.returncode != 0:
            with status_expander:
                st.error(f"Failed at {step}")
            st.session_state.running = False
            break
        with status_expander:
            st.success(f"Completed: {step}")
    else:
        st.session_state.running = False
        st.success(f"Pipeline completed at {fmt_ts(time.time())}")
        rerun()

st.markdown("<div class='card'>", unsafe_allow_html=True)
row = st.columns([3, 1])
with row[0]:
    st.subheader("Top 10 Buy Now Opportunities")
with row[1]:
    top10_table, top10_source, has_buy, top10_diags = build_top10_table(predictions_df, signals_df)
    if top10_table is not None and not top10_table.empty:
        buffer = io.BytesIO(); top10_table.to_excel(buffer, index=False, sheet_name="Top10")
        st.download_button("Download Excel", data=buffer.getvalue(), file_name="top_10_buy_now.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
if top10_table is None or top10_table.empty:
    render_diagnostics(top10_diags)
else:
    def style_action(val: str) -> str:
        return "background-color:#22c55e;color:white;font-weight:700" if "BUY" in str(val) else "background-color:#94a3b8;color:white;font-weight:700"
    styled = top10_table.style.applymap(style_action, subset=["Action"])
    st.dataframe(styled, width="stretch", height=320)
    if top10_source:
        st.caption(f"Source: {top10_source}")
st.markdown("</div>", unsafe_allow_html=True)

predictions_path = artifact_paths.get("predictions")
signals_path = artifact_paths.get("signals")
weights_path = artifact_paths.get("weights")
returns_path = artifact_paths.get("returns")
weights_df = artifacts.get("weights") if artifacts else None
returns_df = artifacts.get("returns") if artifacts else None
last_files = [p for p in (signals_path, weights_path, metrics_json_path, returns_path) if p and p.exists()]
last_run = fmt_ts(max(p.stat().st_mtime for p in last_files)) if last_files else None
st.caption(f"Last pipeline run: {last_run or 'No pipeline run detected yet.'}")
last_pred_timestamp = None
if predictions_path and predictions_path.exists():
    last_pred_timestamp = datetime.fromtimestamp(predictions_path.stat().st_mtime)
elif signals_path and signals_path.exists():
    last_pred_timestamp = datetime.fromtimestamp(signals_path.stat().st_mtime)
last_pred_label = last_pred_timestamp.strftime("%Y-%m-%d %H:%M") if last_pred_timestamp else "N/A"

pipeline_cfg = get_config()
active_tickers = pipeline_cfg.get("universe", {}).get("tickers", [])
if signals_df is not None and "ticker" in signals_df.columns:
    active_tickers = sorted(set(signals_df["ticker"].dropna().astype(str)))
available_horizons = (
    sorted(set(signals_df["horizon"].dropna().astype(str)))
    if signals_df is not None and "horizon" in signals_df.columns
    else []
)

perf = metrics_data
weights = weights_series(weights_df)
upper = st.columns(3)
with upper[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True); st.subheader("Portfolio Overview")
    total_return = (metrics or {}).get("total_return") or (metrics or {}).get("cumulative_return")
    volatility = (metrics or {}).get("volatility")
    st.write(f"Total Return: {total_return * 100:.2f}%" if total_return is not None else "Total Return: N/A")
    st.write(f"Volatility: {volatility * 100:.2f}%" if volatility is not None else "Volatility: N/A")
    st.line_chart(perf, height=200) if perf is not None else st.info("Compute metrics to view performance data.")
    st.markdown("</div>", unsafe_allow_html=True)
with upper[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True); st.subheader("Signal Strength Heatmap")
    signals_df = artifacts.get("signals") if artifacts else None
    timestamp_label = last_pred_label
    selected_horizon = "unknown"
    if signals_df is None or signals_df.empty:
        render_diagnostics(
            make_diagnostic(
                MISSING_REQUIRED_ARTIFACT,
                detected={"artifact": "signals"},
                suggestion="Load signals via the artifact loader or run the pipeline.",
                message="Signals not loaded.",
            )
        )
    else:
        if "date" in signals_df.columns:
            last_signal_ts = pd.to_datetime(signals_df["date"], errors="coerce").dropna()
            if not last_signal_ts.empty:
                timestamp_label = last_signal_ts.max().strftime("%Y-%m-%d %H:%M")
        if "horizon" in signals_df.columns:
            horizon_mode = signals_df["horizon"].dropna().astype(str)
            if not horizon_mode.empty:
                selected_horizon = horizon_mode.mode().iloc[0]
    st.markdown(
        f"""
        <div class="sq-card soft" style="text-align:center;">
            <div class="sq-label">Last prediction timestamp</div>
            <div class="sq-stat" style="color: var(--sq-warn); font-size:18px;">{timestamp_label}</div>
            <div class="sq-muted">Horizon · {selected_horizon}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if return_dispersion is not None and return_dispersion < 1e-6:
    st.error("Predictions appear degenerate; re-run predict / check QA.")
elif probability_dispersion is not None and probability_dispersion < 1e-6:
    st.error("Prediction probabilities appear degenerate; re-run predict / check QA.")

st.markdown(
    """
    <div class="sq-card" style="margin-top:14px;">
        <div class="sq-label">Run Pipeline</div>
        <div class="sq-muted">Execute the pipeline to refresh predictions and metadata:</div>
    </div>
    """,
    unsafe_allow_html=True,
)
pipeline_cfg = get_config()
use_sentiment_default = bool(pipeline_cfg.get("sentiment", {}).get("enabled", False))
run_use_sentiment = st.checkbox("Use sentiment features", value=use_sentiment_default)
universe_override = st.text_input("Universe override (comma-separated)", value="")
run_cmd = ["python", "-m", "spectraquant.cli.main", "refresh"]
if run_use_sentiment:
    run_cmd.append("--use-sentiment")
if universe_override:
    run_cmd.extend(["--universe", universe_override])
if st.button("Refresh Pipeline"):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root_dir}/src"
    with st.spinner("Running pipeline refresh..."):
        result = subprocess.run(run_cmd, capture_output=True, text=True, env=env, cwd=str(root_dir))
    if result.returncode == 0:
        st.success("Pipeline refresh completed.")
    else:
        st.error("Pipeline refresh failed. Review logs below.")
    if result.stdout:
        st.text_area("Pipeline output", value=result.stdout, height=200)
    if result.stderr:
        st.text_area("Pipeline errors", value=result.stderr, height=200)
st.code(
    "\n".join(
        [
            "PYTHONPATH=src python -m spectraquant.cli.main refresh",
            "PYTHONPATH=src python -m spectraquant.cli.main refresh --use-sentiment",
            "PYTHONPATH=src python -m spectraquant.cli.main refresh --use-sentiment --universe \"nifty50,ftse100\"",
        ]
    ),
    language="bash",
)

mid_left, mid_right = st.columns([1.2, 1.8])
with mid_left:
    if signals_df is None or signals_df.empty:
        st.info("Price history not available yet.")
    else:
        score_col = "score" if "score" in signals_df.columns else "signal" if "signal" in signals_df.columns else None
        pivot_col = "horizon" if "horizon" in signals_df.columns else "date" if "date" in signals_df.columns else None
        if score_col and pivot_col:
            heat = signals_df.pivot_table(index="ticker", columns=pivot_col, values=score_col, aggfunc="mean")
            st.dataframe(heat.style.background_gradient(cmap="RdYlGn"), width="stretch", height=220)
        elif not score_col:
            st.info("No signal strength column available.")
        else:
            st.info("Price history not available yet.")

st.markdown("### Feature Importance")
importance_df = explain_df if isinstance(explain_df, pd.DataFrame) else None
if importance_df is None:
    st.info("Compute explainability to view feature importance.")
elif importance_df.empty:
    render_diagnostics(
        make_diagnostic(
            EMPTY_AFTER_DROPNA,
            detected={"artifact": "feature_importance", "rows": 0},
            suggestion="Ensure a LightGBM model.txt exists in models/promoted or models/latest.",
            message="Feature importance unavailable.",
        )
    )
else:
    sort_choice = st.selectbox("Sort features by", ["importance", "alphabetical"], index=0)
    display_df = importance_df.copy()
    if sort_choice == "alphabetical":
        display_df = display_df.sort_values("feature", ascending=True)
    else:
        display_df = display_df.sort_values("importance", ascending=False)
    display_df = display_df.reset_index(drop=True)
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "importance": st.column_config.NumberColumn(format="%.4f"),
            "importance_pct": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download feature importance CSV",
        data=csv_data,
        file_name="feature_importance.csv",
        mime="text/csv",
    )

st.markdown("### Investment Simulator")
sim_left, sim_right = st.columns([1.1, 1.5])
sim_horizons = available_horizons or ["1d"]
default_index = sim_horizons.index("1d") if "1d" in sim_horizons else 0
with sim_left:
    st.markdown('<div class="sq-card">', unsafe_allow_html=True)
    if active_tickers:
        sim_tickers = st.multiselect("Tickers", active_tickers, default=[active_tickers[0]])
    else:
        sim_tickers = st.multiselect("Tickers", ["Not available yet"], default=["Not available yet"], disabled=True)
    sim_horizon = st.selectbox("Horizon", sim_horizons, index=default_index)
    alpha_threshold = st.number_input("Signal score threshold", min_value=0.0, value=0.0, step=1.0)
    sim_amount = st.number_input("Investment amount", min_value=0.0, value=10000.0, step=1000.0)
    holding_days = int(st.number_input("Holding period (days)", min_value=1, value=30, step=1))
    sip_mode = st.selectbox("Contribution mode", ["None", "Monthly SIP"], index=0)
    sip_amount = 0.0
    if sip_mode == "Monthly SIP":
        sip_amount = st.number_input("Monthly SIP amount", min_value=0.0, value=1000.0, step=100.0)
    apply_fees = st.toggle("Apply fees/slippage", value=True)
    if apply_fees:
        transaction_cost_pct = st.number_input(
            "One-way transaction cost (%)", min_value=0.0, value=0.10, step=0.05
        )
        management_fee_pct = st.number_input(
            "Annual management fee (%)", min_value=0.0, value=0.50, step=0.05
        )
    else:
        transaction_cost_pct = 0.0
        management_fee_pct = 0.0
    st.markdown("</div>", unsafe_allow_html=True)

sim_cache_key = _cache_key(
    "sim",
    run_id,
    {"horizon": sim_horizon, "alpha_threshold": alpha_threshold, "tickers": sim_tickers},
)

if simulate_clicked:
    artifacts = st.session_state.get(_cache_key("artifacts", run_id))
    signals_df = artifacts.get("signals") if artifacts else None
    returns_df = artifacts.get("returns") if artifacts else None
    prices_df = (
        load_latest_prices(sim_tickers[0])
        if sim_tickers and sim_tickers[0] and sim_tickers[0] != "Not available yet"
        else None
    )
    weights, diagnostics, aligned_returns = simulate_portfolio_cached(
        run_id,
        signals_df,
        returns_df,
        prices_df,
        horizon=sim_horizon,
        alpha_threshold=alpha_threshold,
    )
    st.session_state[sim_cache_key] = {
        "weights": weights,
        "diagnostics": diagnostics,
        "aligned_returns": aligned_returns,
    }
    logger.info("Simulated portfolio for run %s", run_id)

with sim_right:
    st.markdown('<div class="sq-card">', unsafe_allow_html=True)
    sim_state = st.session_state.get(sim_cache_key, {})
    sim_weights = sim_state.get("weights") if isinstance(sim_state, dict) else None
    sim_diags = sim_state.get("diagnostics") if isinstance(sim_state, dict) else None
    if sim_diags:
        render_diagnostics(sim_diags)
    if sim_weights is None or sim_weights.empty:
        st.info("Run the portfolio simulation to view weights.")
    else:
        weights_df_view = sim_weights.reset_index()
        weights_df_view.columns = ["Ticker", "Weight"]
        st.dataframe(weights_df_view, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with upper[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True); st.subheader("Portfolio Allocation")
    if weights is None or weights.empty:
        render_diagnostics(
            make_diagnostic(
                MISSING_REQUIRED_ARTIFACT,
                detected={"artifact": "weights"},
                suggestion="Load portfolio weights via the artifact loader or run the pipeline.",
                message="Portfolio weights not loaded.",
            )
        )
    else:
        weights_frame = weights.reset_index(); weights_frame.columns = ["Ticker", "Weight"]
        chart = alt.Chart(weights_frame).mark_arc(innerRadius=40).encode(theta="Weight", color="Ticker", tooltip=["Ticker", "Weight"])
        st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

lower = st.columns([2, 1])
with lower[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True); st.subheader("Performance Chart")
    st.line_chart(perf, height=220) if perf is not None else st.info("Compute metrics to view performance data.")
    st.markdown("</div>", unsafe_allow_html=True)
with lower[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True); st.subheader("Recent Alerts")
    alerts = [f"Pipeline completed: {last_run}" if last_run else "No pipeline run detected yet."]
    if not has_buy:
        alerts.append("No BUY signals passed threshold")
    if weights_df is not None and not weights_df.empty:
        alerts.append("Portfolio rebalanced successfully")
    for alert in alerts:
        st.write(f"• {alert}")
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Debug", expanded=False):
    st.write(f"Repo root: {ROOT_DIR}"); st.write(f"CWD: {Path.cwd()}")
    st.write(f"Selected run: {run_id}")
    st.write(f"Predictions: {predictions_path if predictions_path and predictions_path.exists() else 'Missing'}")
    st.write(f"Signals: {signals_path if signals_path and signals_path.exists() else 'Missing'}")
    st.write(f"Portfolio weights: {weights_path if weights_path and weights_path.exists() else 'Missing'}")
    st.write(f"Portfolio returns: {returns_path if returns_path and returns_path.exists() else 'Missing'}")
    st.write(f"Portfolio metrics: {metrics_json_path if metrics_json_path and metrics_json_path.exists() else 'Missing'}")

with st.expander("Debug logs (this session)", expanded=False):
    log_lines = list(get_recent_logs(200))
    if log_lines:
        st.code("\n".join(log_lines), language="text")
    else:
        st.info("No logs captured yet.")
