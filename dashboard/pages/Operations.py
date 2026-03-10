import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from spectraquant.config import get_config
from dashboard.utils.diagnostics import MISSING_REQUIRED_ARTIFACT, make_diagnostic, render_diagnostics
from spectraquant.universe.loader import (
    MISSING_UNIVERSE_FILE,
    NO_VALID_TICKERS_AFTER_CLEAN,
    UNIVERSE_SCHEMA_INVALID,
)
from dashboard.utils.logging import get_recent_logs
from dashboard.utils.manifest import discover_runs, load_run_manifest, resolve_manifest_paths
from dashboard.utils.streamlit_compat import rerun
from spectraquant.universe.loader import load_nse_universe
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

if "run_id" not in st.session_state:
    st.session_state.run_id = "latest"


def _run_option_label(run_id: str, has_manifest: bool) -> str:
    marker = "✅" if has_manifest else "—"
    return f"{run_id} {marker}"


def render_run_selector() -> dict[str, Optional[Path]]:
    runs = discover_runs(BASE_DIR)
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
        st.session_state.run_id = selected
        rerun()
    run_info = run_lookup[selected]
    return {"run_id": run_info.run_id, "run_dir": run_info.run_dir, "manifest_path": run_info.manifest_path}

def format_size(num_bytes: int) -> str:
    if num_bytes is None:
        return "-"
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def latest_file(
    path: Path,
    pattern: Optional[str] = None,
    exclude_prefixes: tuple[str, ...] = ("__pycache__",),
) -> tuple[Optional[Path], str]:
    if path.is_file():
        return path, "OK"
    if path.is_dir():
        glob_pattern = pattern or "*"
        candidates = [
            p
            for p in path.glob(glob_pattern)
            if p.is_file() and not any(p.name.startswith(prefix) for prefix in exclude_prefixes)
        ]
        if not candidates:
            return None, "Missing"
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest, "OK"
    return None, "Missing"


def artifact_rows() -> pd.DataFrame:
    artifacts = [
        ("Raw prices", DATA_DIR / "prices", "*.csv"),
        ("Processed meta", DATA_DIR / "processed" / "meta.csv", None),
        ("Processed features", DATA_DIR / "processed", "X.npy"),
        ("Models", MODELS_DIR, "*"),
        ("Training metadata", MODELS_DIR / "training_metadata.json", None),
        ("Predictions", REPORTS_DIR / "predictions", "*.csv"),
        ("Signals", REPORTS_DIR / "signals", "*.csv"),
        ("Portfolio returns", REPORTS_DIR / "portfolio", "*returns*.csv"),
        ("Portfolio weights", REPORTS_DIR / "portfolio", "*weights*.csv"),
        ("Portfolio metrics", REPORTS_DIR / "portfolio", "*.json"),
    ]

    records = []
    for name, path, pattern in artifacts:
        latest, status = latest_file(path, pattern)
        if latest:
            stat = latest.stat()
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size = format_size(stat.st_size)
            file_name = latest.name
            file_path = latest.as_posix()
        else:
            modified = "-"
            size = "-"
            file_name = "-"
            file_path = "-"
        records.append(
            {
                "Artifact": name,
                "Latest": file_name,
                "Modified": modified,
                "Size": size,
                "Status": status,
                "Path": file_path,
            }
        )
    return pd.DataFrame(records)


def load_training_metadata() -> Optional[dict]:
    metadata_path = MODELS_DIR / "training_metadata.json"
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            return None
    return None


def retrain_due(metadata: Optional[dict], interval_days: int) -> tuple[bool, Optional[datetime], Optional[datetime]]:
    if not metadata:
        next_due = datetime.utcnow()
        return True, None, next_due
    last_trained_str = metadata.get("last_trained")
    if not last_trained_str:
        return True, None, datetime.utcnow()
    try:
        last_trained = datetime.fromisoformat(last_trained_str)
    except ValueError:
        return True, None, datetime.utcnow()
    next_due = last_trained + timedelta(days=interval_days)
    return datetime.utcnow() >= next_due, last_trained, next_due


def preview_csv(path: Optional[Path], title: str, missing_command: Optional[str] = None) -> None:
    with st.expander(title, expanded=False):
        if not path or not path.exists():
            if missing_command:
                st.info("File not available yet. Generate it with:")
                st.code(missing_command, language="bash")
            else:
                st.info("File not available yet.")
            return
        try:
            df = pd.read_csv(path)
            st.markdown(f"**Path:** {path}")
            st.dataframe(df.head(50))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not preview {path.name}: {exc}")


def render_environment(cfg: dict) -> None:
    st.subheader("Environment Defaults")
    tickers = cfg.get("universe", {}).get("tickers", [])
    markets = sorted({"UK" if t.endswith(".L") else "India" for t in tickers if t.endswith((".L", ".NS"))})
    synthetic = cfg.get("data", {}).get("synthetic", False)
    source = cfg.get("data", {}).get("source", "")
    st.info(
        "Real UK/India defaults active; synthetic data is OFF" if not synthetic else "Synthetic data enabled"
    )
    st.markdown(
        f"**Active tickers ({len(tickers)}):** {', '.join(tickers[:10])}{' ...' if len(tickers) > 10 else ''}"
    )
    st.markdown(f"**Detected markets:** {', '.join(markets) if markets else 'N/A'}")
    st.markdown(f"**Synthetic mode:** {'OFF' if not synthetic else 'ON'}")
    st.markdown(f"**Data source:** {source or 'unspecified'}")


def render_latest_artifact(
    title: str,
    path: Path,
    pattern: str,
    missing_command: str,
) -> Optional[Path]:
    latest, status = latest_file(path, pattern)
    st.subheader(title)
    if status == "Missing" or not latest:
        st.warning("Missing artifact.")
        st.markdown("Generate it with:")
        st.code(missing_command, language="bash")
        return None
    stat = latest.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"**Latest file:** `{latest.name}`")
    st.markdown(f"**Last modified:** {modified}")
    st.markdown(f"**Path:** `{latest}`")
    return latest


st.title("Operations")
st.caption("System status, artifacts, and retraining cadence")
st.warning("Analysis-only. No execution.")

run_context = render_run_selector()
run_id = run_context["run_id"]
run_dir = run_context["run_dir"]
manifest = load_run_manifest(run_dir) if run_dir else None
manifest_paths = resolve_manifest_paths(manifest, run_dir) if manifest else {}
if run_id != "latest" and manifest is None:
    render_diagnostics(
        make_diagnostic(
            MISSING_REQUIRED_ARTIFACT,
            detected={"artifact": "manifest.json", "run_id": run_id},
            suggestion="Select a run with manifest.json or regenerate artifacts with manifest output.",
            message="Manifest missing for selected run.",
        )
    )

with st.expander("Run manifest", expanded=False):
    if manifest:
        st.json(manifest)
    else:
        st.info("No manifest available for this run.")

if manifest_paths:
    st.subheader("Manifest paths")
    st.dataframe(
        pd.DataFrame(
            [{"Artifact": key, "Path": value.as_posix()} for key, value in manifest_paths.items()]
        ),
        use_container_width=True,
    )

config = get_config()
render_environment(config)

st.subheader("India (NSE) Universe")
india_cfg = (config.get("universe") or {}).get("india", {})
india_path = india_cfg.get("path")
if india_path:
    tickers, meta, diagnostics = load_nse_universe(
        india_path,
        symbol_column=india_cfg.get("symbol_column", "SYMBOL"),
        suffix=india_cfg.get("suffix", ".NS"),
        filter_series_eq=bool(india_cfg.get("filter_series_eq", True)),
    )
    if diagnostics:
        for diag in diagnostics:
            suggestion = "Verify the NSE universe CSV path and schema."
            if diag.code == NO_VALID_TICKERS_AFTER_CLEAN:
                suggestion = "Ensure the universe CSV contains valid NSE symbols."
            render_diagnostics(
                make_diagnostic(
                    diag.code,
                    detected=diag.details,
                    suggestion=suggestion,
                    message=diag.message,
                )
            )
    st.markdown(f"**Raw symbols:** {meta.get('raw_count', 0)}")
    st.markdown(f"**EQ filtered:** {meta.get('eq_count', 0)}")
    st.markdown(f"**Yahoo tickers:** {len(tickers)}")
    st.markdown(f"**Dropped:** {meta.get('dropped_count', 0)}")
    dropped_reasons = meta.get("dropped_reasons") or {}
    if dropped_reasons:
        st.caption(
            "Dropped reasons — blank/nan: {blank}, duplicates: {dup}".format(
                blank=dropped_reasons.get("blank_or_nan", 0),
                dup=dropped_reasons.get("duplicates", 0),
            )
        )
else:
    st.info("India universe path not configured.")

st.subheader("Artifacts table")
df_artifacts = artifact_rows()
if not df_artifacts.empty:
    display_df = df_artifacts.copy()
    display_df["Path"] = display_df["Path"].apply(lambda p: f"file://{p}" if p not in {"-", None} else "-")
    st.dataframe(
        display_df[["Artifact", "Status", "Modified", "Path", "Latest", "Size"]],
        use_container_width=True,
        column_config={"Path": st.column_config.LinkColumn("Path")},
    )
else:
    st.info("No artifacts found yet.")

st.subheader("Training Metadata")
metadata = load_training_metadata()
if metadata:
    st.json(metadata)
else:
    st.info("Training metadata not found yet.")

interval = config.get("mlops", {}).get("retrain_interval_days", 7)
due, last_trained_dt, next_due_dt = retrain_due(metadata, interval)
status_text = "YES" if due else "NO"
st.metric("Retrain due", status_text)
if last_trained_dt:
    st.markdown(f"**Last trained:** {last_trained_dt.isoformat()}")
if next_due_dt:
    st.markdown(f"**Next due:** {next_due_dt.date().isoformat()}")

st.subheader("Pipeline outputs")
latest_sig = render_latest_artifact(
    "Latest signals file",
    REPORTS_DIR / "signals",
    "*.csv",
    "python -m src.spectraquant.cli.main signals",
)
latest_weights = render_latest_artifact(
    "Latest portfolio weights",
    REPORTS_DIR / "portfolio",
    "*weights*.csv",
    "python -m src.spectraquant.cli.main portfolio",
)
latest_returns = render_latest_artifact(
    "Latest portfolio returns",
    REPORTS_DIR / "portfolio",
    "*returns*.csv",
    "python -m src.spectraquant.cli.main portfolio",
)

st.subheader("Quick Previews")
latest_pred, _ = latest_file(REPORTS_DIR / "predictions", "*.csv")
preview_csv(latest_pred, "Latest predictions", "python -m src.spectraquant.cli.main predict")
preview_csv(latest_sig, "Latest signals", "python -m src.spectraquant.cli.main signals")
preview_csv(latest_returns, "Latest portfolio returns", "python -m src.spectraquant.cli.main portfolio")
preview_csv(latest_weights, "Latest portfolio weights", "python -m src.spectraquant.cli.main portfolio")

st.subheader("Actions")
col1, col2 = st.columns(2)
with col1:
    if st.button("Copy CLI commands"):
        commands = "\n".join(
            [
                "python -m src.spectraquant.cli.main health-check",
                "python -m src.spectraquant.cli.main download",
                "python -m src.spectraquant.cli.main build-dataset",
                "python -m src.spectraquant.cli.main train",
                "python -m src.spectraquant.cli.main predict",
                "python -m src.spectraquant.cli.main signals",
                "python -m src.spectraquant.cli.main portfolio",
                "python -m src.spectraquant.cli.main retrain",
            ]
        )
        st.code(commands, language="bash")
with col2:
    if st.button("Refresh status"):
        rerun()

with st.expander("Debug logs (this session)", expanded=False):
    log_lines = list(get_recent_logs(200))
    if log_lines:
        st.code("\n".join(log_lines), language="text")
    else:
        st.info("No logs captured yet.")

st.caption("Navigation: use the sidebar to switch pages.")
