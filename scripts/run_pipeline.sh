#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

commands=(
  "python -m spectraquant.cli.main download"
  "python -m spectraquant.cli.main build-dataset"
  "python -m spectraquant.cli.main train"
  "python -m spectraquant.cli.main predict"
  "python -m spectraquant.cli.main signals"
  "python -m spectraquant.cli.main portfolio"
  "python -m spectraquant.cli.main health-check"
)

for cmd in "${commands[@]}"; do
  echo "[RUN] $cmd"
  eval "$cmd"
  if [[ "$cmd" == "python -m spectraquant.cli.main predict" ]]; then
    echo "[CHECK] verifying predictions artifact"
    python - <<'PY'
from pathlib import Path
import pandas as pd

pred_dir = Path("reports/predictions")
files = sorted(pred_dir.glob("predictions_*.csv"), key=lambda p: p.stat().st_mtime)
if not files:
    raise SystemExit("No predictions CSV found in reports/predictions after predict step.")
latest = files[-1]
df = pd.read_csv(latest)
required = {"expected_return_annual", "expected_return_horizon", "probability"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(
        f"Predictions file {latest} missing columns: {sorted(missing)}"
    )
print(f"[OK] {latest} contains {sorted(required)}")
PY
  fi
done
