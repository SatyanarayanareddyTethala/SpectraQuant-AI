#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "Running dashboard compile checks..."
python -m compileall -q dashboard

PREDICTIONS_DIR="$ROOT_DIR/reports/predictions"
shopt -s nullglob
pred_files=("$PREDICTIONS_DIR"/*.csv)
shopt -u nullglob

if [ ${#pred_files[@]} -eq 0 ]; then
  echo "No predictions CSV found in ${PREDICTIONS_DIR}."
  echo "Generate predictions with:"
  echo "  python -m src.spectraquant.cli.main predict"
else
  echo "Found predictions artifact: ${pred_files[0]}"
fi

HEADLESS_FLAG=""
if [ "${1:-}" = "--headless" ]; then
  HEADLESS_FLAG="--server.headless true"
fi

PORT="${PORT:-8501}"
echo "Starting Streamlit at http://localhost:${PORT}"
exec streamlit run dashboard/app.py --server.port "${PORT}" ${HEADLESS_FLAG}
