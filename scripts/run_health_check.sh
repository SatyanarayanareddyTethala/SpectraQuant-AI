#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
cd "$ROOT_DIR"

echo "[RUN] python -m src.spectraquant.cli.main health-check"
python -m src.spectraquant.cli.main health-check
