#!/usr/bin/env python3
"""Compile and import-check SpectraQuant modules."""
from __future__ import annotations

import compileall
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectraquant.config import get_config  # noqa: E402


def _compile_dir(path: Path) -> bool:
    if not path.exists():
        return True
    return bool(compileall.compile_dir(str(path), quiet=1))


def main() -> int:
    tests_dir = ROOT / "tests"
    config_path = ROOT / "config.yaml"

    ok = True
    ok = _compile_dir(SRC_DIR) and ok
    ok = _compile_dir(tests_dir) and ok

    try:
        import spectraquant  # noqa: F401
        import spectraquant.cli.main  # noqa: F401
        import spectraquant.data.retention  # noqa: F401
        import spectraquant.portfolio.simulator  # noqa: F401
        import spectraquant.data.yf_batch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"Import failure: {exc}")
        ok = False

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle) or {}
    try:
        cfg = get_config()
        tickers = cfg.get("data", {}).get("tickers", [])
        print(f"Config tickers ({len(tickers)}): {tickers}")
        invalid = [
            str(ticker).upper()
            for ticker in tickers
            if not (
                str(ticker).upper().endswith(".L") or str(ticker).upper().endswith(".NS")
            )
        ]
        if invalid:
            print(
                "Invalid tickers detected (first 10: %s). "
                "Tickers must end with .L or .NS." % ", ".join(invalid[:10])
            )
            ok = False
    except ValueError as exc:
        print(f"Config validation failure: {exc}")
        ok = False

    if ok:
        print("Compile/import checks passed.")
        return 0
    print("Compile/import checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
