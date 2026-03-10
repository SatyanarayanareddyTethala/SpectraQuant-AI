#!/usr/bin/env python3
"""Inspect resolved universe tickers."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectraquant.config import DEFAULT_CONFIG  # noqa: E402
from spectraquant.universe import resolve_universe  # noqa: E402


def _merge_defaults(cfg: dict) -> dict:
    merged = dict(cfg)
    for key, value in DEFAULT_CONFIG.items():
        if key not in merged:
            merged[key] = value
        elif isinstance(merged[key], dict) and isinstance(value, dict):
            nested = value.copy()
            nested.update(merged[key])
            merged[key] = nested
    return merged


def main() -> int:
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    else:
        loaded = {}

    cfg = _merge_defaults(loaded if isinstance(loaded, dict) else {})
    tickers, meta = resolve_universe(cfg)
    print("Universe meta:", meta)
    print("Resolved tickers (%s): %s" % (len(tickers), tickers))

    if not tickers:
        print("Universe resolved empty; configure tickers or CSV files.")
        return 1
    if meta.get("invalid_suffix_count"):
        print("Invalid tickers detected; fix suffixes before running.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
