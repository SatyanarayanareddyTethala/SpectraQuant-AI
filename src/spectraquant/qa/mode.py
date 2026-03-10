"""Helpers for configuring QA gate behavior."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


@dataclass(frozen=True)
class GateMode:
    test_mode: bool
    force_pass: bool
    mode: Literal["strict", "lenient"]


def resolve_gate_mode(cfg: Dict) -> GateMode:
    raw_test_mode = cfg.get("test_mode", False)
    if isinstance(raw_test_mode, dict):
        test_mode = bool(raw_test_mode.get("enabled", False))
    else:
        test_mode = bool(raw_test_mode)
    qa_cfg = cfg.get("qa", {}) if isinstance(cfg, dict) else {}
    force_pass = bool(qa_cfg.get("force_pass_tests", False))
    mode_str = str(qa_cfg.get("mode", "strict")).lower()
    mode: Literal["strict", "lenient"] = "lenient" if mode_str == "lenient" else "strict"
    return GateMode(test_mode=test_mode, force_pass=force_pass, mode=mode)
