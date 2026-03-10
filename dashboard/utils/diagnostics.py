from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DATE_INDEX_NOT_DATETIME = "DATE_INDEX_NOT_DATETIME"
TIMEZONE_MISMATCH = "TIMEZONE_MISMATCH"
SIGNAL_RETURN_MISALIGNMENT = "SIGNAL_RETURN_MISALIGNMENT"
NO_SIGNAL_AFTER_FILTER = "NO_SIGNAL_AFTER_FILTER"
HORIZON_OUT_OF_RANGE = "HORIZON_OUT_OF_RANGE"
MISSING_REQUIRED_ARTIFACT = "MISSING_REQUIRED_ARTIFACT"
EMPTY_AFTER_DROPNA = "EMPTY_AFTER_DROPNA"

SUPPORTED_REASON_CODES = {
    DATE_INDEX_NOT_DATETIME,
    TIMEZONE_MISMATCH,
    SIGNAL_RETURN_MISALIGNMENT,
    NO_SIGNAL_AFTER_FILTER,
    HORIZON_OUT_OF_RANGE,
    MISSING_REQUIRED_ARTIFACT,
    EMPTY_AFTER_DROPNA,
}


@dataclass(frozen=True)
class Diagnostic:
    code: str
    detected: dict[str, Any]
    suggestion: str
    message: str


class DiagnosticError(Exception):
    def __init__(self, diagnostic: Diagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


def make_diagnostic(
    code: str,
    detected: dict[str, Any],
    suggestion: str,
    message: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        detected=detected,
        suggestion=suggestion,
        message=message or code.replace("_", " ").title(),
    )


def _format_detected(detected: dict[str, Any]) -> str:
    items = []
    for key, value in detected.items():
        items.append(f"{key}={value}")
    return ", ".join(items) if items else "N/A"


def render_diagnostics(diagnostics: Diagnostic | Iterable[Diagnostic] | None) -> None:
    if diagnostics is None:
        return
    if isinstance(diagnostics, Diagnostic):
        diagnostics = [diagnostics]
    diagnostics = list(diagnostics)
    if not diagnostics:
        return
    import streamlit as st

    with st.container():
        st.markdown("**Diagnostics**")
        for diag in diagnostics:
            st.markdown(f"- **{diag.code}** — {diag.message}")
            st.caption(f"Detected: {_format_detected(diag.detected)}")
            st.caption(f"Fix: {diag.suggestion}")
