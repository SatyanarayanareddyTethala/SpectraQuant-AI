from __future__ import annotations

import streamlit as st


def rerun() -> None:
    """Backward-safe Streamlit rerun helper."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
