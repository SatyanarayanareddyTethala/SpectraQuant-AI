"""Regression test for UnboundLocalError: Path in resolve_universe().

The bug: a `from pathlib import Path` inside the `if "news" in selected_sets:`
block caused Python to treat Path as a local variable for the entire function.
When that branch was not taken (e.g. universe set is "nifty50"), accessing Path
on the else-branch raised UnboundLocalError.
"""
from __future__ import annotations

from unittest.mock import patch

from spectraquant.universe import resolve_universe


def test_resolve_universe_nifty50_does_not_raise_unbound_local_error() -> None:
    """resolve_universe with nifty50 must not raise UnboundLocalError for Path."""
    cfg = {
        "universe": {
            "selected_sets": ["nifty50"],
        }
    }
    # _fetch_universe_set tries network/disk; patch it to return a stub list
    with patch(
        "spectraquant.universe._fetch_universe_set",
        return_value=["RELIANCE.NS", "TCS.NS"],
    ):
        tickers, meta = resolve_universe(cfg)

    assert isinstance(tickers, list)
    assert len(tickers) > 0
