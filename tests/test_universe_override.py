from __future__ import annotations

from pathlib import Path

import pytest

from spectraquant import config as config_module
from spectraquant.universe import parse_universe_override


FIXTURE_CONFIG = Path(__file__).resolve().parent / "fixtures" / "config.yaml"


def test_universe_override_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRAQUANT_UNIVERSE", "RELIANCE.NS,TCS.NS")
    monkeypatch.setattr(config_module, "CONFIG_PATH", FIXTURE_CONFIG)
    cfg = config_module.get_config()
    assert cfg["universe"]["tickers"] == ["RELIANCE.NS", "TCS.NS"]
    assert cfg["universe"]["selected_sets"] == []


def test_parse_universe_override_rejects_mixed_tokens() -> None:
    mode, tokens = parse_universe_override("RELIANCE.NS,TCS.NS", {})
    assert mode == "tickers"
    assert tokens == ["RELIANCE.NS", "TCS.NS"]
    with pytest.raises(ValueError, match="mixes tickers and set names"):
        parse_universe_override("RELIANCE.NS,nifty50", {})
