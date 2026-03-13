from __future__ import annotations

import logging

from spectraquant import config as config_module


def _minimal_cfg() -> dict:
    return {
        "data": {"synthetic": False, "tickers": ["INFY.NS", "TCS.NS"]},
        "universe": {"tickers": ["INFY.NS", "TCS.NS"], "selected_sets": []},
        "mlops": {"auto_retrain": True},
        "portfolio": {},
        "alpha": {},
    }


def _reset_init_log_guard() -> None:
    with config_module._INIT_LOG_LOCK:
        config_module._INIT_LOGGED_KEYS.clear()


def test_validate_runtime_defaults_dedupes_init_info_logs(monkeypatch, caplog):
    _reset_init_log_guard()
    monkeypatch.delenv("SPECTRAQUANT_VERBOSE", raising=False)
    cfg = _minimal_cfg()

    with caplog.at_level(logging.INFO):
        config_module.validate_runtime_defaults(cfg)
        config_module.validate_runtime_defaults(cfg)

    assert caplog.text.count("Active universe loaded: 2 symbols") == 1
    assert caplog.text.count("Synthetic mode enabled: False") == 1
    assert caplog.text.count("Markets detected: India") == 1


def test_validate_runtime_defaults_verbose_keeps_debug_detail(monkeypatch, caplog):
    _reset_init_log_guard()
    monkeypatch.setenv("SPECTRAQUANT_VERBOSE", "true")
    cfg = _minimal_cfg()

    with caplog.at_level(logging.DEBUG):
        config_module.validate_runtime_defaults(cfg)
        config_module.validate_runtime_defaults(cfg)

    assert caplog.text.count("Active universe loaded: 2 symbols") == 1
    assert caplog.text.count("Active universe tickers: INFY.NS, TCS.NS") == 2
