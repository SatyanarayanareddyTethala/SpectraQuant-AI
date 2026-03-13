from __future__ import annotations

from spectraquant.cli import main


def test_load_config_uses_process_cache(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_get_config():
        calls["count"] += 1
        return {"data": {}, "universe": {}, "qa": {}}

    monkeypatch.setattr(main, "_CONFIG_CACHE", None)
    monkeypatch.setattr(main, "get_config", _fake_get_config)

    cfg1 = main._load_config()
    cfg2 = main._load_config()

    assert calls["count"] == 1
    assert cfg1 == cfg2
    assert cfg1 is not cfg2
