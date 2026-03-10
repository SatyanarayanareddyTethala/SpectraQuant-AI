from __future__ import annotations

from typer.testing import CliRunner

from spectraquant_v3.cli.main import app
from spectraquant_v3.crypto.symbols.registry import build_registry_from_config
from spectraquant_v3.crypto.universe.universe_engine import UniverseEngine


def _cfg(symbols: list[str] | None = None) -> dict:
    syms = symbols or ["BTC", "ETH", "USDT", "WBTC"]
    return {
        "run": {"mode": "normal"},
        "cache": {"root": "data/cache"},
        "qa": {},
        "execution": {"mode": "paper"},
        "portfolio": {"allocator": "equal_weight"},
        "crypto": {
            "symbols": syms,
            "exchanges": ["binance", "coinbase"],
            "universe_mode": "static",
            "universe_filters": {
                "exclude_stablecoins": True,
                "exclude_wrapped_assets": True,
                "min_market_cap_usd": 100,
                "min_daily_volume_usd": 100,
                "min_listing_age_days": 30,
                "require_exchange_coverage": True,
                "required_exchanges": ["binance"],
                "require_tradable_mapping": True,
                "collect_all_fail_reasons": True,
                "top_market_cap": {"enabled": False, "limit": 2},
            },
            "quality_gate": {
                "min_market_cap_usd": 0,
                "min_24h_volume_usd": 0,
                "min_age_days": 0,
                "require_tradable_mapping": True,
            },
        },
    }


def test_filter_stablecoin() -> None:
    cfg = _cfg(["BTC", "USDT"])
    reg = build_registry_from_config(cfg)
    rows = UniverseEngine(cfg, reg).evaluate(
        market_data={
            "BTC": {"market_cap_usd": 1_000, "daily_volume_usd": 1_000, "listing_age_days": 100, "exchanges": ["binance"]},
            "USDT": {"market_cap_usd": 1_000, "daily_volume_usd": 1_000, "listing_age_days": 100, "is_stablecoin": True, "exchanges": ["binance"]},
        }
    )
    by = {r.canonical_symbol: r for r in rows}
    assert by["BTC"].included
    assert not by["USDT"].included
    assert by["USDT"].reason == "stablecoin_excluded"


def test_filter_wrapped_asset() -> None:
    cfg = _cfg(["BTC", "WBTC"])
    reg = build_registry_from_config(cfg)
    rows = UniverseEngine(cfg, reg).evaluate(
        market_data={
            "BTC": {"market_cap_usd": 1_000, "daily_volume_usd": 1_000, "listing_age_days": 100, "exchanges": ["binance"]},
            "WBTC": {"market_cap_usd": 1_000, "daily_volume_usd": 1_000, "listing_age_days": 100, "is_wrapped": True, "exchanges": ["binance"]},
        }
    )
    assert next(r for r in rows if r.canonical_symbol == "WBTC").reason == "wrapped_asset_excluded"


def test_filters_volume_age_exchange_and_market_cap() -> None:
    cfg = _cfg(["BTC", "ETH", "SOL", "ADA"])
    reg = build_registry_from_config(cfg)
    rows = UniverseEngine(cfg, reg).evaluate(
        market_data={
            "BTC": {"market_cap_usd": 200, "daily_volume_usd": 200, "listing_age_days": 100, "exchanges": ["binance"]},
            "ETH": {"market_cap_usd": 50, "daily_volume_usd": 200, "listing_age_days": 100, "exchanges": ["binance"]},
            "SOL": {"market_cap_usd": 200, "daily_volume_usd": 50, "listing_age_days": 10, "exchanges": ["coinbase"]},
            "ADA": {"market_cap_usd": 200, "daily_volume_usd": 200, "listing_age_days": 100, "exchanges": []},
        }
    )
    by = {r.canonical_symbol: r for r in rows}
    assert by["ETH"].reason == "below_min_market_cap"
    assert by["SOL"].reason == "below_min_daily_volume"
    assert "below_min_listing_age" in by["SOL"].fail_reasons
    assert by["ADA"].reason.startswith("missing_exchange_coverage")


def test_top_market_cap_mode() -> None:
    cfg = _cfg(["BTC", "ETH", "SOL"])
    cfg["crypto"]["universe_filters"]["top_market_cap"] = {"enabled": True, "limit": 2}
    reg = build_registry_from_config(cfg)
    rows = UniverseEngine(cfg, reg).evaluate(
        market_data={
            "BTC": {"market_cap_usd": 300, "daily_volume_usd": 200, "listing_age_days": 100, "exchanges": ["binance"]},
            "ETH": {"market_cap_usd": 200, "daily_volume_usd": 200, "listing_age_days": 100, "exchanges": ["binance"]},
            "SOL": {"market_cap_usd": 100, "daily_volume_usd": 200, "listing_age_days": 100, "exchanges": ["binance"]},
        }
    )
    by = {r.canonical_symbol: r for r in rows}
    assert by["BTC"].included and by["ETH"].included
    assert not by["SOL"].included
    assert "outside_top_market_cap" in by["SOL"].fail_reasons


def test_cli_universe_prints_table(monkeypatch) -> None:
    cfg = _cfg(["BTC", "USDT"])

    def _fake_get_crypto_config(_config_dir=None):
        return cfg

    monkeypatch.setattr("spectraquant_v3.core.config.get_crypto_config", _fake_get_crypto_config)
    runner = CliRunner()
    result = runner.invoke(app, ["crypto", "universe"])
    assert result.exit_code == 0
    lines = [ln.strip() for ln in result.output.splitlines() if ln.strip()]
    assert lines[0] == "symbol included reason"
    assert any(ln.startswith("BTC ") for ln in lines[1:])
    assert any(ln.startswith("USDT ") for ln in lines[1:])
