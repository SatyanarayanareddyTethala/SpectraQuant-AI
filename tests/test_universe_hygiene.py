import logging
# replaced hardcoded ticker
import json
import os
import shutil
from pathlib import Path

import pytest

from spectraquant.universe import load_universe_from_config, resolve_universe


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _config_for_universe(path):
    return {
        "data": {"max_tickers_per_run": 0},
        "universe": {
            "india": {"tickers_file": str(path)},
            "uk": {},
        },
    }


def test_universe_cleaning_dedupes_and_logs_metrics(tmp_path, caplog):
    universe_path = tmp_path / "universe_dirty.csv"
    shutil.copy(FIXTURES / "universe_dirty.csv", universe_path)
    config = _config_for_universe(universe_path)

    caplog.set_level(logging.INFO, logger="spectraquant.universe")
    tickers = load_universe_from_config(config)

    assert tickers == ["TICKER1.NS", "ticker2.L", "TICKER3.NS"]

    metrics_log = next(
        record.message
        for record in caplog.records
        if record.message.startswith("Loaded tickers:")
    )
    assert "raw=9" in metrics_log
    assert "cleaned=4" in metrics_log
    assert "deduped=3" in metrics_log
    assert "duplicates=1" in metrics_log


def test_universe_empty_after_cleaning_raises(tmp_path):
    universe_path = tmp_path / "universe_empty.csv"
    shutil.copy(FIXTURES / "universe_empty.csv", universe_path)
    config = _config_for_universe(universe_path)

    with pytest.raises(ValueError, match="Universe is empty after cleaning"):
        load_universe_from_config(config)


def test_resolve_universe_selected_sets_override_data_tickers(tmp_path):
    universe_path = tmp_path / "universe.csv"
    universe_path.write_text("SYMBOL\nFROMFILE\n", encoding="utf-8")
    config = {
        "data": {"tickers": ["DATA1.NS", "DATA2.L"], "max_tickers_per_run": 0},
        "universe": {
            "selected_sets": ["india"],
            "india": {"tickers_file": str(universe_path), "source": "csv"},
        },
    }
    tickers, meta = resolve_universe(config)
    assert tickers == ["FROMFILE.NS"]
    assert "india" in meta["source"]


def test_resolve_universe_accepts_path_alias(tmp_path):
    universe_path = tmp_path / "universe.csv"
    universe_path.write_text("SYMBOL\nAAA\nBBB\n", encoding="utf-8")
    config = {
        "data": {"max_tickers_per_run": 0},
        "universe": {
            "selected_sets": ["india"],
            "india": {"path": str(universe_path), "source": "csv"},
        },
    }
    tickers, meta = resolve_universe(config)
    assert tickers == ["AAA.NS", "BBB.NS"]
    assert "india" in meta["source"]


def test_resolve_universe_prefers_tickers_file_when_both_set(tmp_path, caplog):
    path_primary = tmp_path / "universe_primary.csv"
    path_secondary = tmp_path / "universe_secondary.csv"
    path_primary.write_text("SYMBOL\nPRIMARY\n", encoding="utf-8")
    path_secondary.write_text("SYMBOL\nSECONDARY\n", encoding="utf-8")
    config = {
        "data": {"max_tickers_per_run": 0},
        "universe": {
            "selected_sets": ["india"],
            "india": {
                "tickers_file": str(path_primary),
                "path": str(path_secondary),
                "source": "csv",
            },
        },
    }
    caplog.set_level(logging.WARNING, logger="spectraquant.universe")
    tickers, _ = resolve_universe(config)
    assert tickers == ["PRIMARY.NS"]
    assert any("tickers_file" in record.message for record in caplog.records)


def test_resolve_universe_selected_sets_loads_files(tmp_path):
    universe_path = tmp_path / "universe.csv"
    universe_path.write_text("AAA.NS\nBBB.NS\nCCC.NS\n", encoding="utf-8")
    config = {
        "data": {"tickers": [], "max_tickers_per_run": 0},
        "universe": {
            "selected_sets": ["india"],
            "india": {"tickers_file": str(universe_path)},
        },
    }
    tickers, meta = resolve_universe(config)
    assert tickers == ["AAA.NS", "BBB.NS", "CCC.NS"]
    assert "india" in meta["source"]


def test_resolve_universe_appends_region_suffix(tmp_path):
    universe_path = tmp_path / "universe.csv"
    universe_path.write_text("RELIANCE\nTCS\n", encoding="utf-8")
    config = {
        "data": {"tickers": [], "max_tickers_per_run": 0},
        "universe": {
            "india": {"tickers_file": str(universe_path)},
            "uk": {},
        },
    }
    tickers, _ = resolve_universe(config)
    assert tickers == ["RELIANCE.NS", "TCS.NS"]


def test_resolve_universe_equity_list_returns_100():
    equity_path = ROOT / "data" / "universe" / "EQUITY_L.csv"
    config = {
        "data": {"max_tickers_per_run": 0},
        "universe": {
            "selected_sets": ["india"],
            "india": {
                "source": "csv",
                "tickers_file": str(equity_path),
                "symbol_column": "SYMBOL",
                "suffix": ".NS",
                "filter_series_eq": True,
            },
        },
    }
    tickers, meta = resolve_universe(config)
    assert len(tickers) == 100
    assert "india" in meta["source"]


def test_resolve_universe_news_set_returns_tickers_from_json(tmp_path):
    """resolve_universe with selected_sets=['news'] loads tickers from news_universe_latest.json."""
    cache_dir = tmp_path / "data" / "news_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "news_universe_latest.json").write_text(
        json.dumps({"asof_utc": "2026-01-01T00:00:00Z", "tickers": ["A.NS", "B.NS", "C.NS"]}),
        encoding="utf-8",
    )
    config = {"universe": {"selected_sets": ["news"]}, "data": {}}
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        tickers, meta = resolve_universe(config)
    finally:
        os.chdir(old_cwd)
    assert set(tickers) == {"A.NS", "B.NS", "C.NS"}
    assert meta["source"] == "news_universe_latest.json"


def test_resolve_universe_news_set_empty_when_json_missing(tmp_path):
    """resolve_universe with selected_sets=['news'] returns empty when JSON is not present."""
    config = {"universe": {"selected_sets": ["news"]}, "data": {}}
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        tickers, meta = resolve_universe(config)
    finally:
        os.chdir(old_cwd)
    # No JSON file → no news universe tickers; empty result is expected
    assert tickers == []
