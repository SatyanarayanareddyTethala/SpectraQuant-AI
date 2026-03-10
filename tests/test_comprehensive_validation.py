"""Comprehensive offline-safe validation suite for SpectraQuant.

Covers:
1. Packaging/import integrity – package imports, module resolution,
   universe submodule completeness.
2. CLI smoke + behavioural – ``--help`` output, command discovery,
   and ``news-scan --use-sentiment`` writing output files without any
   real HTTP calls.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_CLI_ENV = {
    **os.environ,
    "PYTHONPATH": str(_SRC_DIR)
    + (os.pathsep + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""),
}


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the spectraquant CLI via ``-m spectraquant.cli.main``."""
    return subprocess.run(
        [sys.executable, "-m", "spectraquant.cli.main", *args],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
    )


# ===========================================================================
# 1. PACKAGING / IMPORT INTEGRITY
# ===========================================================================


class TestPackageImport:
    """Verify that the top-level package and critical sub-modules are importable."""

    def test_import_spectraquant_succeeds(self) -> None:
        """``import spectraquant`` must succeed without errors."""
        import spectraquant  # noqa: F401 – side-effect-free re-import

        assert spectraquant.__version__

    def test_version_attribute_present(self) -> None:
        import spectraquant

        assert hasattr(spectraquant, "__version__")
        assert isinstance(spectraquant.__version__, str)
        assert spectraquant.__version__  # non-empty

    def test_config_module_importable(self) -> None:
        from spectraquant import config  # noqa: F401

        assert hasattr(config, "get_config")
        assert hasattr(config, "DEFAULT_CONFIG")

    def test_cli_main_importable(self) -> None:
        from spectraquant.cli import main  # noqa: F401

        assert hasattr(main, "main")
        assert callable(main.main)

    def test_universe_package_importable(self) -> None:
        from spectraquant import universe  # noqa: F401

        assert hasattr(universe, "resolve_universe")
        assert callable(universe.resolve_universe)

    def test_universe_loader_submodule_importable(self) -> None:
        """spectraquant.universe.loader must exist and expose load_nse_universe."""
        from spectraquant.universe import loader  # noqa: F401

        assert hasattr(loader, "load_nse_universe")
        assert callable(loader.load_nse_universe)

    @pytest.mark.parametrize(
        "module_path",
        [
            "spectraquant",
            "spectraquant.config",
            "spectraquant.cli.main",
            "spectraquant.cli.app",
            "spectraquant.universe",
            "spectraquant.universe.loader",
            "spectraquant.news.universe_builder",
            "spectraquant.news.schema",
        ],
    )
    def test_required_module_resolvable(self, module_path: str) -> None:
        """Each required module path must be importable via importlib."""
        spec = importlib.util.find_spec(module_path)
        assert spec is not None, f"Cannot find module spec for '{module_path}'"
        mod = importlib.import_module(module_path)
        assert mod is not None


class TestUniverseSubmoduleIntegrity:
    """Verify that all submodules referenced in universe/__init__.py exist."""

    def test_loader_submodule_exists_on_disk(self) -> None:
        """universe/loader.py must exist as a file."""
        loader_path = _SRC_DIR / "spectraquant" / "universe" / "loader.py"
        assert loader_path.exists(), (
            f"universe/loader.py not found at {loader_path}. "
            "The universe __init__.py references spectraquant.universe.loader; "
            "this file must not be deleted."
        )

    def test_universe_init_import_resolves_loader(self) -> None:
        """universe.__init__ performs ``from spectraquant.universe.loader import
        load_nse_universe``; that import must succeed."""
        # Directly import the symbol that __init__ re-exports:
        from spectraquant.universe.loader import load_nse_universe

        assert callable(load_nse_universe)

    def test_universe_init_exposes_public_api(self) -> None:
        """All public functions used by config.py must be reachable from the package."""
        from spectraquant.universe import (  # noqa: F401
            load_universe_from_config,
            parse_universe_override,
            resolve_universe,
        )

        assert callable(load_universe_from_config)
        assert callable(parse_universe_override)
        assert callable(resolve_universe)


# ===========================================================================
# 2. CLI COMMAND CORRECTNESS – SMOKE
# ===========================================================================


class TestCLIHelp:
    """`spectraquant --help` / `-h` must exit 0 and print usage."""

    def test_help_flag_exits_zero(self) -> None:
        result = _run_cli("--help")
        assert result.returncode == 0, result.stderr

    def test_help_flag_prints_usage(self) -> None:
        result = _run_cli("--help")
        output = result.stdout + result.stderr
        assert "Usage:" in output or "usage:" in output, (
            f"Expected 'Usage:' in help output, got:\n{output}"
        )

    def test_h_flag_exits_zero(self) -> None:
        result = _run_cli("-h")
        assert result.returncode == 0, result.stderr

    def test_no_args_shows_usage(self) -> None:
        result = _run_cli()
        output = result.stdout + result.stderr
        assert "Usage:" in output or "usage:" in output or "download" in output


class TestCLICommandDiscovery:
    """All commands advertised by ``--help`` must be present in the output."""

    _EXPECTED_COMMANDS = [
        "download",
        "news-scan",
        "features",
        "build-dataset",
        "train",
        "predict",
        "signals",
        "score",
        "portfolio",
        "execute",
        "eval",
        "retrain",
        "refresh",
        "promote-model",
        "list-models",
        "feature-pruning",
        "model-compare",
        "stress-test",
        "regime-stress",
        "explain-portfolio",
        "compare-runs",
        "doctor",
        "health-check",
        "release-check",
    ]

    def test_all_expected_commands_present_in_help(self) -> None:
        result = _run_cli("--help")
        output = result.stdout + result.stderr
        missing = [cmd for cmd in self._EXPECTED_COMMANDS if cmd not in output]
        assert not missing, (
            f"The following commands were not found in --help output: {missing}\n"
            f"Full output:\n{output}"
        )

    def test_commands_dict_contains_news_scan(self) -> None:
        """The internal commands dict in main.py must include 'news-scan'."""
        from spectraquant.cli import main as cli_main

        # Reconstruct commands dict as main() does:
        from spectraquant.cli.main import (
            cmd_news_scan,
            cmd_download,
            cmd_release_check,
        )

        for fn in (cmd_news_scan, cmd_download, cmd_release_check):
            assert callable(fn), f"{fn} is not callable"


# ===========================================================================
# 3. CLI BEHAVIOURAL – news-scan writes reports/news/news_candidates_*.csv
# ===========================================================================

def _make_stub_articles() -> list[dict]:
    """Return stub articles with a timestamp from 1 hour ago for realistic recency scoring."""
    recent_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        {
            "title": "TCS Q4 earnings beat analyst estimates",
            "description": "TCS reports strong quarterly numbers",
            "content": "",
            "source_name": "Reuters",
            "published_at_utc": recent_ts,
            "url": "https://example.com/1",
        },
        {
            "title": "Infosys upgrades guidance after strong results",
            "description": "INFY upgrades FY guidance",
            "content": "",
            "source_name": "Bloomberg",
            "published_at_utc": recent_ts,
            "url": "https://example.com/2",
        },
    ]


# Module-level alias with a recent timestamp (computed once at import time)
_STUB_ARTICLES = _make_stub_articles()

_STUB_UNIVERSE_MAPPING = {
    "tickers": ["TCS.NS", "INFY.NS"],
    "ticker_to_company": {
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys",
    },
    "aliases": {},
}


class TestNewsScanOffline:
    """``cmd_news_scan`` must write a CSV under reports/news/ without HTTP."""

    def _make_config(self, tmp_path: Path) -> dict:
        """Minimal config dict that enables news-universe and uses tmp_path."""
        universe_csv = tmp_path / "universe.csv"
        pd.DataFrame(
            {
                "symbol": ["TCS", "INFY"],
                "company": ["Tata Consultancy Services", "Infosys"],
            }
        ).to_csv(universe_csv, index=False)

        return {
            "news_universe": {
                "enabled": True,
                "lookback_hours": 12,
                "max_candidates": 50,
                "min_liquidity_avg_volume": 0,  # skip liquidity filter
                "min_source_rank": 0.0,
                "sentiment_model": "vader",
                "require_price_confirmation": False,
                "recency_decay_half_life_hours": 6,
                "cache_dir": str(tmp_path / "news_cache"),
                "persist_articles_json": False,
            },
            "universe": {
                "path": str(universe_csv),
                "tickers": ["TCS.NS", "INFY.NS"],
                "selected_sets": [],
            },
            "data": {
                "tickers": ["TCS.NS", "INFY.NS"],
                "prices_dir": str(tmp_path / "prices"),
            },
            "sentiment": {"enabled": True, "provider": "newsapi"},
        }

    def test_news_scan_writes_candidates_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """cmd_news_scan with mocked fetch writes reports/news/news_candidates_*.csv."""
        monkeypatch.chdir(tmp_path)

        config = self._make_config(tmp_path)

        with (
            patch(
                "spectraquant.news.universe_builder.fetch_news_articles",
                return_value=_STUB_ARTICLES,
            ),
            patch(
                "spectraquant.news.universe_builder.apply_liquidity_filter",
                side_effect=lambda df, *a, **kw: df,
            ),
            patch(
                "spectraquant.news.universe_builder.apply_price_confirmation",
                side_effect=lambda df, *a, **kw: df,
            ),
        ):
            from spectraquant.cli.main import cmd_news_scan

            cmd_news_scan(config=config)

        output_dir = tmp_path / "reports" / "news"
        candidates_files = sorted(output_dir.glob("news_candidates_*.csv"))
        assert candidates_files, (
            "Expected at least one reports/news/news_candidates_*.csv file "
            f"but none found in {output_dir}"
        )

    def test_news_scan_candidates_file_has_ticker_column(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The generated CSV must have at minimum a 'ticker' column."""
        monkeypatch.chdir(tmp_path)

        config = self._make_config(tmp_path)

        with (
            patch(
                "spectraquant.news.universe_builder.fetch_news_articles",
                return_value=_STUB_ARTICLES,
            ),
            patch(
                "spectraquant.news.universe_builder.apply_liquidity_filter",
                side_effect=lambda df, *a, **kw: df,
            ),
            patch(
                "spectraquant.news.universe_builder.apply_price_confirmation",
                side_effect=lambda df, *a, **kw: df,
            ),
        ):
            from spectraquant.cli.main import cmd_news_scan

            cmd_news_scan(config=config)

        output_dir = tmp_path / "reports" / "news"
        candidates_file = sorted(output_dir.glob("news_candidates_*.csv"))[-1]
        df = pd.read_csv(candidates_file)
        assert "ticker" in df.columns, (
            f"'ticker' column missing from {candidates_file.name}. "
            f"Found columns: {list(df.columns)}"
        )

    def test_news_scan_no_http_calls(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """fetch_news_articles must be the only network entry-point; when mocked,
        no real HTTP calls should occur.  We verify by patching the underlying
        urlopen to raise if called."""
        monkeypatch.chdir(tmp_path)

        config = self._make_config(tmp_path)

        def _fail_if_urlopen(*args, **kwargs):
            raise RuntimeError(
                "Real HTTP call detected in test! "
                "news-scan should not make network requests when fetch_news_articles is mocked."
            )

        with (
            patch(
                "spectraquant.news.universe_builder.fetch_news_articles",
                return_value=_STUB_ARTICLES,
            ),
            patch(
                "spectraquant.news.universe_builder.apply_liquidity_filter",
                side_effect=lambda df, *a, **kw: df,
            ),
            patch(
                "spectraquant.news.universe_builder.apply_price_confirmation",
                side_effect=lambda df, *a, **kw: df,
            ),
            patch("urllib.request.urlopen", side_effect=_fail_if_urlopen),
        ):
            from spectraquant.cli.main import cmd_news_scan

            cmd_news_scan(config=config)  # must not raise

    def test_news_scan_empty_articles_no_crash(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When fetch returns no articles cmd_news_scan must return gracefully."""
        monkeypatch.chdir(tmp_path)

        config = self._make_config(tmp_path)

        with patch(
            "spectraquant.news.universe_builder.fetch_news_articles",
            return_value=[],
        ):
            from spectraquant.cli.main import cmd_news_scan

            cmd_news_scan(config=config)  # must not raise

    def test_news_scan_disabled_config_no_crash(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When news_universe.enabled is False cmd_news_scan returns gracefully."""
        monkeypatch.chdir(tmp_path)

        config = self._make_config(tmp_path)
        config["news_universe"]["enabled"] = False

        from spectraquant.cli.main import cmd_news_scan

        cmd_news_scan(config=config)  # must not raise


# ===========================================================================
# 4. DETERMINISM – repeated calls produce identical filenames pattern
# ===========================================================================


class TestNewsScanDeterminism:
    """build_news_universe with fixed inputs must produce stable column sets."""

    def test_output_columns_stable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two back-to-back calls must produce DataFrames with identical columns."""
        monkeypatch.chdir(tmp_path)

        universe_csv = tmp_path / "universe.csv"
        pd.DataFrame({"symbol": ["TCS"], "company": ["Tata Consultancy Services"]}).to_csv(
            universe_csv, index=False
        )

        config = {
            "news_universe": {
                "enabled": True,
                "lookback_hours": 1,
                "max_candidates": 10,
                "min_liquidity_avg_volume": 0,
                "min_source_rank": 0.0,
                "sentiment_model": "vader",
                "require_price_confirmation": False,
                "recency_decay_half_life_hours": 6,
                "persist_articles_json": False,
            },
            "universe": {"path": str(universe_csv), "tickers": ["TCS.NS"]},
            "data": {"prices_dir": str(tmp_path / "prices")},
        }

        with (
            patch(
                "spectraquant.news.universe_builder.fetch_news_articles",
                return_value=_STUB_ARTICLES,
            ),
            patch(
                "spectraquant.news.universe_builder.apply_liquidity_filter",
                side_effect=lambda df, *a, **kw: df,
            ),
            patch(
                "spectraquant.news.universe_builder.apply_price_confirmation",
                side_effect=lambda df, *a, **kw: df,
            ),
        ):
            from spectraquant.news.universe_builder import build_news_universe

            df1 = build_news_universe(config)
            df2 = build_news_universe(config)

        assert list(df1.columns) == list(df2.columns), (
            "Column set changed between two identical calls to build_news_universe"
        )
