"""Tests for news-first crypto universe builder."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from spectraquant.crypto.universe.news_crypto_universe import (
    build_news_crypto_universe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article(title: str, summary: str = "", hours_ago: float = 1.0) -> dict:
    """Create a minimal article dict for testing."""
    pub = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {
        "title": title,
        "summary": summary,
        "url": f"https://example.com/{title.replace(' ', '-')}",
        "published_utc": pub,
        "source": "test",
    }


def _base_cfg(**overrides) -> dict:
    """Return a minimal config dict with optional overrides."""
    cfg = {
        "crypto": {
            "symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "news_first": True,
            "news_max_symbols": 10,
            "news_strict": False,
        },
        "news_ai": {
            "enabled": True,
            "recency_half_life_hours": 6.0,
        },
    }
    for key, val in overrides.items():
        section, _, field = key.partition(".")
        if field:
            cfg.setdefault(section, {})[field] = val
        else:
            cfg[section] = val
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildNewsCryptoUniverse:
    """Unit tests for build_news_crypto_universe."""

    @patch("spectraquant.news.collector.collect_rss")
    @patch("spectraquant.news.dedupe.dedupe_articles")
    def test_returns_symbols_from_articles(self, mock_dedupe, mock_rss):
        """Symbols mentioned in news articles are returned."""
        articles = [
            _make_article("Bitcoin surges past $100K", "BTC rally continues"),
            _make_article("Ethereum upgrade scheduled", "ETH to improve throughput"),
        ]
        mock_rss.return_value = articles
        mock_dedupe.return_value = articles

        result = build_news_crypto_universe(_base_cfg())

        assert isinstance(result, list)
        assert len(result) > 0
        assert "BTC-USD" in result
        assert "ETH-USD" in result

    @patch("spectraquant.news.collector.collect_rss")
    @patch("spectraquant.news.dedupe.dedupe_articles")
    def test_max_symbols_caps_results(self, mock_dedupe, mock_rss):
        """Result is capped at news_max_symbols."""
        articles = [
            _make_article("Bitcoin news"),
            _make_article("Ethereum news"),
            _make_article("Solana news"),
            _make_article("Cardano news"),
            _make_article("Polkadot news"),
        ]
        mock_rss.return_value = articles
        mock_dedupe.return_value = articles

        cfg = _base_cfg(**{"crypto.news_max_symbols": 2})
        result = build_news_crypto_universe(cfg)

        assert len(result) <= 2

    @patch("spectraquant.news.collector.collect_rss")
    @patch("spectraquant.news.dedupe.dedupe_articles")
    def test_fallback_when_no_articles(self, mock_dedupe, mock_rss):
        """Falls back to configured symbols when no articles collected."""
        mock_rss.return_value = []
        mock_dedupe.return_value = []

        cfg = _base_cfg()
        result = build_news_crypto_universe(cfg)

        assert result == ["BTC-USD", "ETH-USD", "SOL-USD"]

    @patch("spectraquant.news.collector.collect_rss")
    @patch("spectraquant.news.dedupe.dedupe_articles")
    def test_strict_mode_raises_when_no_articles(self, mock_dedupe, mock_rss):
        """Strict mode raises RuntimeError when no articles available."""
        mock_rss.return_value = []
        mock_dedupe.return_value = []

        cfg = _base_cfg(**{"crypto.news_strict": True})

        with pytest.raises(RuntimeError, match="No crypto symbols found from news"):
            build_news_crypto_universe(cfg)

    @patch("spectraquant.news.collector.collect_rss")
    @patch("spectraquant.news.dedupe.dedupe_articles")
    def test_strict_mode_raises_when_no_symbols_extracted(self, mock_dedupe, mock_rss):
        """Strict mode raises when articles exist but no symbols are found."""
        articles = [_make_article("The weather is nice today")]
        mock_rss.return_value = articles
        mock_dedupe.return_value = articles

        cfg = _base_cfg(**{"crypto.news_strict": True})

        with pytest.raises(RuntimeError, match="No crypto symbols found from news"):
            build_news_crypto_universe(cfg)

    @patch("spectraquant.news.collector.collect_rss")
    @patch("spectraquant.news.dedupe.dedupe_articles")
    def test_fallback_when_no_symbols_extracted(self, mock_dedupe, mock_rss):
        """Non-strict mode falls back when no symbols found in articles."""
        articles = [_make_article("The weather is nice today")]
        mock_rss.return_value = articles
        mock_dedupe.return_value = articles

        cfg = _base_cfg()
        result = build_news_crypto_universe(cfg)

        assert result == ["BTC-USD", "ETH-USD", "SOL-USD"]

    @patch("spectraquant.news.collector.collect_rss")
    @patch("spectraquant.news.dedupe.dedupe_articles")
    def test_symbols_formatted_as_usd_pairs(self, mock_dedupe, mock_rss):
        """All returned symbols should be in SYM-USD format."""
        articles = [_make_article("Bitcoin and Solana pump")]
        mock_rss.return_value = articles
        mock_dedupe.return_value = articles

        result = build_news_crypto_universe(_base_cfg())

        for sym in result:
            assert sym.endswith("-USD"), f"Expected SYM-USD format, got {sym}"

    @patch("spectraquant.news.collector.collect_rss")
    @patch("spectraquant.news.dedupe.dedupe_articles")
    def test_recency_favours_newer_articles(self, mock_dedupe, mock_rss):
        """More recent articles should contribute higher scores."""
        articles = [
            _make_article("Ethereum breaking news", hours_ago=0.5),
            _make_article("Bitcoin historical analysis", hours_ago=48.0),
        ]
        mock_rss.return_value = articles
        mock_dedupe.return_value = articles

        cfg = _base_cfg(**{"crypto.news_max_symbols": 1})
        result = build_news_crypto_universe(cfg)

        # ETH appeared in newer article so should rank first
        assert result[0] == "ETH-USD"


class TestBuildNewsCryptoUniversePipelineIntegration:
    """Integration tests for news-first in the pipeline."""

    @patch("spectraquant.news.collector.collect_rss")
    @patch("spectraquant.news.dedupe.dedupe_articles")
    def test_pipeline_uses_news_symbols_when_news_first(self, mock_dedupe, mock_rss):
        """When news_first=True, pipeline uses news-derived symbols."""
        articles = [_make_article("Dogecoin moons again")]
        mock_rss.return_value = articles
        mock_dedupe.return_value = articles

        cfg = {
            "crypto": {
                "enabled": True,
                "news_first": True,
                "news_max_symbols": 10,
                "news_strict": False,
                "symbols": ["BTC-USD"],
                "prices_dir": "/tmp/nonexistent_prices",
                "universe_csv": "src/spectraquant/crypto/universe/crypto_universe.csv",
            },
            "news_ai": {"enabled": False, "recency_half_life_hours": 6.0},
            "onchain_ai": {"enabled": False},
            "agents": {"enabled": False},
            "crypto_meta_policy": {"enabled": False},
            "crypto_portfolio": {},
            "test_mode": {"enabled": True},
        }

        from spectraquant.pipeline.crypto_run import run_crypto_pipeline

        result = run_crypto_pipeline(cfg=cfg, dry_run=True)

        # Check symbols in the artifact
        import json
        from pathlib import Path
        artifact = json.loads(Path(result["artifact_path"]).read_text())
        # Symbols are stored as canonical form (e.g., "DOGE") for consistency
        assert "DOGE" in artifact["symbols"]

    def test_pipeline_uses_csv_universe_when_news_first_off(self):
        """When news_first=False (default), pipeline uses CSV universe."""
        from pathlib import Path

        # Resolve the CSV path relative to the project root
        csv_path = str(
            Path(__file__).resolve().parents[1]
            / "src"
            / "spectraquant"
            / "crypto"
            / "universe"
            / "crypto_universe.csv"
        )

        cfg = {
            "crypto": {
                "enabled": True,
                "news_first": False,
                "symbols": ["BTC-USD"],
                "prices_dir": "/tmp/nonexistent_prices",
                "universe_csv": csv_path,
            },
            "news_ai": {"enabled": False},
            "onchain_ai": {"enabled": False},
            "agents": {"enabled": False},
            "crypto_meta_policy": {"enabled": False},
            "crypto_portfolio": {},
            "test_mode": {"enabled": True},
        }

        from spectraquant.pipeline.crypto_run import run_crypto_pipeline

        result = run_crypto_pipeline(cfg=cfg, dry_run=True)

        # Pipeline should succeed with CSV universe
        assert "artifact_path" in result
