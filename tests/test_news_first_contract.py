"""Contract tests for news-first mode.

These tests verify the three core invariants described in the problem
statement:

1. Normalisation – a raw provider ``{date, text}`` item is correctly
   normalised into all canonical fields with documented fallback behaviour.
2. Dedupe robustness – missing ``source_name`` / ``published_at_utc`` does
   not crash; results are stable across repeated calls.
3. Universe extraction – given mock articles that mention tickers, the
   candidates table is non-empty and contains the expected tickers.
4. Strict-universe contract – when ``from_news=True`` and there are no
   candidates, ``NewsUniverseEmptyError`` is raised and a run manifest is
   written so there is always an audit trail.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from spectraquant.news.schema import normalize_article, dedupe_key, CanonicalArticle
from spectraquant.news.universe_builder import (
    dedupe_articles,
    load_universe_mapping,
    score_impact,
)


# ---------------------------------------------------------------------------
# 1. Normalisation tests
# ---------------------------------------------------------------------------

class TestNormalizeArticle:
    def test_text_only_provider_item_yields_canonical_fields(self):
        """Legacy {date, text} provider item normalises to all required fields."""
        item = {"date": "2026-02-21T10:00:00Z", "text": "TCS beats earnings estimates"}
        result = normalize_article(item)
        assert result["title"] == "TCS beats earnings estimates"
        assert result["description"] == "TCS beats earnings estimates"
        assert result["content"] == "TCS beats earnings estimates"
        assert result["published_at_utc"] == "2026-02-21T10:00:00Z"
        assert result["source_name"] == "unknown"
        assert result["url"] == ""

    def test_native_newsapi_item_normalises_correctly(self):
        """NewsAPI-native article with all fields preserves them."""
        item = {
            "title": "Infosys Q4 Results",
            "description": "Infosys reported strong Q4 results",
            "content": "Infosys Ltd reported a 12% rise in net profit",
            "publishedAt": "2026-02-21T08:30:00Z",
            "source": {"name": "Reuters"},
            "url": "https://reuters.com/article/1",
        }
        result = normalize_article(item)
        assert result["title"] == "Infosys Q4 Results"
        assert result["description"] == "Infosys reported strong Q4 results"
        assert result["content"] == "Infosys Ltd reported a 12% rise in net profit"
        assert result["published_at_utc"] == "2026-02-21T08:30:00Z"
        assert result["source_name"] == "Reuters"
        assert result["url"] == "https://reuters.com/article/1"

    def test_empty_text_and_no_title_yields_untitled(self):
        """An item with no usable text produces the 'untitled' sentinel."""
        result = normalize_article({})
        assert result["title"] == "untitled"
        assert result["content"] == ""

    def test_long_text_title_is_truncated(self):
        """Title derived from text is capped at 120 characters."""
        item = {"text": "A" * 200}
        result = normalize_article(item)
        assert len(result["title"]) <= 120

    def test_explicit_title_capped_at_140_chars(self):
        item = {"title": "B" * 200, "text": "something"}
        result = normalize_article(item)
        assert len(result["title"]) <= 140

    def test_source_name_from_published_at_utc_field(self):
        """published_at_utc field alias is respected."""
        item = {"text": "news", "published_at_utc": "2026-01-01T00:00:00Z"}
        result = normalize_article(item)
        assert result["published_at_utc"] == "2026-01-01T00:00:00Z"

    def test_source_name_fallback_chain(self):
        """source_name falls back from explicit field → nested dict → 'unknown'."""
        # explicit field wins
        assert normalize_article({"text": "x", "source_name": "Bloomberg"})["source_name"] == "Bloomberg"
        # nested source dict
        assert normalize_article({"text": "x", "source": {"name": "FT"}})["source_name"] == "FT"
        # no source at all
        assert normalize_article({"text": "x"})["source_name"] == "unknown"


# ---------------------------------------------------------------------------
# 2. Dedupe robustness tests
# ---------------------------------------------------------------------------

class TestDedupeArticles:
    def test_exact_duplicates_are_collapsed(self):
        articles = [
            {"title": "TCS wins deal", "published_at_utc": "2026-02-21T10:00:00Z", "source_name": "Reuters",
             "content": "TCS wins deal", "description": "", "url": ""},
            {"title": "TCS wins deal", "published_at_utc": "2026-02-21T10:00:00Z", "source_name": "Reuters",
             "content": "TCS wins deal", "description": "", "url": ""},
        ]
        assert len(dedupe_articles(articles)) == 1

    def test_distinct_articles_are_preserved(self):
        articles = [
            {"title": "TCS wins deal", "published_at_utc": "2026-02-21T10:00:00Z", "source_name": "Reuters",
             "content": "TCS wins deal", "description": "", "url": ""},
            {"title": "Infosys beats estimates", "published_at_utc": "2026-02-21T11:00:00Z", "source_name": "Bloomberg",
             "content": "Infosys beats", "description": "", "url": ""},
        ]
        assert len(dedupe_articles(articles)) == 2

    def test_missing_source_name_does_not_crash(self):
        """Dedupe should not raise when source_name is absent."""
        articles = [
            {"title": "News A", "published_at_utc": "2026-02-21T10:00:00Z",
             "content": "some content", "description": "", "url": ""},
            {"title": "News A", "published_at_utc": "2026-02-21T10:00:00Z",
             "content": "some content", "description": "", "url": ""},
        ]
        result = dedupe_articles(articles)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_missing_published_at_does_not_crash(self):
        """Dedupe should not raise when published_at_utc is absent."""
        articles = [
            {"title": "News B", "source_name": "X", "content": "content", "description": "", "url": ""},
        ]
        result = dedupe_articles(articles)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_dedupe_is_stable(self):
        """Repeated calls with the same input produce the same output."""
        articles = [
            {"title": "Story", "published_at_utc": "2026-02-21T09:00:00Z", "source_name": "Y",
             "content": "body", "description": "", "url": ""},
        ]
        assert dedupe_articles(articles) == dedupe_articles(articles)

    def test_empty_list_returns_empty_list(self):
        assert dedupe_articles([]) == []


# ---------------------------------------------------------------------------
# 3. Universe extraction test
# ---------------------------------------------------------------------------

class TestUniverseExtraction:
    def test_news_candidates_non_empty_for_matched_tickers(self, tmp_path):
        """score_impact returns non-empty DataFrame with expected tickers when
        articles mention companies from the universe."""
        universe_csv = tmp_path / "universe.csv"
        pd.DataFrame({
            "symbol": ["TCS.NS", "INFY.NS"],
            "company_name": ["Tata Consultancy Services", "Infosys"],
        }).to_csv(universe_csv, index=False)

        mapping = load_universe_mapping(str(universe_csv))

        articles = [
            {
                "title": "Tata Consultancy Services wins large government contract",
                "description": "TCS secures a new deal worth billions",
                "content": "Tata Consultancy Services has secured a large contract",
                "source_name": "Reuters",
                "published_at_utc": "2026-02-21T09:00:00Z",
                "url": "",
            },
            {
                "title": "Infosys beats Q4 estimates",
                "description": "Infosys reported strong Q4 results",
                "content": "Infosys Ltd reported excellent quarterly numbers",
                "source_name": "Bloomberg",
                "published_at_utc": "2026-02-21T10:00:00Z",
                "url": "",
            },
        ]

        config = {"news_universe": {"sentiment_model": "none"}}
        result = score_impact(articles, mapping, config)

        assert not result.empty, "score_impact should return non-empty candidates"
        assert "ticker" in result.columns
        found_tickers = set(result["ticker"].tolist())
        # At least one of the mentioned companies must be resolved
        assert found_tickers & {"TCS.NS", "INFY.NS"}, (
            f"Expected TCS.NS or INFY.NS in candidates, got {found_tickers}"
        )
        # No negative scores (article_score defaults to source_weight * recency ≥ 0)
        assert (result["score"] >= 0).all()


# ---------------------------------------------------------------------------
# 4. Strict-universe contract tests
# ---------------------------------------------------------------------------

class TestStrictNewsFirstContract:
    def test_from_news_true_no_candidates_file_raises(self, tmp_path):
        """When from_news=True and no candidates file exists, NewsUniverseEmptyError
        is raised instead of silently falling back to the full universe."""
        from spectraquant.cli.main import _resolve_download_tickers, NewsUniverseEmptyError

        cfg = {
            "news_universe": {"enabled": True},
            "universe": {"tickers": ["X.NS"]},
        }
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(NewsUniverseEmptyError):
                _resolve_download_tickers(cfg, from_news=True)
        finally:
            os.chdir(old_cwd)

    def test_from_news_true_no_candidates_file_writes_manifest(self, tmp_path):
        """When from_news=True raises NewsUniverseEmptyError, a run manifest
        must be written to reports/run/ so every run has an audit trail."""
        from spectraquant.cli.main import _resolve_download_tickers, NewsUniverseEmptyError

        cfg = {
            "news_universe": {"enabled": True},
            "universe": {"tickers": ["X.NS"]},
        }
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(NewsUniverseEmptyError):
                _resolve_download_tickers(cfg, from_news=True)
        finally:
            os.chdir(old_cwd)

        run_dir = tmp_path / "reports" / "run"
        manifests = list(run_dir.glob("run_manifest_*.json"))
        assert manifests, "A run manifest must be written on early exit"

        manifest = json.loads(manifests[0].read_text())
        assert manifest["from_news"] is True
        assert manifest["effective_tickers"] == []
        assert manifest["exit_reason"] is not None
        assert manifest["exit_reason"] != ""

    def test_from_news_true_empty_candidates_csv_raises(self, tmp_path):
        """When from_news=True and the candidates CSV is empty, raises."""
        from spectraquant.cli.main import _resolve_download_tickers, NewsUniverseEmptyError

        news_dir = tmp_path / "reports" / "news"
        news_dir.mkdir(parents=True)
        # Write a candidates file with no rows
        pd.DataFrame({"ticker": []}).to_csv(news_dir / "news_candidates_20260221_100000.csv", index=False)

        cfg = {"news_universe": {"enabled": True}, "universe": {"tickers": ["X.NS"]}}
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(NewsUniverseEmptyError):
                _resolve_download_tickers(cfg, from_news=True)
        finally:
            os.chdir(old_cwd)

    def test_from_news_false_no_candidates_falls_back(self, tmp_path):
        """When from_news=False (permissive mode), missing candidates file
        falls back to full universe without raising."""
        from spectraquant.cli.main import _resolve_download_tickers

        cfg = {
            "news_universe": {"enabled": False},
            "universe": {"tickers": ["X.NS"]},
            "data": {"tickers": ["X.NS"]},
        }
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            tickers, _ = _resolve_download_tickers(cfg, from_news=False)
        finally:
            os.chdir(old_cwd)

        assert isinstance(tickers, tuple)

    def test_from_news_true_with_valid_candidates_returns_news_tickers_only(self, tmp_path):
        """When from_news=True and valid candidates exist, only those tickers
        are returned – the config universe is NOT used."""
        from spectraquant.cli.main import _resolve_download_tickers

        news_dir = tmp_path / "reports" / "news"
        news_dir.mkdir(parents=True)
        pd.DataFrame({"ticker": ["A.NS", "B.NS"]}).to_csv(
            news_dir / "news_candidates_20260221_100000.csv", index=False
        )

        cfg = {
            "news_universe": {"enabled": True, "max_candidates": 10},
            "universe": {"tickers": ["X.NS", "Y.NS", "Z.NS"]},
            "data": {"tickers": ["X.NS", "Y.NS", "Z.NS"]},
        }
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            tickers, meta = _resolve_download_tickers(cfg, from_news=True)
        finally:
            os.chdir(old_cwd)

        assert set(tickers) == {"A.NS", "B.NS"}, (
            f"Expected only news candidates, got {tickers}"
        )
        assert "X.NS" not in tickers, "Config universe tickers must not bleed into news-first tickers"
