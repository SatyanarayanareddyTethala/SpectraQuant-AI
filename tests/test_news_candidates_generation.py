import pandas as pd
from spectraquant.news.universe_builder import dedupe_articles, score_impact


def test_scoring_and_ordering():
    arts = [
        {"title": "A beat", "description": "upgrade", "content": "", "source_name": "S1", "published_at_utc": "2099-01-01T00:00:00Z"},
        {"title": "B miss", "description": "downgrade", "content": "", "source_name": "S1", "published_at_utc": "2099-01-01T00:00:00Z"},
    ]
    matches = {0: [{"ticker": "A.NS"}], 1: [{"ticker": "B.NS"}]}
    cfg = {"news_universe": {"sentiment_model": "vader", "recency_decay_half_life_hours": 6}}
    out = score_impact(arts, matches, cfg)
    assert out.iloc[0]["ticker"] == "A.NS"


def test_dedupe_works():
    arts = [{"title": "X", "source_name": "S", "published_at_utc": "2024-01-01T00:00:00Z"}] * 2
    out = dedupe_articles(arts)
    assert len(out) == 1
