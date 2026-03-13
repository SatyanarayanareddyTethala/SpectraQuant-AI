from __future__ import annotations

import pandas as pd

from spectraquant.data import sentiment


def test_sentiment_disabled_makes_no_provider_calls(monkeypatch) -> None:
    called = {"news": False, "social": False}

    def _news(*args, **kwargs):
        called["news"] = True
        return []

    def _social(*args, **kwargs):
        called["social"] = True
        return []

    monkeypatch.setattr(sentiment, "_fetch_news_items", _news)
    monkeypatch.setattr(sentiment, "_fetch_social_items", _social)

    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    cfg = {"sentiment": {"enabled": False}}

    result = sentiment.get_sentiment_features("INFY.NS", dates, cfg)

    assert not called["news"]
    assert not called["social"]
    assert (result[sentiment.SENTIMENT_COLUMNS] == 0.0).all().all()
