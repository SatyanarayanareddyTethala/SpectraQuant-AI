from __future__ import annotations

from spectraquant.data import sentiment


def test_prefetch_materializes_dates_once_and_reuses_for_workers(monkeypatch) -> None:
    iter_calls = {"count": 0}

    class OneShotDates:
        def __iter__(self):
            iter_calls["count"] += 1
            for d in ["2024-01-01", "2024-01-01", "2024-01-02"]:
                yield d

    lengths: list[int] = []

    def _fake_get_sentiment_features(_ticker, dates, _config):
        lengths.append(len(dates))
        return None

    monkeypatch.setattr(sentiment, "get_sentiment_features", _fake_get_sentiment_features)

    cfg = {"sentiment": {"enabled": True, "async_prefetch": True, "prefetch_workers": 2}}
    sentiment.prefetch_sentiment_cache(["AAA.NS", "BBB.NS"], OneShotDates(), cfg)

    assert iter_calls["count"] == 1
    assert lengths == [2, 2]
