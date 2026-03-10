import pandas as pd
from spectraquant.cli.main import _resolve_download_tickers


def test_from_news_subset(tmp_path):
    news = tmp_path / "reports" / "news"
    news.mkdir(parents=True)
    pd.DataFrame({"ticker": ["A.NS", "B.NS", "C.NS"]}).to_csv(news / "news_candidates_20240101_000000.csv", index=False)
    cfg = {"news_universe": {"enabled": True, "max_candidates": 2}, "data": {}, "universe": {"tickers": ["X.NS"]}}
    import os
    cwd = os.getcwd(); os.chdir(tmp_path)
    try:
        tickers, _ = _resolve_download_tickers(cfg, from_news=True)
    finally:
        os.chdir(cwd)
    assert tickers == ("A.NS", "B.NS")


def test_from_news_fallback_missing_file():
    cfg = {"news_universe": {"enabled": True}, "data": {"tickers": ["X.NS"]}, "universe": {"tickers": ["X.NS"]}}
    tickers, _ = _resolve_download_tickers(cfg, from_news=False)
    assert isinstance(tickers, tuple)
