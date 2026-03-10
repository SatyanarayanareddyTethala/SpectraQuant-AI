"""Sentiment ingestion utilities for SpectraQuant."""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import importlib.util

import pandas as pd

from spectraquant.sentiment.newsapi_provider import fetch_news_items

logger = logging.getLogger(__name__)

SENTIMENT_DIR = Path("data/sentiment")

NEWS_COLUMNS = [
    "news_sentiment_avg",
    "news_sentiment_std",
    "news_count",
]
SOCIAL_COLUMNS = [
    "social_sentiment_avg",
    "social_sentiment_std",
    "social_count",
]
SENTIMENT_COLUMNS = NEWS_COLUMNS + SOCIAL_COLUMNS


def _ensure_sentiment_dir() -> None:
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    dt = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.normalize()


def _sentiment_cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return SENTIMENT_DIR / f"{safe}_daily.json"


def load_sentiment_cache(ticker: str) -> pd.DataFrame:
    _ensure_sentiment_dir()
    path = _sentiment_cache_path(ticker)
    if not path.exists():
        return pd.DataFrame(columns=["date", *SENTIMENT_COLUMNS])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Sentiment cache for %s is invalid JSON; ignoring.", ticker)
        return pd.DataFrame(columns=["date", *SENTIMENT_COLUMNS])
    df = pd.DataFrame(payload)
    if df.empty:
        return pd.DataFrame(columns=["date", *SENTIMENT_COLUMNS])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.normalize()
    return df


def save_sentiment_cache(ticker: str, df: pd.DataFrame) -> None:
    _ensure_sentiment_dir()
    path = _sentiment_cache_path(ticker)
    if df.empty:
        return
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    path.write_text(df.to_json(orient="records", date_format="iso"), encoding="utf-8")


def _extract_api_key(cfg: dict, keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in cfg and cfg[key]:
            return str(cfg[key])
    for key in keys:
        env_value = os.getenv(key.upper())
        if env_value:
            return env_value
    return None


@lru_cache(maxsize=1)
def _load_finbert_pipeline():
    if importlib.util.find_spec("transformers") is None:
        logger.warning("transformers not installed; sentiment features disabled.")
        return None
    from transformers import pipeline

    model_name = "yiyanghkust/finbert-tone"
    logger.info("Loading FinBERT sentiment pipeline (%s)", model_name)
    return pipeline("text-classification", model=model_name, tokenizer=model_name, truncation=True)


def _score_sentiment(texts: list[str]) -> list[float]:
    if not texts:
        return []
    pipeline = _load_finbert_pipeline()
    if pipeline is None:
        return [0.0 for _ in texts]
    results = pipeline(texts)
    scores: list[float] = []
    for result in results:
        label = str(result.get("label", "")).lower()
        score = float(result.get("score", 0.0))
        if "pos" in label:
            scores.append(score)
        elif "neg" in label:
            scores.append(-score)
        else:
            scores.append(0.0)
    return scores


def _fetch_json(url: str, headers: dict[str, str] | None = None) -> dict:
    request = Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response:  # noqa: S310 - intentional API access
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _fetch_news_items(ticker: str, start_date: str, end_date: str, config: dict) -> list[dict[str, Any]]:
    sentiment_cfg = config.get("sentiment") or {}
    if not sentiment_cfg.get("enabled", True):
        return []
    if not sentiment_cfg.get("use_news", True):
        logger.info("News sentiment disabled via config; skipping news sentiment for %s.", ticker)
        return []
    news_cfg = sentiment_cfg.get("news", {})
    provider = str(sentiment_cfg.get("provider") or news_cfg.get("provider", "newsapi") or "").lower()
    if provider in {"", "none", "disabled", "off", "false"}:
        logger.info("News sentiment provider disabled; skipping news sentiment for %s.", ticker)
        return []

    if provider == "finnhub":
        api_key = _extract_api_key(news_cfg, ["api_key", "finnhub_key", "finnhub_api_key"])
        if not api_key:
            logger.warning("No Finnhub API key configured; skipping news sentiment for %s.", ticker)
            return []
        params = urlencode({"symbol": ticker, "from": start_date, "to": end_date, "token": api_key})
        url = f"https://finnhub.io/api/v1/company-news?{params}"
        try:
            payload = _fetch_json(url)
        except HTTPError as exc:
            if exc.code == 426:
                logger.warning("Finnhub returned HTTP 426 for %s; skipping news sentiment.", ticker)
            else:
                logger.warning("Finnhub request failed for %s: %s", ticker, exc)
            return []
        except URLError as exc:
            logger.warning("Finnhub request failed for %s: %s", ticker, exc)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Finnhub request failed for %s: %s", ticker, exc)
            return []
        items = []
        for item in payload or []:
            headline = item.get("headline") or ""
            summary = item.get("summary") or ""
            content = f"{headline}. {summary}".strip()
            if not content:
                continue
            published = _normalize_date(item.get("datetime", item.get("date")))
            if published is None and item.get("datetime"):
                published = pd.to_datetime(item.get("datetime"), unit="s", utc=True, errors="coerce")
            if published is None:
                continue
            items.append({"date": published, "text": content})
        return items

    items = fetch_news_items(ticker, start_date, end_date, config)
    normalized: list[dict[str, Any]] = []
    for item in items:
        published = _normalize_date(item.get("date"))
        if published is None:
            continue
        normalized.append({"date": published, "text": item.get("text", "")})
    return normalized


def _fetch_social_items(ticker: str, config: dict) -> list[dict[str, Any]]:
    sentiment_cfg = config.get("sentiment") or {}
    if not sentiment_cfg.get("enabled", True):
        return []
    social_cfg = sentiment_cfg.get("social", {})
    if social_cfg.get("enabled") is False:
        logger.info("Social sentiment disabled via config; skipping social sentiment for %s.", ticker)
        return []
    provider = str(social_cfg.get("provider", "stocktwits") or "").lower()
    if provider in {"", "none", "disabled", "off", "false"}:
        logger.info("Social sentiment provider disabled; skipping social sentiment for %s.", ticker)
        return []
    if provider == "twitter":
        logger.warning("Twitter API integration not configured; skipping social sentiment for %s.", ticker)
        return []
    if provider != "stocktwits":
        logger.warning("Unknown social sentiment provider '%s'; skipping %s.", provider, ticker)
        return []
    symbol = ticker.split(".")[0]
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    try:
        payload = _fetch_json(url)
    except HTTPError as exc:
        if exc.code == 426:
            logger.warning("Stocktwits returned HTTP 426 for %s; skipping social sentiment.", ticker)
        else:
            logger.warning("Stocktwits request failed for %s: %s", ticker, exc)
        return []
    except URLError as exc:
        logger.warning("Stocktwits request failed for %s: %s", ticker, exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stocktwits request failed for %s: %s", ticker, exc)
        return []
    items = []
    for message in payload.get("messages", []) if isinstance(payload, dict) else []:
        body = message.get("body") or ""
        created_at = _normalize_date(message.get("created_at"))
        if not body or created_at is None:
            continue
        items.append({"date": created_at, "text": body})
    return items


def _aggregate_scores(items: list[dict[str, Any]], prefix: str) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["date", f"{prefix}_sentiment_avg", f"{prefix}_sentiment_std", f"{prefix}_count"])
    texts = [item["text"] for item in items]
    scores = _score_sentiment(texts)
    scored_items = []
    for item, score in zip(items, scores):
        dt = _normalize_date(item.get("date"))
        if dt is None:
            continue
        scored_items.append({"date": dt, "score": score})
    if not scored_items:
        return pd.DataFrame(columns=["date", f"{prefix}_sentiment_avg", f"{prefix}_sentiment_std", f"{prefix}_count"])
    df = pd.DataFrame(scored_items)
    grouped = df.groupby("date")
    agg = grouped["score"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(
        columns={
            "mean": f"{prefix}_sentiment_avg",
            "std": f"{prefix}_sentiment_std",
            "count": f"{prefix}_count",
        }
    )
    return agg


def get_sentiment_features(ticker: str, dates: Iterable[Any], config: dict) -> pd.DataFrame:
    _ensure_sentiment_dir()
    dates = pd.to_datetime(list(dates), utc=True, errors="coerce")
    if hasattr(dates, "dt"):
        dates = dates.dt.normalize()
        dates = dates[dates.notna()]
    else:
        dates = dates.normalize()
        dates = dates[pd.notna(dates)]
    if dates.empty:
        return pd.DataFrame(columns=["date", *SENTIMENT_COLUMNS])

    sentiment_cfg = config.get("sentiment") or {}
    if not sentiment_cfg.get("enabled", False):
        empty = pd.DataFrame({"date": dates})
        for col in SENTIMENT_COLUMNS:
            empty[col] = 0.0
        return empty

    cache_df = load_sentiment_cache(ticker)
    cache_df = cache_df.copy()
    cache_df["date"] = pd.to_datetime(cache_df["date"], utc=True, errors="coerce").dt.normalize()
    cache_df = cache_df.dropna(subset=["date"])

    end_date = dates.max().date().isoformat()
    use_news = bool(sentiment_cfg.get("use_news", True))
    use_social = bool(sentiment_cfg.get("use_social", True))
    lookback_days = int(sentiment_cfg.get("lookback_days", 30) or 30)
    newsapi_limit = int(sentiment_cfg.get("newsapi_max_lookback_days", lookback_days) or lookback_days)
    effective_lookback = min(lookback_days, newsapi_limit) if use_news else lookback_days
    if effective_lookback > 0:
        start_dt = dates.max() - pd.Timedelta(days=effective_lookback)
        start_date = max(start_dt, dates.min()).date().isoformat()
    else:
        start_date = dates.min().date().isoformat()

    cached_dates = set(cache_df["date"].dropna().dt.normalize())
    missing_dates = [d for d in dates.unique() if d not in cached_dates]

    refresh = bool(sentiment_cfg.get("refresh_cache", True))
    if missing_dates and refresh:
        logger.info("Fetching sentiment for %s (%s -> %s)", ticker, start_date, end_date)
        news_items = _fetch_news_items(ticker, start_date, end_date, config) if use_news else []
        social_items = _fetch_social_items(ticker, config) if use_social else []
        news_df = _aggregate_scores(news_items, "news")
        social_df = _aggregate_scores(social_items, "social")
        if not news_df.empty or not social_df.empty:
            merged = pd.merge(news_df, social_df, on="date", how="outer")
            merged = merged.fillna(0.0)
            cache_df = pd.concat([cache_df, merged], ignore_index=True)
            cache_df = cache_df.drop_duplicates(subset=["date"], keep="last")
            save_sentiment_cache(ticker, cache_df)

    if cache_df.empty:
        empty = pd.DataFrame({"date": dates})
        for col in SENTIMENT_COLUMNS:
            empty[col] = 0.0
        return empty

    cache_df = cache_df.drop_duplicates(subset=["date"], keep="last")
    cache_df = cache_df.set_index("date")
    features = cache_df.reindex(dates, fill_value=0.0).reset_index()
    features = features.rename(columns={"index": "date"})
    features = features.fillna(0.0)
    return features
