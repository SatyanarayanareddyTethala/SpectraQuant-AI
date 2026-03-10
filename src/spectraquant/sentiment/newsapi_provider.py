"""NewsAPI sentiment provider."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)

_RUN_CACHE: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
_RUN_QUERIED: set[str] = set()
_LAST_REQUEST_AT = 0.0
_MIN_REQUEST_INTERVAL = 1.1
_WARNED_MISSING_KEY = False


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def _get_api_key() -> str | None:
    _load_dotenv()
    return os.getenv("NEWSAPI_KEY")


def _throttle() -> None:
    global _LAST_REQUEST_AT
    elapsed = time.time() - _LAST_REQUEST_AT
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _LAST_REQUEST_AT = time.time()


def _resolve_query(ticker: str, config: dict) -> str:
    sentiment_cfg = config.get("sentiment", {}) if isinstance(config, dict) else {}
    aliases = sentiment_cfg.get("ticker_aliases") or sentiment_cfg.get("company_names")
    universe_aliases = config.get("universe", {}).get("company_names") if isinstance(config, dict) else {}
    if isinstance(aliases, dict) and ticker in aliases:
        return str(aliases[ticker])
    if isinstance(universe_aliases, dict) and ticker in universe_aliases:
        return str(universe_aliases[ticker])
    return ticker.replace(".NS", "").replace(".L", "").replace("-", " ").replace("&", " ")


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response:  # noqa: S310 - intentional API access
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def fetch_news_items(ticker: str, start_date: str, end_date: str, config: dict) -> list[dict[str, Any]]:
    """Fetch recent NewsAPI articles for a ticker."""
    sentiment_cfg = config.get("sentiment") or {}
    max_articles = int(sentiment_cfg.get("max_articles_per_ticker", 50) or 50)
    query = _resolve_query(ticker, config)
    cache_key = (query, start_date, end_date, max_articles)
    if cache_key in _RUN_CACHE:
        return list(_RUN_CACHE[cache_key])

    api_key = _get_api_key()
    if not api_key:
        if sentiment_cfg.get("enabled", False) and sentiment_cfg.get("use_news", True):
            raise ValueError(
                "NEWSAPI_KEY is required when NewsAPI sentiment is enabled. "
                "Set NEWSAPI_KEY in your environment or .env file."
            )
        global _WARNED_MISSING_KEY
        if not _WARNED_MISSING_KEY:
            logger.warning("NEWSAPI_KEY missing; disabling NewsAPI sentiment fetch.")
            _WARNED_MISSING_KEY = True
        return []

    if ticker in _RUN_QUERIED:
        logger.info("Skipping additional NewsAPI query for %s in this run.", ticker)
        return []
    _RUN_QUERIED.add(ticker)

    lookback_limit = int(sentiment_cfg.get("newsapi_max_lookback_days", 30) or 30)
    if lookback_limit > 0:
        try:
            end_dt = pd.to_datetime(end_date, utc=True, errors="coerce")
            start_dt = pd.to_datetime(start_date, utc=True, errors="coerce")
            if pd.notna(end_dt) and pd.notna(start_dt):
                min_start = end_dt - pd.Timedelta(days=lookback_limit)
                if start_dt < min_start:
                    start_date = min_start.date().isoformat()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to enforce NewsAPI lookback window; using provided dates.")

    params = urlencode(
        {
            "q": query,
            "from": start_date,
            "to": end_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(100, max_articles),
            "apiKey": api_key,
        }
    )
    url = f"https://newsapi.org/v2/everything?{params}"
    try:
        _throttle()
        payload = _fetch_json(url)
    except HTTPError as exc:
        if exc.code == 426:
            logger.warning("NewsAPI returned HTTP 426 for %s; skipping news sentiment.", ticker)
        else:
            logger.warning("NewsAPI request failed for %s: %s", ticker, exc)
        return []
    except URLError as exc:
        logger.warning("NewsAPI request failed for %s: %s", ticker, exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("NewsAPI request failed for %s: %s", ticker, exc)
        return []

    items: list[dict[str, Any]] = []
    for article in payload.get("articles", []) if isinstance(payload, dict) else []:
        title = article.get("title") or ""
        description = article.get("description") or ""
        content = f"{title}. {description}".strip()
        if not content:
            continue
        published = article.get("publishedAt")
        if not published:
            continue
        items.append({"date": published, "text": content})
        if len(items) >= max_articles:
            break

    _RUN_CACHE[cache_key] = list(items)
    return items
