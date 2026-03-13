"""
Universe builder from news articles.

This module implements news-driven ticker universe construction:
1. Fetch recent market-moving news articles
2. Extract and score ticker mentions
3. Apply liquidity and price confirmation filters
4. Output ranked candidates for further analysis
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spectraquant.news.schema import CanonicalArticle, normalize_article, dedupe_key
from spectraquant.utils.news_universe import write_latest_news_universe

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# NSE ticker normalization
# -------------------------------------------------------
def _normalize_nse_ticker(t: str) -> str:
    t = str(t).strip()
    if not t:
        return t
    if t.endswith(".NS"):
        return t
    return f"{t}.NS"


# -------------------------------------------------------
# Ambiguous tickers: common English words / market acronyms
# Only allow these via company/alias match, not pure ticker token match.
# -------------------------------------------------------
AMBIGUOUS_TICKERS = {
    "GLOBAL", "FOCUS", "ADVANCE", "DOLLAR", "BSE", "MCX"
}



def fetch_news_articles(config: dict) -> list[dict]:
    """Fetch recent news articles using configured provider.

    IMPORTANT:
    The current provider spectraquant.sentiment.newsapi_provider.fetch_news_items()
    returns items with schema:
        {"date": "<ISO timestamp>", "text": "<news snippet>"}

    This function normalizes provider items into the canonical article schema used by this module:
        - title: str
        - description: str
        - content: str
        - source_name: str
        - published_at_utc: str (ISO)
        - url: str
    """
    news_cfg = config.get("news_universe", {})
    sentiment_cfg = config.get("sentiment", {})

    if not news_cfg.get("enabled", False):
        logger.info("News universe disabled; returning empty articles")
        return []

    lookback_hours = int(news_cfg.get("lookback_hours", 12))
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=lookback_hours)

    provider = sentiment_cfg.get("provider", "newsapi")
    if provider != "newsapi":
        logger.warning("Only 'newsapi' provider is currently supported for news universe")
        return []

    try:
        from spectraquant.sentiment.newsapi_provider import fetch_news_items
    except ImportError as e:
        logger.warning("Could not import newsapi_provider: %s", e)
        return []

    # Prefer India/NSE relevant query terms (generic global terms lead to irrelevant results)
    query_terms = news_cfg.get(
        "query_terms",
        ["NSE India", "BSE India", "Nifty", "Sensex", "Indian stocks", "India earnings"],
    )

    articles: list[dict] = []
    start_date = start_time.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")

    for query in query_terms:
        try:
            items = fetch_news_items(query, start_date, end_date, config)

            for item in items:
                canonical = normalize_article(item)
                # Skip items that have no usable text content
                if not canonical["content"] and canonical["title"] == "untitled":
                    continue
                articles.append(dict(canonical))
        except Exception as e:
            logger.warning("Failed to fetch news for query '%s': %s", query, e)
            continue

    logger.info("Fetched %d articles across %d queries", len(articles), len(query_terms))

    # Optionally persist articles
    if news_cfg.get("persist_articles_json", False):
        cache_dir = Path(news_cfg.get("cache_dir", "data/news_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / f"articles_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        logger.info("Persisted %d articles to %s", len(articles), cache_file)

    return articles


def dedupe_articles(articles: list[dict]) -> list[dict]:
    """Deduplicate articles using the canonical schema dedupe key.

    Uses ``spectraquant.news.schema.dedupe_key`` which is based on
    title + published_at_utc + source_name + url so that exact duplicates
    (same story, same source, same timestamp) are collapsed while different
    stories are preserved.  Gracefully handles missing fields.
    """
    if not articles:
        return []

    seen: set[str] = set()
    unique_articles: list[dict] = []

    for article in articles:
        # Build a CanonicalArticle view of the dict (handles missing fields)
        canonical = normalize_article(article)
        key = dedupe_key(canonical)

        if key not in seen:
            seen.add(key)
            unique_articles.append(article)

    logger.info("Deduplicated %d articles to %d unique", len(articles), len(unique_articles))
    return unique_articles


def load_universe_mapping(
    universe_csv_path: str | Path,
    aliases_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load universe mapping for ticker matching."""
    universe_path = Path(universe_csv_path)
    if not universe_path.exists():
        logger.warning("Universe file not found: %s", universe_path)
        return {"tickers": [], "ticker_to_company": {}, "aliases": {}}

    df = pd.read_csv(universe_path)

    ticker_col = None
    for col in ["symbol", "ticker", "SYMBOL", "TICKER", "Symbol", "Ticker"]:
        if col in df.columns:
            ticker_col = col
            break

    if ticker_col is None:
        logger.warning("Could not find ticker column in %s", universe_path)
        return {"tickers": [], "ticker_to_company": {}, "aliases": {}}

    tickers = [
        _normalize_nse_ticker(x)
        for x in df[ticker_col].dropna().astype(str).tolist()
    ]

    ticker_to_company: dict[str, str] = {}
    company_col = None
    for col in ["company", "name", "company_name", "COMPANY", "NAME", "Company", "Name"]:
        if col in df.columns:
            company_col = col
            break

    if company_col:
        for _, row in df.iterrows():
            ticker = _normalize_nse_ticker(row.get(ticker_col, ""))
            company = str(row.get(company_col, ""))
            if ticker and company and company.lower() not in ["nan", "none", ""]:
                ticker_to_company[ticker] = company

    logger.info("Loaded %d tickers, %d with company names", len(tickers), len(ticker_to_company))

    aliases: dict[str, str] = {}
    if aliases_path:
        aliases_file = Path(aliases_path)
        if aliases_file.exists():
            try:
                alias_df = pd.read_csv(aliases_file)
                if "ticker" in alias_df.columns and "alias" in alias_df.columns:
                    for _, row in alias_df.iterrows():
                        ticker = str(row["ticker"])
                        alias = str(row["alias"]).lower()
                        if ticker and alias and alias not in ["nan", "none", ""]:
                            aliases[alias] = ticker
                    logger.info("Loaded %d ticker aliases", len(aliases))
            except Exception as e:
                logger.warning("Failed to load aliases from %s: %s", aliases_file, e)

    return {
        "tickers": tickers,
        "ticker_to_company": ticker_to_company,
        "aliases": aliases,
    }


def _match_ticker_in_text(text: str, ticker: str) -> bool:
    """Check if ticker appears in text with word boundaries."""
    if not text or not ticker:
        return False

    import re

    text_upper = text.upper()
    ticker_clean = ticker.replace(".NS", "").replace(".L", "").upper()
    pattern = r"\b" + re.escape(ticker_clean) + r"\b"
    return bool(re.search(pattern, text_upper))


def _match_company_in_text(text: str, company: str, ticker: str, aliases: dict[str, str]) -> bool:
    """Check if company name or relevant alias appears in text (conservative token match)."""
    if not text or not company:
        return False

    text_lower = text.lower()
    company_lower = company.lower()

    company_tokens = [t for t in company_lower.split() if len(t) > 2]
    if company_tokens and all(token in text_lower for token in company_tokens):
        return True

    for alias, mapped_ticker in aliases.items():
        if mapped_ticker == ticker and alias in text_lower:
            return True

    return False


def match_articles_to_universe(
    articles: list[dict],
    universe_mapping: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    """Match articles to tickers from a loaded universe mapping."""
    if not articles:
        return {}

    tickers = universe_mapping.get("tickers", [])
    ticker_to_company = universe_mapping.get("ticker_to_company", {})
    aliases = universe_mapping.get("aliases", {})
    if not tickers:
        return {}

    results: dict[int, list[dict[str, Any]]] = {}
    for idx, article in enumerate(articles):
        text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
        matches: list[dict[str, Any]] = []
        for ticker in tickers:
            company = ticker_to_company.get(ticker, "")
            if _match_ticker_in_text(text, ticker) or (
                company and _match_company_in_text(text, company, ticker, aliases)
            ):
                matches.append({"ticker": ticker, "match_type": "text"})
        if matches:
            results[idx] = matches

    return results


def score_impact(
    articles: list[dict],
    universe_mapping: dict[str, Any],
    config: dict,
) -> pd.DataFrame:
    """Score ticker impact from articles."""
    if not articles:
        logger.info("No articles to score")
        return pd.DataFrame(columns=["ticker", "score", "mentions", "top_headlines", "asof_utc"])

    news_cfg = config.get("news_universe", {})

    precomputed_matches = (
        isinstance(universe_mapping, dict)
        and "tickers" not in universe_mapping
        and all(isinstance(k, int) for k in universe_mapping.keys())
    )

    ticker_articles_map: dict[str, list[dict]] = {}

    if precomputed_matches:
        for article_idx, matches in universe_mapping.items():
            if not (0 <= article_idx < len(articles)):
                continue
            article = articles[article_idx]
            for match in matches or []:
                ticker = str((match or {}).get("ticker", "")).strip()
                if not ticker:
                    continue
                ticker_articles_map.setdefault(ticker, []).append(article)
    else:
        tickers = universe_mapping.get("tickers", [])
        ticker_to_company = universe_mapping.get("ticker_to_company", {})
        aliases = universe_mapping.get("aliases", {})

        if not tickers:
            logger.warning("No tickers in universe mapping")
            return pd.DataFrame(columns=["ticker", "score", "mentions", "top_headlines", "asof_utc"])

        for ticker in tickers:
            company = ticker_to_company.get(ticker, "")
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
                ticker_clean = ticker.replace(".NS", "").replace(".L", "").upper()
                token_match = _match_ticker_in_text(text, ticker)
                if ticker_clean in AMBIGUOUS_TICKERS:
                    token_match = False

                if token_match or (
                    company and _match_company_in_text(text, company, ticker, aliases)
                ):
                    ticker_articles_map.setdefault(ticker, []).append(article)

    if not ticker_articles_map:
        logger.info("No ticker mentions found in articles")
        return pd.DataFrame(columns=["ticker", "score", "mentions", "top_headlines", "asof_utc"])

    source_weights: dict[str, float] = {}
    source_weights_path = news_cfg.get("source_weights_path")
    if source_weights_path and Path(source_weights_path).exists():
        try:
            weights_df = pd.read_csv(source_weights_path)
            if "source" in weights_df.columns and "weight" in weights_df.columns:
                source_weights = dict(zip(weights_df["source"], weights_df["weight"]))
                logger.info("Loaded %d source weights", len(source_weights))
        except Exception as e:
            logger.warning("Failed to load source weights: %s", e)

    half_life_hours = float(news_cfg.get("recency_decay_half_life_hours", 6))
    min_source_rank = float(news_cfg.get("min_source_rank", 0.0))
    sentiment_model = str(news_cfg.get("sentiment_model", "none"))

    now = datetime.now(timezone.utc)

    def _simple_sentiment(article: dict) -> float:
        if sentiment_model != "vader":
            return 0.0
        txt = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}".lower()
        pos_words = ["beat", "upgrade", "gain", "surge", "rally", "positive"]
        neg_words = ["miss", "downgrade", "drop", "fall", "plunge", "negative"]
        score = 0.0
        score += sum(1 for w in pos_words if w in txt)
        score -= sum(1 for w in neg_words if w in txt)
        return score

    rows = []
    for ticker, ticker_articles in ticker_articles_map.items():
        total_score = 0.0
        headlines: list[str] = []

        for article in ticker_articles:
            sentiment = _simple_sentiment(article)

            source = article.get("source_name", "") or ""
            source_weight = float(source_weights.get(source, 1.0))
            if source_weight < min_source_rank:
                continue

            published_str = str(article.get("published_at_utc", "") or "")
            try:
                published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                age_hours = (now - published).total_seconds() / 3600
                recency = 0.5 ** (age_hours / max(half_life_hours, 1e-6))
            except Exception:
                recency = 0.5

            article_score = sentiment * source_weight * recency
            if article_score == 0.0:
                article_score = source_weight * recency

            total_score += article_score

            title = str(article.get("title", "") or "")
            if title and len(headlines) < 3:
                headlines.append(title)

        if total_score > 0:
            rows.append(
                {
                    "ticker": ticker,
                    "score": total_score,
                    "mentions": len(ticker_articles),
                    "top_headlines": " || ".join(headlines[:3]),
                    "asof_utc": now,
                }
            )

    if not rows:
        logger.info("No ticker mentions found in articles")
        return pd.DataFrame(columns=["ticker", "score", "mentions", "top_headlines", "asof_utc"])

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    logger.info("Scored %d tickers with mentions", len(df))
    return df


def apply_liquidity_filter(
    candidates: pd.DataFrame,
    prices_dir: str | Path,
    min_avg_volume: float,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Filter candidates by minimum average volume."""
    if candidates.empty:
        return candidates

    if min_avg_volume <= 0:
        logger.info("No volume filter applied (min_avg_volume <= 0)")
        return candidates

    prices_path = Path(prices_dir)
    if not prices_path.exists():
        logger.warning("Prices directory not found: %s; skipping liquidity filter", prices_path)
        return candidates

    filtered_tickers: list[str] = []

    for ticker in candidates["ticker"]:
        ticker_file = prices_path / f"{ticker}.csv"
        if not ticker_file.exists():
            ticker_file = prices_path / f"{ticker}.parquet"

        if not ticker_file.exists():
            logger.debug("No price data for %s; keeping in candidates", ticker)
            filtered_tickers.append(ticker)
            continue

        try:
            df = pd.read_parquet(ticker_file) if ticker_file.suffix == ".parquet" else pd.read_csv(ticker_file)

            if "Volume" not in df.columns and "volume" not in df.columns:
                logger.debug("No Volume column for %s; keeping in candidates", ticker)
                filtered_tickers.append(ticker)
                continue

            vol_col = "Volume" if "Volume" in df.columns else "volume"
            date_col = "Date" if "Date" in df.columns else "date"
            df_sorted = df.sort_values(date_col, ascending=False)
            recent_volume = float(df_sorted.head(lookback_days)[vol_col].mean())

            if recent_volume >= min_avg_volume:
                filtered_tickers.append(ticker)
            else:
                logger.debug("Filtered %s: avg_volume=%.0f < %.0f", ticker, recent_volume, min_avg_volume)

        except Exception as e:
            logger.warning("Failed to check volume for %s: %s; keeping in candidates", ticker, e)
            filtered_tickers.append(ticker)

    result = candidates[candidates["ticker"].isin(filtered_tickers)].copy()
    logger.info("Liquidity filter: %d -> %d tickers", len(candidates), len(result))
    return result


def apply_price_confirmation(
    candidates: pd.DataFrame,
    prices_dir: str | Path,
    config: dict,
) -> pd.DataFrame:
    """Apply price confirmation filter."""
    if candidates.empty:
        return candidates

    news_cfg = config.get("news_universe", {})
    if not news_cfg.get("require_price_confirmation", False):
        logger.info("Price confirmation disabled")
        candidates = candidates.copy()
        candidates["confirm_status"] = "skipped"
        return candidates

    confirm_cfg = news_cfg.get("confirmation", {})
    method = confirm_cfg.get("method", "gap_or_volume")
    gap_threshold = float(confirm_cfg.get("gap_abs_return_threshold", 0.015))
    volume_z_threshold = float(confirm_cfg.get("volume_z_threshold", 1.5))
    lookback_days = int(confirm_cfg.get("lookback_days", 20))

    prices_path = Path(prices_dir)
    if not prices_path.exists():
        logger.warning("Prices directory not found: %s; marking all as unknown", prices_path)
        candidates = candidates.copy()
        candidates["confirm_status"] = "unknown"
        return candidates

    confirmed_tickers: list[str] = []
    statuses: dict[str, str] = {}

    for ticker in candidates["ticker"]:
        ticker_file = prices_path / f"{ticker}.csv"
        if not ticker_file.exists():
            ticker_file = prices_path / f"{ticker}.parquet"

        if not ticker_file.exists():
            statuses[ticker] = "unknown"
            confirmed_tickers.append(ticker)
            continue

        try:
            df = pd.read_parquet(ticker_file) if ticker_file.suffix == ".parquet" else pd.read_csv(ticker_file)
            date_col = "Date" if "Date" in df.columns else "date"
            df = df.sort_values(date_col, ascending=False)

            if len(df) < 2:
                statuses[ticker] = "unknown"
                confirmed_tickers.append(ticker)
                continue

            close_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else "close")
            latest_close = float(df.iloc[0][close_col])
            prev_close = float(df.iloc[1][close_col])

            if prev_close == 0:
                statuses[ticker] = "unknown"
                confirmed_tickers.append(ticker)
                continue

            abs_return = abs((latest_close - prev_close) / prev_close)
            gap_confirmed = abs_return >= gap_threshold

            volume_confirmed = False
            if "Volume" in df.columns or "volume" in df.columns:
                vol_col = "Volume" if "Volume" in df.columns else "volume"
                recent_vol = float(df.iloc[0][vol_col])

                if len(df) >= lookback_days + 1:
                    hist_vol = df.iloc[1 : lookback_days + 1][vol_col]
                    vol_mean = float(hist_vol.mean())
                    vol_std = float(hist_vol.std())

                    if vol_std > 0:
                        vol_z = (recent_vol - vol_mean) / vol_std
                        volume_confirmed = vol_z >= volume_z_threshold

            if method == "gap":
                confirmed = gap_confirmed
            elif method == "volume":
                confirmed = volume_confirmed
            else:  # gap_or_volume
                confirmed = gap_confirmed or volume_confirmed

            if confirmed:
                statuses[ticker] = "confirmed"
                confirmed_tickers.append(ticker)
            else:
                statuses[ticker] = "rejected"

        except Exception as e:
            logger.warning("Failed to check confirmation for %s: %s; marking unknown", ticker, e)
            statuses[ticker] = "unknown"
            confirmed_tickers.append(ticker)

    out = candidates.copy()
    out["confirm_status"] = out["ticker"].map(statuses)
    result = out[out["confirm_status"].isin(["confirmed", "unknown"])].copy()

    logger.info(
        "Price confirmation: %d -> %d tickers (%d confirmed, %d unknown)",
        len(out),
        len(result),
        int((result["confirm_status"] == "confirmed").sum()),
        int((result["confirm_status"] == "unknown").sum()),
    )
    return result


def build_news_universe(config: dict) -> pd.DataFrame:
    """Main entry point: build news-driven universe."""
    news_cfg = config.get("news_universe", {})

    if not news_cfg.get("enabled", False):
        logger.info("News universe disabled; returning empty DataFrame")
        return pd.DataFrame(columns=["ticker", "score", "mentions", "top_headlines", "asof_utc"])

    logger.info("Step 1: Fetching news articles...")
    articles = fetch_news_articles(config)

    logger.info("Step 2: Deduplicating articles...")
    articles = dedupe_articles(articles)

    logger.info("Step 3: Loading universe mapping...")
    universe_path = config.get("universe", {}).get("path", "data/universe/universe_nse.csv")
    aliases_path = news_cfg.get("aliases_path")
    universe_mapping = load_universe_mapping(universe_path, aliases_path)

    logger.info("Step 4: Scoring ticker impact...")
    candidates = score_impact(articles, universe_mapping, config)

    if candidates.empty:
        logger.info("No candidates found")
        return candidates

    logger.info("Step 5: Applying liquidity filter...")
    prices_dir = config.get("data", {}).get("prices_dir", "data/prices")
    min_volume = float(news_cfg.get("min_liquidity_avg_volume", 200000))
    candidates = apply_liquidity_filter(candidates, prices_dir, min_volume)

    logger.info("Step 6: Applying price confirmation...")
    candidates = apply_price_confirmation(candidates, prices_dir, config)

    max_candidates = int(news_cfg.get("max_candidates", 50))
    if len(candidates) > max_candidates:
        candidates = candidates.head(max_candidates)
        logger.info("Limited to top %d candidates", max_candidates)

    output_dir = Path("reports/news")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"news_candidates_{timestamp}.csv"
    candidates.to_csv(output_file, index=False)
    logger.info("Wrote %d candidates to %s", len(candidates), output_file)

    # Persist a stable 'latest' universe for downstream stages (predict/refresh)
    write_latest_news_universe(candidates, source_csv_path=str(output_file))

    return candidates


def run_news_universe_scan(config: dict, universe_csv_path: str | Path | None = None) -> pd.DataFrame:
    """Backwards-compatible wrapper invoked by CLI commands."""
    if universe_csv_path:
        config = dict(config)
        universe_cfg = dict(config.get("universe", {}))
        universe_cfg["path"] = str(universe_csv_path)
        config["universe"] = universe_cfg
    return build_news_universe(config)