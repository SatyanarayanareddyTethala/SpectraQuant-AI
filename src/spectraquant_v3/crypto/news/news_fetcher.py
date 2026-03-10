from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any, Callable, Protocol


class NewsAdapter(Protocol):
    def fetch_page(self, page: int, page_size: int = 50) -> tuple[list[dict[str, Any]], bool]: ...


class CryptoPanicAdapter:
    def __init__(self, provider: Any, retries: int = 2, backoff_seconds: float = 0.2) -> None:
        self.provider = provider
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def fetch_page(self, page: int, page_size: int = 50) -> tuple[list[dict[str, Any]], bool]:
        params = {"page": page, "page_size": page_size}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if hasattr(self.provider, "get_news"):
                    try:
                        items = self.provider.get_news(page=page, page_size=page_size)
                    except TypeError:
                        items = self.provider.get_news()
                    if isinstance(items, list):
                        return items, len(items) >= page_size
                if hasattr(self.provider, "_get"):
                    body = self.provider._get("/posts/", params=params)  # noqa: SLF001
                    items = body.get("results", []) if isinstance(body, dict) else []
                    has_more = bool(body.get("next")) if isinstance(body, dict) else False
                    return list(items or []), has_more
                return [], False
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds)
        raise RuntimeError(f"CryptoPanicAdapter failed after retries: {last_error}")


class CoinDeskRSSAdapter:
    COINDESK_RSS = "https://www.coindesk.com/arc/outboundfeeds/rss/"

    def __init__(self, session: Any | None = None, retries: int = 2) -> None:
        self.session = session
        self.retries = retries

    def _get_text(self, url: str) -> str:
        if self.session is not None:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            return response.text
        import requests  # noqa: PLC0415

        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.text

    def fetch_page(self, page: int, page_size: int = 50) -> tuple[list[dict[str, Any]], bool]:
        del page, page_size
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                xml_text = self._get_text(self.COINDESK_RSS)
                root = ET.fromstring(xml_text)
                rows: list[dict[str, Any]] = []
                for item in root.findall("./channel/item"):
                    rows.append(
                        {
                            "guid": item.findtext("guid") or item.findtext("link"),
                            "title": item.findtext("title") or "",
                            "link": item.findtext("link") or "",
                            "pubDate": item.findtext("pubDate") or "",
                        }
                    )
                return rows, False
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.2)
        raise RuntimeError(f"CoinDeskRSSAdapter failed after retries: {last_error}")


def _published_at_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    published_at = str(row.get("published_at") or row.get("pubDate") or "")
    article_id = str(row.get("id") or row.get("guid") or "")
    url = str(row.get("url") or row.get("link") or "")
    return (published_at, article_id, url)


def fetch_articles(
    adapter: NewsAdapter,
    max_pages: int = 1,
    page_size: int = 50,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        items, has_more = adapter.fetch_page(page=page, page_size=page_size)
        collected.extend(items)
        if not has_more:
            break
    # Deterministic ordering: descending by time, then by id/url for stable ties.
    collected_sorted = sorted(collected, key=_published_at_sort_key, reverse=True)
    if normalizer is None:
        return collected_sorted
    return [normalizer(item) for item in collected_sorted]
