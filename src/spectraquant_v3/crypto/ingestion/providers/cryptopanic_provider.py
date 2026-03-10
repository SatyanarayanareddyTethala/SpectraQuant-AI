"""CryptoPanic news API provider for SpectraQuant-AI-V3.

Fetches curated crypto news from the CryptoPanic API.
Pass a ``_session`` object in tests to avoid real network calls.

Base URL: ``https://cryptopanic.com/api/v1``
"""

from __future__ import annotations

import logging
from typing import Any

from spectraquant_v3.core.errors import DataSchemaError, SpectraQuantError

logger = logging.getLogger(__name__)

_BASE_URL = "https://cryptopanic.com/api/v1"

# Fields extracted from each raw news item into a normalised dict.
_NEWS_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "url",
    "published_at",
    "currencies",
    "votes",
    "source",
)


def _import_requests() -> Any:
    """Lazily import requests and raise a clear error when absent."""
    try:
        import requests  # noqa: PLC0415
        return requests
    except ImportError as exc:
        raise SpectraQuantError(
            "requests is not installed. Install it with: pip install requests"
        ) from exc


class CryptoPanicProvider:
    """Fetch crypto news from CryptoPanic.

    Args:
        api_key:  CryptoPanic API key.  An empty string uses the public
            (rate-limited) access; a valid key enables authenticated calls.
        _session: Optional requests-compatible session for testing.
        base_url: Override the CryptoPanic base URL.
    """

    def __init__(
        self,
        api_key: str = "",
        _session: Any | None = None,
        base_url: str = _BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._session = _session
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        requests = _import_requests()
        self._session = requests.Session()
        return self._session

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        """Perform a GET and return the parsed JSON body.

        Raises:
            DataSchemaError:   On non-JSON or malformed response.
            SpectraQuantError: On HTTP errors or network failures.
        """
        url = f"{self._base_url}/{path.lstrip('/')}"
        if self._api_key:
            params = {**params, "auth_token": self._api_key}

        session = self._get_session()
        try:
            response = session.get(url, params=params, timeout=30)
        except Exception as exc:
            raise SpectraQuantError(
                f"CryptoPanicProvider: network error for '{url}': {exc}"
            ) from exc

        if not response.ok:
            raise SpectraQuantError(
                f"CryptoPanicProvider: HTTP {response.status_code} for '{url}': "
                f"{response.text[:200]}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise DataSchemaError(
                f"CryptoPanicProvider: failed to parse JSON from '{url}': {exc}"
            ) from exc

    @staticmethod
    def _normalise_item(raw: dict) -> dict:
        """Extract and normalise fields from a single raw news item."""
        return {
            "id": raw.get("id"),
            "title": raw.get("title", ""),
            "url": raw.get("url", ""),
            "published_at": raw.get("published_at", ""),
            "currencies": raw.get("currencies") or [],
            "votes": raw.get("votes") or {},
            "source": raw.get("source") or {},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_news(
        self,
        currencies: list[str] | None = None,
        filter_: str = "hot",
    ) -> list[dict]:
        """Fetch news items from CryptoPanic.

        Args:
            currencies: Optional list of currency tickers to filter by,
                e.g. ``["BTC", "ETH"]``.  When ``None`` all currencies are
                returned.
            filter_:    Feed filter; one of ``"rising"``, ``"hot"``,
                ``"bullish"``, ``"bearish"``, ``"important"``,
                ``"saved"``, ``"lol"``.

        Returns:
            List of normalised news dicts, each with keys: ``id``, ``title``,
            ``url``, ``published_at``, ``currencies``, ``votes``, ``source``.

        Raises:
            DataSchemaError:   When the API response is malformed.
            SpectraQuantError: On network or HTTP errors.
        """
        params: dict[str, Any] = {"filter": filter_}
        if currencies:
            params["currencies"] = ",".join(c.upper() for c in currencies)

        body = self._get("/posts/", params=params)

        if not isinstance(body, dict):
            raise DataSchemaError(
                f"CryptoPanicProvider: expected dict response, got {type(body).__name__}."
            )

        results = body.get("results")
        if results is None:
            raise DataSchemaError(
                "CryptoPanicProvider: API response missing 'results' key. "
                f"Keys present: {list(body.keys())}"
            )
        if not isinstance(results, list):
            raise DataSchemaError(
                f"CryptoPanicProvider: 'results' is not a list "
                f"(got {type(results).__name__})."
            )

        news = [self._normalise_item(item) for item in results if isinstance(item, dict)]
        logger.debug(
            "CryptoPanicProvider: fetched %d news items (filter=%s)", len(news), filter_
        )
        return news
