"""CoinGecko REST API provider for SpectraQuant-AI-V3.

Fetches coin metadata, market data, and OHLCV history from the public
CoinGecko API (v3).  Pass a custom ``_session`` object in tests to avoid
real network calls.

Base URL: ``https://api.coingecko.com/api/v3``
"""

from __future__ import annotations

import logging
from typing import Any

from spectraquant_v3.core.errors import DataSchemaError, SpectraQuantError, SymbolResolutionError

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.coingecko.com/api/v3"


def _import_requests() -> Any:
    """Lazily import requests and raise a clear error when absent."""
    try:
        import requests  # noqa: PLC0415
        return requests
    except ImportError as exc:
        raise SpectraQuantError(
            "requests is not installed. Install it with: pip install requests"
        ) from exc


class CoinGeckoProvider:
    """Fetch market data, coin lists, and OHLCV from CoinGecko.

    Args:
        _session: Optional requests-compatible session object injected for
            testing.  When ``None`` (default) the provider creates a new
            ``requests.Session`` on first use.
        base_url: Override the CoinGecko base URL (useful for proxies /
            testing).
    """

    def __init__(
        self,
        _session: Any | None = None,
        base_url: str = _BASE_URL,
    ) -> None:
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

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Perform a GET request and return the parsed JSON body.

        Raises:
            SymbolResolutionError: On HTTP 404.
            DataSchemaError:       On non-JSON or unexpected response body.
            SpectraQuantError:     On other HTTP or network errors.
        """
        url = f"{self._base_url}/{path.lstrip('/')}"
        session = self._get_session()
        try:
            response = session.get(url, params=params or {}, timeout=30)
        except Exception as exc:
            raise SpectraQuantError(
                f"CoinGeckoProvider: network error for '{url}': {exc}"
            ) from exc

        if response.status_code == 404:
            raise SymbolResolutionError(
                f"CoinGeckoProvider: resource not found at '{url}' (HTTP 404)."
            )
        if not response.ok:
            raise SpectraQuantError(
                f"CoinGeckoProvider: HTTP {response.status_code} for '{url}': "
                f"{response.text[:200]}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise DataSchemaError(
                f"CoinGeckoProvider: failed to parse JSON response from '{url}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_coin_market_data(self, coin_id: str) -> dict:
        """Return current price, market cap, and volume for *coin_id*.

        Args:
            coin_id: CoinGecko coin ID, e.g. ``"bitcoin"``, ``"ethereum"``.

        Returns:
            Dict containing at minimum ``id``, ``symbol``, ``name``,
            ``market_data`` (with ``current_price``, ``market_cap``,
            ``total_volume`` sub-keys).

        Raises:
            SymbolResolutionError: When *coin_id* is not found.
            DataSchemaError:       When the response is missing expected keys.
        """
        data = self._get(
            f"/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false",
            },
        )
        if not isinstance(data, dict) or "id" not in data:
            raise DataSchemaError(
                f"CoinGeckoProvider: unexpected market data response for '{coin_id}': "
                f"{str(data)[:200]}"
            )
        logger.debug("CoinGeckoProvider: fetched market data for %s", coin_id)
        return data

    def get_coin_list(self) -> list[dict]:
        """Return the full list of coins with ``id``, ``symbol``, and ``name``.

        Returns:
            List of dicts, each with ``id``, ``symbol``, ``name`` keys.

        Raises:
            DataSchemaError: When the response is not a list.
        """
        data = self._get("/coins/list")
        if not isinstance(data, list):
            raise DataSchemaError(
                f"CoinGeckoProvider: expected list from /coins/list, "
                f"got {type(data).__name__}."
            )
        logger.debug("CoinGeckoProvider: fetched coin list (%d entries)", len(data))
        return data

    def get_ohlcv(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 365,
    ) -> list[list]:
        """Return OHLCV data for *coin_id* over the last *days* days.

        CoinGecko returns candles as ``[timestamp_ms, open, high, low, close]``
        lists (no volume column in the OHLC endpoint).

        Args:
            coin_id:     CoinGecko coin ID, e.g. ``"bitcoin"``.
            vs_currency: Quote currency, e.g. ``"usd"``.
            days:        Number of historical days to fetch.

        Returns:
            List of ``[timestamp_ms, open, high, low, close]`` candles.

        Raises:
            SymbolResolutionError: When *coin_id* is unknown.
            DataSchemaError:       When the response has an unexpected shape.
        """
        data = self._get(
            f"/coins/{coin_id}/ohlc",
            params={"vs_currency": vs_currency, "days": days},
        )
        if not isinstance(data, list):
            raise DataSchemaError(
                f"CoinGeckoProvider: expected list from OHLCV endpoint for "
                f"'{coin_id}', got {type(data).__name__}."
            )
        if data and not isinstance(data[0], (list, tuple)):
            raise DataSchemaError(
                f"CoinGeckoProvider: OHLCV row for '{coin_id}' is not a list/tuple "
                f"(got {type(data[0]).__name__})."
            )
        if data and len(data[0]) < 5:
            raise DataSchemaError(
                f"CoinGeckoProvider: OHLCV row for '{coin_id}' has {len(data[0])} "
                f"columns but at least 5 are required [ts, O, H, L, C]. "
                f"Got: {data[0]!r}."
            )
        logger.debug(
            "CoinGeckoProvider: fetched %d OHLCV rows for %s", len(data), coin_id
        )
        return data
