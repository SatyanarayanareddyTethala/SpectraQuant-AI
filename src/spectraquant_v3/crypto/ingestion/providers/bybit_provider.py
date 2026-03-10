"""Bybit REST API provider for SpectraQuant-AI-V3.

Fetches funding rates and open interest from the Bybit v5 REST API.
Pass a ``_session`` object in tests to avoid real network calls.

Base URL: ``https://api.bybit.com``
"""

from __future__ import annotations

import logging
from typing import Any

from spectraquant_v3.core.errors import DataSchemaError, SpectraQuantError

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.bybit.com"

_FUNDING_RATE_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"symbol", "fundingRate", "fundingRateTimestamp"}
)
_OPEN_INTEREST_REQUIRED_KEYS: frozenset[str] = frozenset({"openInterest", "timestamp"})


def _import_requests() -> Any:
    """Lazily import requests and raise a clear error when absent."""
    try:
        import requests  # noqa: PLC0415
        return requests
    except ImportError as exc:
        raise SpectraQuantError(
            "requests is not installed. Install it with: pip install requests"
        ) from exc


class BybitProvider:
    """Fetch funding rates and open interest from Bybit.

    Args:
        base_url: Override the Bybit base URL.
        _session: Optional requests-compatible session for testing.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        _session: Any | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = _session

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
        """Perform a GET and return the parsed JSON body.

        Raises:
            DataSchemaError:   On non-JSON response.
            SpectraQuantError: On HTTP errors, network failures, or Bybit
                retCode != 0.
        """
        url = f"{self._base_url}/{path.lstrip('/')}"
        session = self._get_session()
        try:
            response = session.get(url, params=params or {}, timeout=30)
        except Exception as exc:
            raise SpectraQuantError(
                f"BybitProvider: network error for '{url}': {exc}"
            ) from exc

        if not response.ok:
            raise SpectraQuantError(
                f"BybitProvider: HTTP {response.status_code} for '{url}': "
                f"{response.text[:200]}"
            )

        try:
            body = response.json()
        except Exception as exc:
            raise DataSchemaError(
                f"BybitProvider: failed to parse JSON from '{url}': {exc}"
            ) from exc

        if not isinstance(body, dict):
            raise DataSchemaError(
                f"BybitProvider: expected dict response from '{url}', "
                f"got {type(body).__name__}."
            )

        ret_code = body.get("retCode")
        if ret_code is not None and ret_code != 0:
            raise SpectraQuantError(
                f"BybitProvider: API error from '{url}' (retCode={ret_code}): "
                f"{body.get('retMsg', 'unknown error')}"
            )

        return body

    def _extract_list(self, body: dict, url: str) -> list:
        """Extract the inner list from a Bybit v5 paginated response."""
        result = body.get("result") or {}
        if not isinstance(result, dict):
            raise DataSchemaError(
                f"BybitProvider: 'result' in response from '{url}' is not a dict "
                f"(got {type(result).__name__})."
            )
        items = result.get("list")
        if items is None:
            raise DataSchemaError(
                f"BybitProvider: 'result.list' missing in response from '{url}'. "
                f"Keys in 'result': {list(result.keys())}"
            )
        if not isinstance(items, list):
            raise DataSchemaError(
                f"BybitProvider: 'result.list' from '{url}' is not a list "
                f"(got {type(items).__name__})."
            )
        return items

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_funding_rate(self, symbol: str, limit: int = 200) -> list[dict]:
        """Return historical funding rate records for *symbol*.

        Args:
            symbol: Bybit linear perpetual symbol, e.g. ``"BTCUSDT"``.
            limit:  Number of records to return (max 200).

        Returns:
            List of dicts with keys: ``symbol``, ``fundingRate``,
            ``fundingRateTimestamp``.

        Raises:
            DataSchemaError:   When the response has an unexpected shape.
            SpectraQuantError: On HTTP, network, or Bybit API errors.
        """
        url_path = "/v5/market/funding/history"
        body = self._get(
            url_path,
            params={"symbol": symbol.upper(), "limit": limit},
        )
        raw_list = self._extract_list(body, url_path)

        result: list[dict] = []
        for i, entry in enumerate(raw_list):
            if not isinstance(entry, dict):
                raise DataSchemaError(
                    f"BybitProvider: non-dict row at index {i} in funding rate "
                    f"response for '{symbol}': {entry!r}."
                )
            missing = _FUNDING_RATE_REQUIRED_KEYS - entry.keys()
            if missing:
                raise DataSchemaError(
                    f"BybitProvider: funding rate row {i} for '{symbol}' is missing "
                    f"required keys: {sorted(missing)}. Got keys: {list(entry.keys())}"
                )
            result.append(
                {
                    "symbol": entry["symbol"],
                    "fundingRate": entry["fundingRate"],
                    "fundingRateTimestamp": entry["fundingRateTimestamp"],
                }
            )

        logger.debug(
            "BybitProvider: fetched %d funding rate records for %s",
            len(result),
            symbol,
        )
        return result

    def get_open_interest(
        self,
        symbol: str,
        interval_time: str = "1h",
        limit: int = 50,
    ) -> list[dict]:
        """Return open interest history for *symbol*.

        Args:
            symbol:        Bybit linear perpetual symbol, e.g. ``"BTCUSDT"``.
            interval_time: Data interval; e.g. ``"5min"``, ``"15min"``,
                           ``"30min"``, ``"1h"``, ``"4h"``, ``"1d"``.
            limit:         Number of records to return (max 200).

        Returns:
            List of dicts with keys: ``symbol``, ``openInterest``,
            ``timestamp``.

        Raises:
            DataSchemaError:   When the response has an unexpected shape.
            SpectraQuantError: On HTTP, network, or Bybit API errors.
        """
        url_path = "/v5/market/open-interest"
        body = self._get(
            url_path,
            params={
                "symbol": symbol.upper(),
                "intervalTime": interval_time,
                "limit": limit,
                "category": "linear",
            },
        )
        raw_list = self._extract_list(body, url_path)

        result: list[dict] = []
        for i, entry in enumerate(raw_list):
            if not isinstance(entry, dict):
                raise DataSchemaError(
                    f"BybitProvider: non-dict row at index {i} in open interest "
                    f"response for '{symbol}': {entry!r}."
                )
            missing = _OPEN_INTEREST_REQUIRED_KEYS - entry.keys()
            if missing:
                raise DataSchemaError(
                    f"BybitProvider: open interest row {i} for '{symbol}' is missing "
                    f"required keys: {sorted(missing)}. Got keys: {list(entry.keys())}"
                )
            result.append(
                {
                    "symbol": symbol.upper(),
                    "openInterest": entry["openInterest"],
                    "timestamp": entry["timestamp"],
                }
            )

        logger.debug(
            "BybitProvider: fetched %d open interest records for %s",
            len(result),
            symbol,
        )
        return result
