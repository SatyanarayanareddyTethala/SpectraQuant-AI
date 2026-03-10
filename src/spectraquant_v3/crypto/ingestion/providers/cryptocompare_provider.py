"""CryptoCompare REST API provider for SpectraQuant-AI-V3.

Fetches daily and hourly OHLCV data from the CryptoCompare min-API.
Pass a ``_session`` object in tests to avoid real network calls.

Base URL: ``https://min-api.cryptocompare.com/data``
"""

from __future__ import annotations

import logging
from typing import Any

from spectraquant_v3.core.errors import DataSchemaError, EmptyPriceDataError, SpectraQuantError

logger = logging.getLogger(__name__)

_BASE_URL = "https://min-api.cryptocompare.com/data"

_REQUIRED_OHLCV_KEYS: frozenset[str] = frozenset(
    {"time", "open", "high", "low", "close", "volumefrom", "volumeto"}
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


class CryptoCompareProvider:
    """Fetch OHLCV data from CryptoCompare.

    Args:
        api_key:  Optional CryptoCompare API key.  When empty the provider
            uses the public (rate-limited) endpoint.
        _session: Optional requests-compatible session for testing.
        base_url: Override the CryptoCompare base URL.
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
            DataSchemaError:   On non-JSON or unexpected response shape.
            SpectraQuantError: On HTTP errors or network failures.
        """
        url = f"{self._base_url}/{path.lstrip('/')}"
        if self._api_key:
            params = {**params, "api_key": self._api_key}

        session = self._get_session()
        try:
            response = session.get(url, params=params, timeout=30)
        except Exception as exc:
            raise SpectraQuantError(
                f"CryptoCompareProvider: network error for '{url}': {exc}"
            ) from exc

        if not response.ok:
            raise SpectraQuantError(
                f"CryptoCompareProvider: HTTP {response.status_code} for '{url}': "
                f"{response.text[:200]}"
            )

        try:
            body = response.json()
        except Exception as exc:
            raise DataSchemaError(
                f"CryptoCompareProvider: failed to parse JSON from '{url}': {exc}"
            ) from exc

        if not isinstance(body, dict):
            raise DataSchemaError(
                f"CryptoCompareProvider: expected dict response from '{url}', "
                f"got {type(body).__name__}."
            )
        if body.get("Response") == "Error":
            raise SpectraQuantError(
                f"CryptoCompareProvider: API error from '{url}': "
                f"{body.get('Message', 'unknown error')}"
            )
        return body

    def _parse_ohlcv(self, body: dict, endpoint: str) -> list[dict]:
        """Extract and validate the ``Data`` list from a CryptoCompare response."""
        data = body.get("Data")
        if data is None:
            raise DataSchemaError(
                f"CryptoCompareProvider: response from '{endpoint}' missing 'Data' key. "
                f"Keys present: {list(body.keys())}"
            )
        if not isinstance(data, list):
            raise DataSchemaError(
                f"CryptoCompareProvider: 'Data' from '{endpoint}' is not a list "
                f"(got {type(data).__name__})."
            )
        if not data:
            raise EmptyPriceDataError(
                f"CryptoCompareProvider: empty OHLCV data from '{endpoint}'. "
                "An empty result is never a valid success."
            )
        # data is guaranteed non-empty here; safe to index.
        first_row = data[0]
        if not isinstance(first_row, dict):
            raise DataSchemaError(
                f"CryptoCompareProvider: OHLCV row from '{endpoint}' is not a dict "
                f"(got {type(first_row).__name__})."
            )
        missing = _REQUIRED_OHLCV_KEYS - first_row.keys()
        if missing:
            raise DataSchemaError(
                f"CryptoCompareProvider: OHLCV row from '{endpoint}' is missing "
                f"required keys: {sorted(missing)}."
            )
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_daily_ohlcv(
        self,
        from_symbol: str,
        to_symbol: str = "USD",
        limit: int = 365,
    ) -> list[dict]:
        """Return daily OHLCV candles for *from_symbol* / *to_symbol*.

        Args:
            from_symbol: Base currency ticker, e.g. ``"BTC"``.
            to_symbol:   Quote currency ticker, e.g. ``"USD"``.
            limit:       Number of daily candles to return (max 2000).

        Returns:
            List of dicts with keys: ``time``, ``open``, ``high``, ``low``,
            ``close``, ``volumefrom``, ``volumeto``.

        Raises:
            EmptyPriceDataError: When the API returns no candles.
            DataSchemaError:     When the response has an unexpected shape.
            SpectraQuantError:   On HTTP or network errors.
        """
        body = self._get(
            "/v2/histoday",
            params={"fsym": from_symbol.upper(), "tsym": to_symbol.upper(), "limit": limit},
        )
        candles = self._parse_ohlcv(body, "histoday")
        logger.debug(
            "CryptoCompareProvider: fetched %d daily candles for %s/%s",
            len(candles),
            from_symbol,
            to_symbol,
        )
        return candles

    def get_hourly_ohlcv(
        self,
        from_symbol: str,
        to_symbol: str = "USD",
        limit: int = 2000,
    ) -> list[dict]:
        """Return hourly OHLCV candles for *from_symbol* / *to_symbol*.

        Args:
            from_symbol: Base currency ticker, e.g. ``"ETH"``.
            to_symbol:   Quote currency ticker, e.g. ``"USD"``.
            limit:       Number of hourly candles to return (max 2000).

        Returns:
            List of dicts with keys: ``time``, ``open``, ``high``, ``low``,
            ``close``, ``volumefrom``, ``volumeto``.

        Raises:
            EmptyPriceDataError: When the API returns no candles.
            DataSchemaError:     When the response has an unexpected shape.
            SpectraQuantError:   On HTTP or network errors.
        """
        body = self._get(
            "/v2/histohour",
            params={"fsym": from_symbol.upper(), "tsym": to_symbol.upper(), "limit": limit},
        )
        candles = self._parse_ohlcv(body, "histohour")
        logger.debug(
            "CryptoCompareProvider: fetched %d hourly candles for %s/%s",
            len(candles),
            from_symbol,
            to_symbol,
        )
        return candles
