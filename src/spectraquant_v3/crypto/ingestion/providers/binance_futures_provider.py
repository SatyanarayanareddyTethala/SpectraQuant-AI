"""Binance Futures REST API provider for SpectraQuant-AI-V3.

Fetches funding rates and open interest data from the Binance USD-M Futures
REST API (fapi.binance.com).  Pass a ``_session`` object in tests to avoid
real network calls.

Base URL: ``https://fapi.binance.com``
"""

from __future__ import annotations

import logging
from typing import Any

from spectraquant_v3.core.errors import DataSchemaError, SpectraQuantError

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://fapi.binance.com"

_FUNDING_RATE_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"symbol", "fundingRate", "fundingTime"}
)
_OPEN_INTEREST_REQUIRED_KEYS: frozenset[str] = frozenset({"symbol", "openInterest", "time"})


def _import_requests() -> Any:
    """Lazily import requests and raise a clear error when absent."""
    try:
        import requests  # noqa: PLC0415
        return requests
    except ImportError as exc:
        raise SpectraQuantError(
            "requests is not installed. Install it with: pip install requests"
        ) from exc


class BinanceFuturesProvider:
    """Fetch funding rates and open interest from Binance USD-M Futures.

    Args:
        base_url: Override the Binance Futures base URL.
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
            SpectraQuantError: On HTTP errors or network failures.
        """
        url = f"{self._base_url}/{path.lstrip('/')}"
        session = self._get_session()
        try:
            response = session.get(url, params=params or {}, timeout=30)
        except Exception as exc:
            raise SpectraQuantError(
                f"BinanceFuturesProvider: network error for '{url}': {exc}"
            ) from exc

        if not response.ok:
            raise SpectraQuantError(
                f"BinanceFuturesProvider: HTTP {response.status_code} for '{url}': "
                f"{response.text[:200]}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise DataSchemaError(
                f"BinanceFuturesProvider: failed to parse JSON from '{url}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_funding_rate(self, symbol: str, limit: int = 100) -> list[dict]:
        """Return historical funding rates for a perpetual contract.

        Args:
            symbol: Binance Futures symbol, e.g. ``"BTCUSDT"``.
            limit:  Number of records to return (max 1000).

        Returns:
            List of dicts with keys: ``symbol``, ``fundingRate``,
            ``fundingTime``.

        Raises:
            DataSchemaError:   When the response has an unexpected shape.
            SpectraQuantError: On HTTP or network errors.
        """
        raw = self._get(
            "/fapi/v1/fundingRate",
            params={"symbol": symbol.upper(), "limit": limit},
        )

        if not isinstance(raw, list):
            raise DataSchemaError(
                f"BinanceFuturesProvider: expected list from fundingRate endpoint "
                f"for '{symbol}', got {type(raw).__name__}."
            )

        result: list[dict] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise DataSchemaError(
                    f"BinanceFuturesProvider: non-dict row at index {i} in "
                    f"fundingRate response for '{symbol}': {entry!r}."
                )
            missing = _FUNDING_RATE_REQUIRED_KEYS - entry.keys()
            if missing:
                raise DataSchemaError(
                    f"BinanceFuturesProvider: fundingRate row {i} for '{symbol}' "
                    f"missing required keys: {sorted(missing)}."
                )
            result.append(
                {
                    "symbol": entry["symbol"],
                    "fundingRate": entry["fundingRate"],
                    "fundingTime": entry["fundingTime"],
                }
            )

        logger.debug(
            "BinanceFuturesProvider: fetched %d funding rate records for %s",
            len(result),
            symbol,
        )
        return result

    def get_open_interest(self, symbol: str) -> dict:
        """Return the current open interest for a perpetual contract.

        Args:
            symbol: Binance Futures symbol, e.g. ``"BTCUSDT"``.

        Returns:
            Dict with keys: ``symbol``, ``openInterest``, ``time``.

        Raises:
            DataSchemaError:   When the response has an unexpected shape.
            SpectraQuantError: On HTTP or network errors.
        """
        raw = self._get(
            "/fapi/v1/openInterest",
            params={"symbol": symbol.upper()},
        )

        if not isinstance(raw, dict):
            raise DataSchemaError(
                f"BinanceFuturesProvider: expected dict from openInterest endpoint "
                f"for '{symbol}', got {type(raw).__name__}."
            )

        missing = _OPEN_INTEREST_REQUIRED_KEYS - raw.keys()
        if missing:
            raise DataSchemaError(
                f"BinanceFuturesProvider: openInterest response for '{symbol}' "
                f"missing required keys: {sorted(missing)}."
            )

        logger.debug(
            "BinanceFuturesProvider: fetched open interest for %s: %s",
            symbol,
            raw.get("openInterest"),
        )
        return {
            "symbol": raw["symbol"],
            "openInterest": raw["openInterest"],
            "time": raw["time"],
        }
