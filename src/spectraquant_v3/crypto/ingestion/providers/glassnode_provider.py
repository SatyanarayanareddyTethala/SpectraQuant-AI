"""Glassnode on-chain metrics provider for SpectraQuant-AI-V3.

Fetches on-chain metrics (active addresses, transaction counts, fee volumes,
etc.) from the Glassnode API.  Pass a ``_session`` object in tests to avoid
real network calls.

Base URL: ``https://api.glassnode.com/v1/metrics``
"""

from __future__ import annotations

import logging
from typing import Any

from spectraquant_v3.core.errors import DataSchemaError, SpectraQuantError

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.glassnode.com/v1/metrics"


def _import_requests() -> Any:
    """Lazily import requests and raise a clear error when absent."""
    try:
        import requests  # noqa: PLC0415
        return requests
    except ImportError as exc:
        raise SpectraQuantError(
            "requests is not installed. Install it with: pip install requests"
        ) from exc


class GlassnodeProvider:
    """Fetch on-chain metrics from Glassnode.

    Args:
        api_key:  Glassnode API key.  Required for most endpoints beyond
            the free tier.
        _session: Optional requests-compatible session for testing.
        base_url: Override the Glassnode base URL.

    Example metric paths::

        "addresses/active_count"
        "transactions/count"
        "fees/volume_sum"
        "market/price_usd_close"
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
                f"GlassnodeProvider: network error for '{url}': {exc}"
            ) from exc

        if response.status_code == 401:
            raise SpectraQuantError(
                f"GlassnodeProvider: authentication failed for '{url}'. "
                "Ensure a valid api_key is supplied."
            )
        if not response.ok:
            raise SpectraQuantError(
                f"GlassnodeProvider: HTTP {response.status_code} for '{url}': "
                f"{response.text[:200]}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise DataSchemaError(
                f"GlassnodeProvider: failed to parse JSON from '{url}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_metric(
        self,
        asset: str,
        metric_path: str,
        since: int | None = None,
        until: int | None = None,
        resolution: str = "24h",
    ) -> list[dict]:
        """Fetch a single on-chain metric for *asset*.

        Args:
            asset:       Coin ticker, e.g. ``"BTC"``, ``"ETH"``.
            metric_path: Glassnode metric path, e.g. ``"addresses/active_count"``.
            since:       Start time as a Unix timestamp (seconds).
            until:       End time as a Unix timestamp (seconds).
            resolution:  Data resolution; e.g. ``"24h"``, ``"1h"``, ``"10m"``.

        Returns:
            List of ``{"t": <unix_timestamp>, "v": <value>}`` dicts ordered
            by ascending timestamp.

        Raises:
            DataSchemaError:   When the response has an unexpected shape.
            SpectraQuantError: On HTTP or network errors.
        """
        params: dict[str, Any] = {
            "a": asset.upper(),
            "i": resolution,
        }
        if since is not None:
            params["s"] = since
        if until is not None:
            params["u"] = until

        raw = self._get(metric_path, params=params)

        if not isinstance(raw, list):
            raise DataSchemaError(
                f"GlassnodeProvider: expected list response for metric "
                f"'{metric_path}' ({asset}), got {type(raw).__name__}."
            )

        normalised: list[dict] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict) or "t" not in entry or "v" not in entry:
                raise DataSchemaError(
                    f"GlassnodeProvider: unexpected row shape at index {i} for "
                    f"metric '{metric_path}' ({asset}): expected {{\"t\": ..., \"v\": ...}}, "
                    f"got {entry!r}."
                )
            normalised.append({"t": entry["t"], "v": entry["v"]})

        logger.debug(
            "GlassnodeProvider: fetched %d rows for metric '%s' (%s, %s)",
            len(normalised),
            metric_path,
            asset,
            resolution,
        )
        return normalised
