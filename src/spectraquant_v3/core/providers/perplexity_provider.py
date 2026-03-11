"""Perplexity-backed news intelligence provider for SpectraQuant-AI-V3.

Uses the Perplexity *sonar* chat-completions API to discover market-moving
catalysts and extract structured event data for a list of symbols.

Design notes
------------
* The adapter is thin — it handles transport, prompt construction, and JSON
  parsing.  It never writes to cache or makes run-mode decisions.
* ``httpx`` is imported lazily to avoid hard dependencies in test / offline
  environments.
* An optional ``_session`` parameter allows tests to inject a mock HTTP
  client without network calls.
* The provider always returns :class:`NewsIntelligenceRecord` instances.
  If the Perplexity response cannot be parsed, it returns an empty list
  and logs a warning rather than raising, so that news intelligence failures
  remain non-fatal.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

from spectraquant_v3.core.errors import SpectraQuantError
from spectraquant_v3.core.news_schema import NewsIntelligenceRecord

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.perplexity.ai"
_DEFAULT_MODEL = "sonar"

# ---------------------------------------------------------------------------
# System prompt: instruct Perplexity to return structured JSON.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a financial news intelligence analyst. Given a list of ticker symbols, \
identify the most significant recent market-moving events for each symbol. \
Return your response as a JSON array of objects. Each object must have exactly \
these fields:
  - "canonical_symbol": string (the ticker)
  - "timestamp": string (ISO-8601 UTC)
  - "event_type": string (e.g. "earnings", "listing", "regulation", \
"security_incident", "partnership", "product_launch", "macro", "general")
  - "sentiment_score": number between -1.0 and 1.0
  - "impact_score": number between 0.0 and 1.0
  - "article_count": integer >= 1
  - "source_urls": list of URL strings
  - "confidence": number between 0.0 and 1.0
  - "rationale": string (brief explanation)

Return ONLY the JSON array, no markdown fences or extra text.\
"""


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _import_httpx() -> Any:
    """Lazily import httpx and raise a clear error when absent."""
    try:
        import httpx  # noqa: PLC0415

        return httpx
    except ImportError as exc:
        raise SpectraQuantError(
            "httpx is not installed. Install it with: pip install httpx"
        ) from exc


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class PerplexityNewsProvider:
    """Fetch structured news intelligence from Perplexity AI.

    Args:
        api_key:   Perplexity API key.  When empty, reads from the
            ``PERPLEXITY_API_KEY`` environment variable.
        model:     Perplexity model identifier (default ``"sonar"``).
        base_url:  Override the API base URL (useful for testing).
        _session:  Optional httpx-compatible client for testing.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        _session: Any | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._session = _session

    # ------------------------------------------------------------------
    # Protocol property
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "perplexity"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return an httpx.Client, creating lazily if needed."""
        if self._session is not None:
            return self._session
        httpx = _import_httpx()
        self._session = httpx.Client(timeout=60)
        return self._session

    def _build_user_prompt(self, symbols: list[str], asset_class: str) -> str:
        asset_hint = f" ({asset_class} assets)" if asset_class else ""
        return (
            f"Analyse the following symbols{asset_hint} and return the "
            f"structured event JSON:\n{', '.join(symbols)}"
        )

    def _call_api(self, symbols: list[str], asset_class: str) -> dict[str, Any]:
        """Send the chat-completion request and return the raw JSON body."""
        if not self._api_key:
            raise SpectraQuantError(
                "PerplexityNewsProvider: no API key configured. "
                "Set the PERPLEXITY_API_KEY environment variable or pass api_key= to the constructor."
            )

        client = self._get_client()
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_user_prompt(symbols, asset_class),
                },
            ],
        }

        response = client.post(url, json=payload, headers=headers)

        if hasattr(response, "status_code") and response.status_code != 200:
            raise SpectraQuantError(
                f"PerplexityNewsProvider: HTTP {response.status_code}: "
                f"{getattr(response, 'text', '')[:300]}"
            )

        if hasattr(response, "json"):
            return response.json() if callable(response.json) else response.json
        raise SpectraQuantError("PerplexityNewsProvider: unexpected response type")

    @staticmethod
    def _parse_records(
        body: dict[str, Any],
        asset_class: str,
    ) -> list[NewsIntelligenceRecord]:
        """Extract :class:`NewsIntelligenceRecord` instances from the API body."""
        # Navigate to the assistant message content.
        choices = body.get("choices") or []
        if not choices:
            logger.warning("PerplexityNewsProvider: response has no choices")
            return []

        content = (choices[0].get("message") or {}).get("content", "")
        if not content:
            logger.warning("PerplexityNewsProvider: assistant message is empty")
            return []

        # Strip markdown code fences if present.
        text = content.strip()
        if text.startswith("```"):
            # Remove opening fence (possibly ```json)
            text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "PerplexityNewsProvider: could not parse JSON from response content"
            )
            return []

        if not isinstance(items, list):
            items = [items]

        records: list[NewsIntelligenceRecord] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                record = NewsIntelligenceRecord(
                    canonical_symbol=str(item.get("canonical_symbol", "")).upper(),
                    asset=asset_class or str(item.get("asset", "")),
                    timestamp=str(
                        item.get("timestamp", "")
                        or datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
                    ),
                    event_type=str(item.get("event_type", "general")),
                    sentiment_score=float(item.get("sentiment_score", 0.0)),
                    impact_score=float(item.get("impact_score", 0.0)),
                    article_count=int(item.get("article_count", 1)),
                    source_urls=list(item.get("source_urls") or []),
                    confidence=float(item.get("confidence", 0.0)),
                    rationale=str(item.get("rationale", "")),
                    provider="perplexity",
                    raw_response=item,
                )
                records.append(record)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "PerplexityNewsProvider: skipping malformed item: %s", exc
                )
        return records

    # ------------------------------------------------------------------
    # Public API — satisfies NewsIntelligenceProvider protocol
    # ------------------------------------------------------------------

    def fetch_intelligence(
        self,
        symbols: list[str],
        *,
        asset_class: str = "",
        max_results: int = 10,
    ) -> list[NewsIntelligenceRecord]:
        """Fetch news intelligence for one or more canonical symbols.

        Args:
            symbols:     List of canonical tickers.
            asset_class: Optional asset-class hint.
            max_results: Soft limit on results per symbol (advisory to the LLM).

        Returns:
            List of :class:`NewsIntelligenceRecord` instances.
        """
        if not symbols:
            return []

        body = self._call_api(symbols, asset_class)
        return self._parse_records(body, asset_class)
