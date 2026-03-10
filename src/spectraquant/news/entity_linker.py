"""Entity linker for news articles.

Maps entity strings extracted from article text to tickers in the
trading universe, and resolves competitor relationships.

The linker uses the same universe mapping structure as
:func:`spectraquant.news.universe_builder.load_universe_mapping` so it
can be initialised from an already-loaded mapping dict.

Usage
-----
>>> from spectraquant.news.entity_linker import EntityLinker
>>> linker = EntityLinker(universe_mapping)
>>> result = linker.link({"title": "Infosys wins cloud deal", "content": ""})
>>> result.tickers   # ["INFY.NS"]
>>> result.competitors  # peer tickers from the sector
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["EntityLinker", "LinkResult"]

# ---------------------------------------------------------------------------
# Simple competitor graph – sector → tickers list
# Used as a fallback when no explicit competitor map is supplied.
# ---------------------------------------------------------------------------

_SECTOR_PEERS: Dict[str, List[str]] = {
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Metals": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS"],
    "Telecom": ["BHARTIARTL.NS", "IDEA.NS", "TATACOMM.NS"],
}

# Reverse: ticker → sector
_TICKER_TO_SECTOR: Dict[str, str] = {
    t: sector for sector, tickers in _SECTOR_PEERS.items() for t in tickers
}


@dataclass
class LinkResult:
    """Result of entity linking for a single article."""

    tickers: List[str] = field(default_factory=list)
    competitors: List[str] = field(default_factory=list)
    entity_strings: List[str] = field(default_factory=list)


class EntityLinker:
    """Links article text to tickers via company name / alias / token matching.

    Parameters
    ----------
    universe_mapping : dict
        As returned by
        :func:`spectraquant.news.universe_builder.load_universe_mapping`:
        keys ``tickers``, ``ticker_to_company``, ``aliases``.
    competitor_map : dict, optional
        Explicit ``{ticker: [peer_ticker, ...]}``.  If not supplied, uses
        the sector-based ``_SECTOR_PEERS`` fallback.
    max_competitors : int
        Maximum number of competitors to return per ticker (default 4).
    """

    def __init__(
        self,
        universe_mapping: Dict[str, Any],
        competitor_map: Optional[Dict[str, List[str]]] = None,
        max_competitors: int = 4,
    ) -> None:
        self._tickers: List[str] = universe_mapping.get("tickers", [])
        self._ticker_to_company: Dict[str, str] = universe_mapping.get("ticker_to_company", {})
        self._aliases: Dict[str, str] = universe_mapping.get("aliases", {})
        self._competitor_map: Dict[str, List[str]] = competitor_map or {}
        self._max_competitors = max_competitors

        # Pre-build per-ticker token sets for fast matching
        self._ticker_tokens: Dict[str, List[str]] = {}
        for ticker in self._tickers:
            company = self._ticker_to_company.get(ticker, "")
            tokens = [t for t in company.lower().split() if len(t) > 2]
            self._ticker_tokens[ticker] = tokens

    def link(self, article: dict) -> LinkResult:
        """Link a single article to tickers and competitors.

        Parameters
        ----------
        article : dict
            Must have at least ``title``; uses ``description`` and
            ``content`` if present.

        Returns
        -------
        LinkResult
        """
        text = " ".join(filter(None, [
            str(article.get("title") or ""),
            str(article.get("description") or ""),
            str(article.get("content") or ""),
        ]))
        text_lower = text.lower()
        text_upper = text.upper()

        matched_tickers: List[str] = []
        entity_strings: List[str] = []

        for ticker in self._tickers:
            ticker_clean = ticker.replace(".NS", "").replace(".L", "").upper()

            # 1) Ticker token match
            if re.search(r"\b" + re.escape(ticker_clean) + r"\b", text_upper):
                matched_tickers.append(ticker)
                entity_strings.append(ticker_clean)
                continue

            # 2) Company name token match
            tokens = self._ticker_tokens.get(ticker, [])
            if tokens and all(t in text_lower for t in tokens):
                company = self._ticker_to_company.get(ticker, ticker_clean)
                matched_tickers.append(ticker)
                entity_strings.append(company)
                continue

            # 3) Alias match
            for alias, alias_ticker in self._aliases.items():
                if alias_ticker == ticker and alias in text_lower:
                    matched_tickers.append(ticker)
                    entity_strings.append(alias)
                    break

        # Deduplicate while preserving order
        seen_tickers: set = set()
        unique_tickers: List[str] = []
        unique_entities: List[str] = []
        for ticker, entity in zip(matched_tickers, entity_strings):
            if ticker not in seen_tickers:
                seen_tickers.add(ticker)
                unique_tickers.append(ticker)
                unique_entities.append(entity)

        # Resolve competitors for matched tickers
        competitors: List[str] = []
        for ticker in unique_tickers:
            peers = self._resolve_competitors(ticker, unique_tickers)
            competitors.extend(peers)
        # Deduplicate competitors and exclude the matched tickers themselves
        comp_seen: set = set(unique_tickers)
        unique_competitors: List[str] = []
        for c in competitors:
            if c not in comp_seen:
                comp_seen.add(c)
                unique_competitors.append(c)
                if len(unique_competitors) >= self._max_competitors:
                    break

        return LinkResult(
            tickers=unique_tickers,
            competitors=unique_competitors,
            entity_strings=unique_entities,
        )

    def _resolve_competitors(self, ticker: str, exclude: List[str]) -> List[str]:
        """Return competitor tickers for *ticker*, excluding *exclude* list."""
        # 1) Explicit map
        if ticker in self._competitor_map:
            return [
                t for t in self._competitor_map[ticker]
                if t not in exclude
            ][: self._max_competitors]

        # 2) Sector-based fallback
        sector = _TICKER_TO_SECTOR.get(ticker)
        if sector:
            return [
                t for t in _SECTOR_PEERS.get(sector, [])
                if t != ticker and t not in exclude
            ][: self._max_competitors]

        # 3) No match
        return []
