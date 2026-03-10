"""Map crypto coin/project names to trading symbols."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Mapping of common names/aliases to canonical trading symbols
_CRYPTO_ALIASES: dict[str, str] = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "ether": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "cardano": "ADA",
    "ada": "ADA",
    "polkadot": "DOT",
    "dot": "DOT",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "chainlink": "LINK",
    "link": "LINK",
    "polygon": "MATIC",
    "matic": "MATIC",
    "uniswap": "UNI",
    "uni": "UNI",
    "aave": "AAVE",
    "litecoin": "LTC",
    "ltc": "LTC",
    "ripple": "XRP",
    "xrp": "XRP",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "shiba inu": "SHIB",
    "shib": "SHIB",
    "binance coin": "BNB",
    "bnb": "BNB",
    "tron": "TRX",
    "trx": "TRX",
    "near protocol": "NEAR",
    "near": "NEAR",
    "arbitrum": "ARB",
    "arb": "ARB",
    "optimism": "OP",
    "cosmos": "ATOM",
    "atom": "ATOM",
    "maker": "MKR",
    "mkr": "MKR",
    "compound": "COMP",
    "comp": "COMP",
}


def extract_symbols(
    text: str,
    known_symbols: set[str] | None = None,
) -> list[str]:
    """Extract crypto ticker symbols mentioned in *text*.

    Parameters
    ----------
    text : str
        Article title, summary, or body text.
    known_symbols : set of str, optional
        Additional valid symbols to recognize.

    Returns
    -------
    list of str
        Unique symbols found, ordered by first appearance.
    """
    if not text:
        return []

    text_lower = text.lower()
    found: dict[str, int] = {}  # symbol -> first position

    # Check aliases
    for alias, symbol in _CRYPTO_ALIASES.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        match = re.search(pattern, text_lower)
        if match and symbol not in found:
            found[symbol] = match.start()

    # Check for $TICKER patterns (e.g., $BTC, $ETH)
    for m in re.finditer(r"\$([A-Z]{2,10})", text):
        sym = m.group(1)
        if sym not in found:
            found[sym] = m.start()

    # Check known symbols
    if known_symbols:
        for sym in known_symbols:
            if re.search(r"\b" + re.escape(sym) + r"\b", text, re.IGNORECASE):
                if sym not in found:
                    found[sym] = text_lower.find(sym.lower())

    result = sorted(found, key=lambda s: found[s])
    return result


def map_articles_to_symbols(
    articles: list[dict[str, Any]],
    known_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Annotate each article with extracted symbols.

    Adds a ``symbols`` key to each article dict.

    Parameters
    ----------
    articles : list of dict
        Articles with ``title`` and ``summary`` keys.
    known_symbols : set of str, optional
        Extra symbols to recognize.

    Returns
    -------
    list of dict
        Same articles with ``symbols`` list added.
    """
    for art in articles:
        text = (art.get("title", "") + " " + art.get("summary", "")).strip()
        art["symbols"] = extract_symbols(text, known_symbols)
    mapped = sum(1 for a in articles if a.get("symbols"))
    logger.info("Mapped symbols for %d / %d articles", mapped, len(articles))
    return articles
