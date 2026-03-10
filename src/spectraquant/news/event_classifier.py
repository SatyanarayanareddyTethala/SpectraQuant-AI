"""Event classifier for news articles.

Implements a rule-based / keyword-driven event classification pipeline
that maps article text to a structured event representation without
requiring external LLM or FinBERT dependencies.

The classifier outputs a :class:`ClassificationResult` containing:
    event_type  – ontology type from ``event_ontology.EVENT_REGISTRY``
    sentiment   – ``"positive"``, ``"negative"``, ``"neutral"``
    magnitude   – estimated impact magnitude [0.0, 1.0]
    uncertainty – model confidence that the classification is correct [0.0, 1.0]

The event-type vocabulary matches the keys in
:data:`spectraquant.news.event_ontology.EVENT_REGISTRY`, with the addition
of ``"unknown"`` for articles that do not match any category.

Usage
-----
>>> from spectraquant.news.event_classifier import EventClassifier
>>> clf = EventClassifier()
>>> result = clf.classify({"title": "Infosys Q4 earnings beat estimates", "content": ""})
>>> result.event_type
'earnings'
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

__all__ = ["EventClassifier", "ClassificationResult"]

# Sentiment hint constants
_SENTIMENT_POSITIVE: int = 1
_SENTIMENT_NEGATIVE: int = -1
_SENTIMENT_NEUTRAL: int = 0

# ---------------------------------------------------------------------------
# Event keyword rules
# Each entry: (event_type, sentiment_hint, keyword_patterns, weight)
# ---------------------------------------------------------------------------

# (event_type, sentiment_hint[_SENTIMENT_POSITIVE/_SENTIMENT_NEGATIVE/_SENTIMENT_NEUTRAL], keywords)
_RULES: List[Tuple[str, int, List[str]]] = [
    # Earnings – positive signals
    ("earnings", _SENTIMENT_POSITIVE, [
        r"earnings beat", r"profit beat", r"earnings surpass", r"beat estimate",
        r"beat consensus", r"above expectation", r"net profit rise", r"net profit jump",
        r"revenue beat", r"strong quarter", r"record profit", r"profit soar",
    ]),
    # Earnings – negative signals
    ("earnings", _SENTIMENT_NEGATIVE, [
        r"earnings miss", r"profit miss", r"below estimate", r"below consensus",
        r"net profit fall", r"net profit drop", r"revenue miss", r"weak quarter",
        r"quarterly loss", r"profit declin",
    ]),
    # Earnings – neutral/general
    ("earnings", _SENTIMENT_NEUTRAL, [
        r"quarterly result", r"annual result", r"q[1-4] result", r"earnings report",
        r"financial result", r"earnings release", r"eps report",
    ]),
    # Guidance
    ("guidance", _SENTIMENT_POSITIVE, [
        r"raise.{0,10}guidance", r"upgrade.{0,10}guidance", r"raise.{0,10}forecast",
        r"positive outlook", r"strong guidance",
    ]),
    ("guidance", _SENTIMENT_NEGATIVE, [
        r"cut.{0,10}guidance", r"lower.{0,10}guidance", r"slash.{0,10}guidance",
        r"downgrade.{0,10}guidance", r"weak guidance", r"warn.{0,10}revenue",
        r"profit warning", r"guidance cut",
    ]),
    ("guidance", _SENTIMENT_NEUTRAL, [r"guidance", r"outlook", r"forecast update"]),
    # Macro
    ("macro", _SENTIMENT_POSITIVE, [
        r"rate cut", r"stimulus package", r"gdp beat", r"cpi fall", r"inflation drop",
        r"rbi cut", r"fed cut",
    ]),
    ("macro", _SENTIMENT_NEGATIVE, [
        r"rate hike", r"interest rate rise", r"gdp miss", r"cpi surge", r"inflation rise",
        r"rbi hike", r"fed hike",
    ]),
    ("macro", _SENTIMENT_NEUTRAL, [
        r"rbi policy", r"fed meeting", r"monetary policy", r"repo rate",
        r"inflation data", r"gdp data", r"budget", r"fiscal",
    ]),
    # M&A
    ("m_and_a", _SENTIMENT_POSITIVE, [
        r"acqui", r"merge[rd]", r"takeover", r"buyout", r"deal agreement",
        r"joint venture", r"strategic partner",
    ]),
    ("m_and_a", _SENTIMENT_NEGATIVE, [
        r"deal collapse", r"merger fail", r"acquisition cancel", r"bid reject",
    ]),
    ("m_and_a", _SENTIMENT_NEUTRAL, [r"stake sale", r"divestiture", r"demerger"]),
    # Regulatory
    ("regulatory", _SENTIMENT_NEGATIVE, [
        r"sebi fine", r"fine impose", r"penalty impose", r"regulatory action",
        r"sebi order", r"ban impos", r"investigation launch",
    ]),
    ("regulatory", _SENTIMENT_POSITIVE, [
        r"regulatory approv", r"nod receiv", r"clearance receiv",
        r"fda approv", r"drug approv",
    ]),
    ("regulatory", _SENTIMENT_NEUTRAL, [r"regulator", r"compliance", r"sebi", r"rbi circular"]),
    # Operations disruption / supply chain
    ("operations_disruption", _SENTIMENT_NEGATIVE, [
        r"plant shut", r"factory clos", r"supply chain disrupt", r"cyber attack",
        r"strike action", r"lockout", r"natural disaster",
    ]),
    ("operations_disruption", _SENTIMENT_NEUTRAL, [r"supply chain", r"operations update"]),
    # Risk events
    ("risk", _SENTIMENT_NEGATIVE, [
        r"credit downgrade", r"rating cut", r"fraud allegation", r"scam",
        r"liquidity crisis", r"default risk", r"ceo resign", r"cfo resign",
        r"management change",
    ]),
    ("risk", _SENTIMENT_POSITIVE, [r"credit upgrade", r"rating raise"]),
    ("risk", _SENTIMENT_NEUTRAL, [r"risk disclosure", r"litigation", r"legal challenge"]),
    # Corporate actions
    ("corporate_action", _SENTIMENT_POSITIVE, [
        r"dividend declared", r"dividend hike", r"bonus share", r"buyback announce",
        r"stock split",
    ]),
    ("corporate_action", _SENTIMENT_NEUTRAL, [r"rights issue", r"ex-dividend", r"record date"]),
    # Product / competition / growth
    ("guidance", _SENTIMENT_POSITIVE, [
        r"product launch", r"new product", r"market expansion",
        r"contract win", r"order win",
    ]),
    ("risk", _SENTIMENT_NEGATIVE, [r"competition intensif", r"market share loss"]),
]

# Magnitude signals (phrase → boost)
_MAGNITUDE_KEYWORDS: List[Tuple[str, float]] = [
    (r"record", 0.2),
    (r"historic", 0.2),
    (r"massive", 0.15),
    (r"surge", 0.15),
    (r"plunge", 0.15),
    (r"soar", 0.15),
    (r"crumble", 0.15),
    (r"billion", 0.1),
    (r"crore", 0.08),
    (r"significant", 0.05),
    (r"unexpected", 0.1),
]

# Source credibility lookup (lower-cased source name → rank [0, 1])
_SOURCE_RANK: Dict[str, float] = {
    "reuters": 0.95,
    "bloomberg": 0.95,
    "financial times": 0.90,
    "economic times": 0.85,
    "business standard": 0.85,
    "mint": 0.80,
    "livemint": 0.80,
    "moneycontrol": 0.78,
    "ndtv profit": 0.75,
    "cnbc": 0.80,
    "cnbctv18": 0.78,
    "the hindu businessline": 0.80,
    "businessline": 0.80,
    "hindu businessline": 0.80,
    "pti": 0.75,
    "ians": 0.72,
    "unknown": 0.40,
}

_DEFAULT_SOURCE_RANK = 0.50


@dataclass
class ClassificationResult:
    """Output of the event classifier for a single article."""

    event_type: str = "unknown"
    sentiment: str = "neutral"
    magnitude: float = 0.0
    uncertainty: float = 1.0
    matched_rules: List[str] = field(default_factory=list)


class EventClassifier:
    """Rule-based event classifier.

    Parameters
    ----------
    min_magnitude : float
        Minimum base magnitude for any matched article (default 0.1).
    """

    def __init__(self, min_magnitude: float = 0.1) -> None:
        self._min_magnitude = min_magnitude
        # Pre-compile patterns for efficiency
        self._compiled: List[Tuple[str, int, re.Pattern]] = [
            (event_type, sentiment_hint, re.compile(pattern, re.IGNORECASE))
            for event_type, sentiment_hint, patterns in _RULES
            for pattern in patterns
        ]
        self._mag_compiled: List[Tuple[re.Pattern, float]] = [
            (re.compile(p, re.IGNORECASE), boost)
            for p, boost in _MAGNITUDE_KEYWORDS
        ]

    def classify(self, article: dict) -> ClassificationResult:
        """Classify a single article (dict with ``title``, ``content``, etc.).

        Parameters
        ----------
        article : dict
            Must have at least ``title``; ``description`` and ``content``
            are used if present.

        Returns
        -------
        ClassificationResult
        """
        text = " ".join(filter(None, [
            str(article.get("title") or ""),
            str(article.get("description") or ""),
            str(article.get("content") or ""),
        ])).lower()

        if not text.strip():
            return ClassificationResult()

        # Count matches per (event_type, sentiment_hint)
        vote_map: Dict[str, Dict[int, int]] = {}
        matched_rules: List[str] = []

        for event_type, sentiment_hint, pattern in self._compiled:
            if pattern.search(text):
                vote_map.setdefault(event_type, {
                    _SENTIMENT_NEUTRAL: 0,
                    _SENTIMENT_POSITIVE: 0,
                    _SENTIMENT_NEGATIVE: 0,
                })
                vote_map[event_type][sentiment_hint] += 1
                matched_rules.append(f"{event_type}:{sentiment_hint}:{pattern.pattern}")

        if not vote_map:
            return ClassificationResult()

        # Choose event_type with most total votes
        best_event = max(vote_map, key=lambda et: sum(vote_map[et].values()))
        votes = vote_map[best_event]
        total_votes = sum(votes.values())

        pos = votes.get(_SENTIMENT_POSITIVE, 0)
        neg = votes.get(_SENTIMENT_NEGATIVE, 0)
        if pos > neg:
            sentiment = "positive"
        elif neg > pos:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Uncertainty: higher when multiple event types match
        n_types = len(vote_map)
        uncertainty = 1.0 / n_types if n_types > 0 else 1.0
        uncertainty = max(0.0, min(1.0, uncertainty))

        # Magnitude: base from vote count + keyword boosts
        base_magnitude = min(1.0, total_votes * 0.15 + self._min_magnitude)
        boost = sum(b for p, b in self._mag_compiled if p.search(text))
        magnitude = min(1.0, base_magnitude + boost)

        return ClassificationResult(
            event_type=best_event,
            sentiment=sentiment,
            magnitude=magnitude,
            uncertainty=uncertainty,
            matched_rules=matched_rules[:10],
        )

    @staticmethod
    def score_source(source_name: str) -> float:
        """Return credibility rank [0, 1] for a source name."""
        if not source_name:
            return _DEFAULT_SOURCE_RANK
        key = source_name.strip().lower()
        # Exact match
        if key in _SOURCE_RANK:
            return _SOURCE_RANK[key]
        # Partial match
        for name, rank in _SOURCE_RANK.items():
            if name in key or key in name:
                return rank
        return _DEFAULT_SOURCE_RANK
