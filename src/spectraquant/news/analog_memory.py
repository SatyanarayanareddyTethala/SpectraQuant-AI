"""Historical analog memory for event-impact retrieval.

Implements Section 6 of the SpectraQuant-AI system specification:
*Retrieve past events similar to a query event and use their observed
outcomes to calibrate probabilistic market-impact predictions.*

Similarity is computed across five dimensions:
- Semantic event similarity (text embedding cosine similarity)
- Sector similarity (exact match bonus)
- Volatility regime similarity (normalised distance)
- Liquidity state similarity (normalised distance)
- Trend regime similarity (signed agreement)

Usage
-----
>>> mem = AnalogMemory()
>>> mem.add(event_record)
>>> analogs = mem.retrieve(query_event, top_k=5)
>>> calibrated = mem.calibrate_prediction(0.03, analogs)
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Similarity weights (sum to 1.0) – tunable via config
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 0.40,
    "sector": 0.20,
    "volatility_regime": 0.15,
    "liquidity_state": 0.15,
    "trend_regime": 0.10,
}


@dataclass
class EventRecord:
    """A stored event with its observed market outcomes.

    Required fields
    ---------------
    event_id : str
        Unique identifier (e.g. SHA256 of ticker + event_type + date).
    event_type : str
        Ontology type, e.g. ``"earnings_beat"``, ``"regulatory_penalty"``.
    ticker : str
        Primary affected ticker.
    sector : str
        GICS sector of the ticker.
    event_text : str
        Canonical description or headline used for embedding.
    volatility_regime : float
        Annualised realised volatility at event time (e.g. 0.25).
    liquidity_state : float
        ADV-normalised spread proxy at event time (lower = more liquid).
    trend_regime : float
        Signed trend strength at event time (positive = uptrend, negative = downtrend).

    Outcome fields (filled post-event)
    ------------------------------------
    observed_return_intraday : float | None
        Observed abnormal return within the same session (≈30 min–2 h).
    observed_return_shortterm : float | None
        Observed abnormal return over 1–3 trading days.
    observed_return_medium : float | None
        Observed abnormal return over 1–4 weeks.

    Internal
    --------
    embedding : np.ndarray | None
        Dense vector for ``event_text``; populated lazily by :class:`AnalogMemory`.
    metadata : dict
        Free-form storage for additional slots.
    """

    event_id: str
    event_type: str
    ticker: str
    sector: str
    event_text: str
    volatility_regime: float
    liquidity_state: float
    trend_regime: float

    observed_return_intraday: float | None = None
    observed_return_shortterm: float | None = None
    observed_return_medium: float | None = None

    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def make_id(cls, ticker: str, event_type: str, date_str: str) -> str:
        """Create a deterministic event ID from key fields."""
        raw = f"{ticker}:{event_type}:{date_str}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class AnalogResult:
    """One retrieved analog with its similarity score and observed outcome."""

    record: EventRecord
    similarity: float
    similarity_breakdown: dict[str, float]


class AnalogMemory:
    """In-memory store of historical event records with similarity retrieval.

    Parameters
    ----------
    weights : dict[str, float], optional
        Per-dimension similarity weights.  Must sum to 1.0 (approximately).
        Keys: ``semantic``, ``sector``, ``volatility_regime``,
        ``liquidity_state``, ``trend_regime``.
    volatility_scale : float
        Normalisation factor for volatility regime differences (annualised vol).
        Default 0.20 (≈ typical market annual vol).
    liquidity_scale : float
        Normalisation factor for liquidity state differences (spread proxy).
        Default 0.01.
    trend_scale : float
        Normalisation factor for trend regime differences.
        Default 1.0 (standardised z-score expected).
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        volatility_scale: float = 0.20,
        liquidity_scale: float = 0.01,
        trend_scale: float = 1.0,
    ) -> None:
        self._weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            self._weights.update(weights)
        self._volatility_scale = volatility_scale
        self._liquidity_scale = liquidity_scale
        self._trend_scale = trend_scale
        self._records: dict[str, EventRecord] = {}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add(self, record: EventRecord) -> None:
        """Add a single event record to the memory store."""
        self._records[record.event_id] = record
        logger.debug("AnalogMemory: added event %s (%s)", record.event_id, record.event_type)

    def add_batch(self, records: list[EventRecord]) -> None:
        """Add multiple records at once."""
        for rec in records:
            self.add(rec)

    def remove(self, event_id: str) -> bool:
        """Remove a record by ID; returns True if found and removed."""
        if event_id in self._records:
            del self._records[event_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _semantic_similarity(
        self, query: EventRecord, candidate: EventRecord
    ) -> float:
        """Cosine similarity between event text embeddings.

        Falls back to Jaccard word-overlap if embeddings are not set.
        """
        q_emb = query.embedding
        c_emb = candidate.embedding

        if q_emb is not None and c_emb is not None:
            q_norm = np.linalg.norm(q_emb)
            c_norm = np.linalg.norm(c_emb)
            if q_norm == 0.0 or c_norm == 0.0:
                return 0.0
            return float(np.dot(q_emb, c_emb) / (q_norm * c_norm))

        # Fallback: Jaccard overlap on lower-cased tokens
        q_tokens = set(query.event_text.lower().split())
        c_tokens = set(candidate.event_text.lower().split())
        if not q_tokens and not c_tokens:
            return 1.0
        intersection = len(q_tokens & c_tokens)
        union = len(q_tokens | c_tokens)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _sector_similarity(query: EventRecord, candidate: EventRecord) -> float:
        """1.0 if same sector, 0.0 otherwise (binary)."""
        return 1.0 if query.sector == candidate.sector else 0.0

    def _volatility_similarity(
        self, query: EventRecord, candidate: EventRecord
    ) -> float:
        """Gaussian similarity on normalised volatility distance."""
        diff = abs(query.volatility_regime - candidate.volatility_regime)
        return math.exp(-0.5 * (diff / max(self._volatility_scale, 1e-8)) ** 2)

    def _liquidity_similarity(
        self, query: EventRecord, candidate: EventRecord
    ) -> float:
        """Gaussian similarity on normalised liquidity distance."""
        diff = abs(query.liquidity_state - candidate.liquidity_state)
        return math.exp(-0.5 * (diff / max(self._liquidity_scale, 1e-8)) ** 2)

    def _trend_similarity(
        self, query: EventRecord, candidate: EventRecord
    ) -> float:
        """Gaussian similarity on normalised trend strength distance."""
        diff = abs(query.trend_regime - candidate.trend_regime)
        return math.exp(-0.5 * (diff / max(self._trend_scale, 1e-8)) ** 2)

    def _composite_similarity(
        self, query: EventRecord, candidate: EventRecord
    ) -> tuple[float, dict[str, float]]:
        """Compute weighted composite similarity and per-dimension breakdown."""
        w = self._weights
        dims: dict[str, float] = {
            "semantic": self._semantic_similarity(query, candidate),
            "sector": self._sector_similarity(query, candidate),
            "volatility_regime": self._volatility_similarity(query, candidate),
            "liquidity_state": self._liquidity_similarity(query, candidate),
            "trend_regime": self._trend_similarity(query, candidate),
        }
        total = sum(w.get(k, 0.0) * v for k, v in dims.items())
        return total, dims

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: EventRecord,
        top_k: int = 10,
        min_similarity: float = 0.0,
        exclude_same_ticker: bool = False,
    ) -> list[AnalogResult]:
        """Retrieve the *top_k* most similar historical events.

        Parameters
        ----------
        query : EventRecord
            The current event to match against.
        top_k : int
            Number of analogs to return.
        min_similarity : float
            Discard candidates with composite similarity below this value.
        exclude_same_ticker : bool
            If True, skip records from the same ticker (avoids look-ahead bias
            in live use; keep False for back-test calibration).

        Returns
        -------
        list[AnalogResult]
            Sorted by descending composite similarity.
        """
        scored: list[AnalogResult] = []

        for rec in self._records.values():
            if rec.event_id == query.event_id:
                continue
            if exclude_same_ticker and rec.ticker == query.ticker:
                continue

            score, breakdown = self._composite_similarity(query, rec)
            if score >= min_similarity:
                scored.append(AnalogResult(record=rec, similarity=score, similarity_breakdown=breakdown))

        scored.sort(key=lambda x: x.similarity, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_prediction(
        self,
        raw_prediction: float,
        analogs: list[AnalogResult],
        horizon: str = "intraday",
        blend_weight: float = 0.3,
    ) -> dict[str, float]:
        """Blend a model prediction with the analog outcome distribution.

        The calibrated return is a similarity-weighted blend of:
        * ``raw_prediction`` (1 - blend_weight)
        * mean analog return (blend_weight)

        Parameters
        ----------
        raw_prediction : float
            Point estimate from the primary model (e.g. expected return).
        analogs : list[AnalogResult]
            Retrieved analogs (from :meth:`retrieve`).
        horizon : str
            One of ``"intraday"``, ``"shortterm"``, ``"medium"``.
        blend_weight : float
            How much to weight the analog mean vs the raw prediction (0–1).

        Returns
        -------
        dict with keys:
            ``calibrated_return``  – blended expected return
            ``analog_mean``        – similarity-weighted mean of analog outcomes
            ``analog_std``         – similarity-weighted std of analog outcomes
            ``analog_count``       – number of analogs with observed outcomes
            ``blend_weight``       – the blend weight used
        """
        outcome_attr = {
            "intraday": "observed_return_intraday",
            "shortterm": "observed_return_shortterm",
            "medium": "observed_return_medium",
        }.get(horizon, "observed_return_intraday")

        weights: list[float] = []
        outcomes: list[float] = []

        for analog in analogs:
            val = getattr(analog.record, outcome_attr, None)
            if val is not None:
                weights.append(analog.similarity)
                outcomes.append(val)

        if not outcomes:
            return {
                "calibrated_return": raw_prediction,
                "analog_mean": float("nan"),
                "analog_std": float("nan"),
                "analog_count": 0,
                "blend_weight": 0.0,
            }

        w_arr = np.array(weights)
        o_arr = np.array(outcomes)
        w_norm = w_arr / w_arr.sum()

        analog_mean = float(np.dot(w_norm, o_arr))
        analog_variance = float(np.dot(w_norm, (o_arr - analog_mean) ** 2))
        analog_std = math.sqrt(analog_variance)

        calibrated = (1.0 - blend_weight) * raw_prediction + blend_weight * analog_mean

        return {
            "calibrated_return": calibrated,
            "analog_mean": analog_mean,
            "analog_std": analog_std,
            "analog_count": len(outcomes),
            "blend_weight": blend_weight,
        }


__all__ = ["AnalogMemory", "EventRecord", "AnalogResult"]
